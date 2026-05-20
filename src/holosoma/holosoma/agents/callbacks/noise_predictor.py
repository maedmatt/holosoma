"""Eval callback: run the noise predictor on each detected touchdown,
plot it live against foot vz, and dump per-tick traces to /tmp on exit.

The predictor is impact-only (trained on peak-detected windows), so the
prediction is held at zero between impacts and set to the model output
on each touchdown. Touchdown detection comes from TouchdownDetector; the
live plot runs in a subprocess via LivePlotProcess.
"""

from __future__ import annotations

import atexit
import queue
import time
from pathlib import Path
from typing import Any

from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import NoisePredictorConfig
from holosoma.managers.observation.terms.locomotion import get_projected_gravity
from holosoma.utils.live_plot import LivePlotProcess
from holosoma.utils.noise_predictor import NoisePredictor
from holosoma.utils.timeseries_log import TimeseriesLogger
from holosoma.utils.touchdown import TouchdownDetector


def _live_plot_worker(q, hz: int, window_s: float) -> None:
    """Rolling-window plot of (pred, vz_l, vz_r) tuples drained from q. None closes the window.

    Uses draw_idle()+flush_events()+sleep rather than plt.pause() because pause
    calls show(block=False) every iteration, which front-asserts the window
    (orderFront on Cocoa, focus_force on Tk) and steals focus from the
    MuJoCo viewer. This loop never calls show() after the initial fig.show().
    """
    import time as _time

    import matplotlib.pyplot as plt

    n = int(window_s * hz)
    t_buf: list[float] = []
    p_buf: list[float] = []
    vl_buf: list[float] = []
    vr_buf: list[float] = []

    fig, (ax_pred, ax_vz) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    (line_pred,) = ax_pred.plot([], [], color="black", lw=1)
    (line_vl,) = ax_vz.plot([], [], color="green", lw=1, label="L")
    (line_vr,) = ax_vz.plot([], [], color="orange", lw=1, label="R")
    ax_pred.set_ylabel("Predicted loudness")
    ax_vz.set_ylabel("Foot vz (m/s, world)")
    ax_vz.set_xlabel("Time (s)")
    ax_vz.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax_vz.legend(loc="lower right", fontsize=8)
    ax_pred.set_title(f"Noise predictor vs foot vz — {window_s:.0f}s window")
    fig.tight_layout()
    fig.show()

    while plt.fignum_exists(fig.number):
        # Drain everything available without blocking.
        sentinel = False
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                sentinel = True
                break
            t, p, vl, vr = item
            t_buf.append(t)
            p_buf.append(p)
            vl_buf.append(vl)
            vr_buf.append(vr)
        if sentinel:
            break

        if len(t_buf) > n:
            t_buf = t_buf[-n:]
            p_buf = p_buf[-n:]
            vl_buf = vl_buf[-n:]
            vr_buf = vr_buf[-n:]

        if t_buf:
            line_pred.set_data(t_buf, p_buf)
            line_vl.set_data(t_buf, vl_buf)
            line_vr.set_data(t_buf, vr_buf)
            ax_vz.set_xlim(t_buf[0], t_buf[-1] + 0.2)
            for ax, vals in ((ax_pred, p_buf), (ax_vz, vl_buf + vr_buf)):
                lo, hi = min(vals), max(vals)
                pad = max(0.1 * (hi - lo), 1e-3)
                ax.set_ylim(lo - pad, hi + pad)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        _time.sleep(0.05)

    plt.close("all")


class NoisePredictorCallback(RLEvalCallback):
    """Per-tick logger of (pred, vz, fz) with a subprocess-backed live plot.

    Knobs (class constants):
      POLICY_HZ      Policy/observation rate, must match training.
      PLOT_WINDOW_S  Width of the rolling x-axis in sim seconds.
      PLOT_QUEUE_MAX Max queued plot updates before drops kick in.
    """

    POLICY_HZ = 50
    PLOT_WINDOW_S = 2.0
    PLOT_QUEUE_MAX = 500
    UPRIGHT_BASE_Z_MIN = 0.65     # m; nominal pelvis is ~0.79
    UPRIGHT_GRAV_Z_MAX = -0.95    # body-frame gravity z; -1 = perfectly upright

    def __init__(self, config: NoisePredictorConfig, training_loop: Any = None) -> None:
        super().__init__(config, training_loop)
        self.env_id = config.env_id
        self.checkpoint_path = config.checkpoint_path
        self._predictor: NoisePredictor | None = None
        self._detector: TouchdownDetector | None = None
        self._plot: LivePlotProcess | None = None
        self._log = TimeseriesLogger()
        self._n_impacts = 0
        self._step = 0
        self._dumped = False

    def _env(self):
        return self.training_loop._unwrap_env()

    def on_pre_evaluate_policy(self) -> None:
        self._predictor = NoisePredictor(self.checkpoint_path, device=self.device)
        self._detector = TouchdownDetector(num_envs=1, num_feet=2)
        logger.info(f"Noise predictor loaded from {self.checkpoint_path}")
        self._log.clear()
        self._n_impacts = 0
        self._step = 0
        self._dumped = False
        atexit.register(self._dump_npz)

        self._plot = LivePlotProcess(
            _live_plot_worker,
            self.POLICY_HZ,
            self.PLOT_WINDOW_S,
            max_queue=self.PLOT_QUEUE_MAX,
        )
        self._plot.start()

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        env = self._env()
        eid = self.env_id

        # Buffer must stay current; cheap, runs every tick. Forward pass is
        # gated below so the model only runs on real (upright) touchdowns.
        ready = self._predictor.update_buffer(env, eid)
        if not ready:
            self._step += 1
            return actor_state

        foot_vel = env.simulator._rigid_body_vel[eid, env.feet_indices, :]   # [2, 3]
        foot_pos = env.simulator._rigid_body_pos[eid, env.feet_indices, :]   # [2, 3]
        foot_f = env.simulator.contact_forces[eid, env.feet_indices, :]      # [2, 3]
        vz = foot_vel[:, 2]
        fz = foot_f[:, 2]
        fired = self._detector.step(vz.unsqueeze(0), fz.unsqueeze(0))[0]
        fired_l, fired_r = bool(fired[0]), bool(fired[1])
        vz_l, vz_r = float(vz[0]), float(vz[1])
        fz_l, fz_r = float(fz[0]), float(fz[1])

        base_z = float(env.simulator.robot_root_states[eid, 2])
        gravity_b = get_projected_gravity(env)[eid]
        upright = base_z > self.UPRIGHT_BASE_Z_MIN and float(gravity_b[2]) < self.UPRIGHT_GRAV_Z_MAX

        is_impact = (fired_l or fired_r) and upright
        value = float(self._predictor.forward()[0]) if is_impact else 0.0

        t = self._step / self.POLICY_HZ
        self._log.append(
            t=t,
            pred=value,
            vz_l=vz_l, vz_r=vz_r, fz_l=fz_l, fz_r=fz_r,
            foot_vel=foot_vel,
            foot_pos=foot_pos,
            foot_force=foot_f,
            base_z=base_z,
            gravity_b=gravity_b,
            cmd=env.command_manager.commands[eid],
        )

        if is_impact:
            self._n_impacts += 1
            foot = "L" if fired_l else "R"
            logger.info(
                f"[noise] step {self._step:>6d} t={t:6.2f}s impact[{foot}]={value:.4f} "
                f"vz_L={vz_l:+.2f} vz_R={vz_r:+.2f} fz_L={fz_l:6.1f} fz_R={fz_r:6.1f}"
            )

        self._plot.publish((t, value, vz_l, vz_r))

        self._step += 1
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        self._dump_npz()
        self._plot.stop()

    def _dump_npz(self) -> None:
        """Write per-tick traces to /tmp. Idempotent — fires from post-eval and atexit."""
        if self._dumped or len(self._log) == 0:
            return
        out_path = Path("/tmp") / f"noise_predictor_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        n = self._log.save(out_path)
        logger.info(f"[noise] saved {n} ticks ({self._n_impacts} impacts) to {out_path}")
        self._dumped = True

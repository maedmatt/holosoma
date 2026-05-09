"""Eval callback that runs the noise predictor and overlays it on foot velocity.

Foot vz is the trivial baseline we want to beat: it's the kinematic feature
most directly tied to impact loudness. The model is impact-only (trained on
peak-detected windows), so the prediction is held at zero between impacts
and set to the model output on each per-foot vz zero-crossing from below.

Everything is logged per tick (pred, vz_{l,r}, fz_{l,r}). Filtering and
analysis happen offline from the npz dump.
"""

from __future__ import annotations

import atexit
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import NoisePredictorConfig
from holosoma.utils.noise_predictor import NoisePredictor


class NoisePredictorCallback(RLEvalCallback):
    """Per-tick logger of (pred, vz, fz) with a rolling-window live plot.

    Touchdown detection: per-foot vz zero-crossing from below. The foot is
    "armed" once its vz drops past -VZ_ARM_THR (it's actually descending),
    and fires on the first tick where vz crosses back to >= 0. The arming
    naturally prevents re-firing on the same descent and ignores
    intra-stance force flicker — both failure modes of a thresholded
    contact-force rising edge.

    Knobs (class constants):
      POLICY_HZ          Policy/observation rate, must match training.
      PLOT_WINDOW_S      Width of the rolling x-axis in sim seconds.
      PLOT_REDRAW_EVERY  Redraw every N policy ticks.
      VZ_ARM_THR         |vz| in m/s required to arm a touchdown. Lower =
                         catches softer descents, higher = rejects noise.
    """

    POLICY_HZ = 50
    PLOT_WINDOW_S = 2.0
    PLOT_REDRAW_EVERY = 5
    VZ_ARM_THR = 0.3

    def __init__(self, config: NoisePredictorConfig, training_loop: Any = None) -> None:
        super().__init__(config, training_loop)
        self.env_id = config.env_id
        self.checkpoint_path = config.checkpoint_path
        self._predictor: NoisePredictor | None = None
        self._t: list[float] = []
        self._pred: list[float] = []
        self._vz_l: list[float] = []
        self._vz_r: list[float] = []
        self._fz_l: list[float] = []
        self._fz_r: list[float] = []
        self._prev_vz_l = 0.0
        self._prev_vz_r = 0.0
        self._armed_l = False
        self._armed_r = False
        self._step = 0
        self._dumped = False
        self._fig = None
        self._axes = None
        self._line_pred = None
        self._line_vz_l = None
        self._line_vz_r = None

    def _env(self):
        return self.training_loop._unwrap_env()

    def on_pre_evaluate_policy(self) -> None:
        self._predictor = NoisePredictor(self.checkpoint_path, device=self.device)
        logger.info(f"Noise predictor loaded from {self.checkpoint_path}")
        for buf in (self._t, self._pred, self._vz_l, self._vz_r, self._fz_l, self._fz_r):
            buf.clear()
        self._prev_vz_l = 0.0
        self._prev_vz_r = 0.0
        self._armed_l = False
        self._armed_r = False
        self._step = 0
        self._dumped = False
        atexit.register(self._dump_npz)
        self._init_plot()

    def _init_plot(self) -> None:
        import matplotlib.pyplot as plt

        plt.ion()
        self._fig, self._axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax_y, ax_v = self._axes
        (self._line_pred,) = ax_y.plot([], [], color="black", lw=1)
        (self._line_vz_l,) = ax_v.plot([], [], color="green", lw=1, label="L")
        (self._line_vz_r,) = ax_v.plot([], [], color="orange", lw=1, label="R")
        ax_y.set_ylabel("Predicted loudness")
        ax_v.set_ylabel("Foot vz (m/s, world)")
        ax_v.set_xlabel("Time (s)")
        ax_v.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax_v.legend(loc="lower right", fontsize=8)
        ax_y.set_title(f"Noise predictor (impact spikes) vs foot vz baseline — {self.PLOT_WINDOW_S:.0f}s window")
        self._fig.tight_layout()
        self._fig.show()

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        if self._predictor is None:
            return actor_state

        env = self._env()
        eid = self.env_id

        pred = self._predictor.predict(env, eid)
        if pred is None:
            self._step += 1
            return actor_state

        vz = env.simulator._rigid_body_vel[eid, env.feet_indices, 2]
        fz = env.simulator.contact_forces[eid, env.feet_indices, 2]
        vz_l, vz_r = float(vz[0]), float(vz[1])
        fz_l, fz_r = float(fz[0]), float(fz[1])

        if vz_l < -self.VZ_ARM_THR:
            self._armed_l = True
        if vz_r < -self.VZ_ARM_THR:
            self._armed_r = True
        fired_l = self._armed_l and self._prev_vz_l < 0 <= vz_l
        fired_r = self._armed_r and self._prev_vz_r < 0 <= vz_r
        if fired_l:
            self._armed_l = False
        if fired_r:
            self._armed_r = False
        self._prev_vz_l, self._prev_vz_r = vz_l, vz_r

        is_impact = fired_l or fired_r
        value = float(pred[0]) if is_impact else 0.0

        t = self._step / self.POLICY_HZ
        self._t.append(t)
        self._pred.append(value)
        self._vz_l.append(vz_l)
        self._vz_r.append(vz_r)
        self._fz_l.append(fz_l)
        self._fz_r.append(fz_r)

        if is_impact:
            foot = "L" if fired_l else "R"
            logger.info(
                f"[noise] step {self._step:>6d} t={t:6.2f}s impact[{foot}]={value:.4f} "
                f"vz_L={vz_l:+.2f} vz_R={vz_r:+.2f} fz_L={fz_l:6.1f} fz_R={fz_r:6.1f}"
            )

        if self._step % self.PLOT_REDRAW_EVERY == 0:
            self._redraw()
        self._step += 1
        return actor_state

    def _redraw(self) -> None:
        if not self._t:
            return
        n = int(self.PLOT_WINDOW_S * self.POLICY_HZ)
        ts = self._t[-n:]
        self._line_pred.set_data(ts, self._pred[-n:])
        self._line_vz_l.set_data(ts, self._vz_l[-n:])
        self._line_vz_r.set_data(ts, self._vz_r[-n:])
        ax_y, ax_v = self._axes
        ax_v.set_xlim(ts[0], ts[-1] + 0.2)
        for ax, vals in ((ax_y, self._pred[-n:]), (ax_v, self._vz_l[-n:] + self._vz_r[-n:])):
            if not vals:
                continue
            lo, hi = min(vals), max(vals)
            pad = max(0.1 * (hi - lo), 1e-3)
            ax.set_ylim(lo - pad, hi + pad)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def on_post_evaluate_policy(self) -> None:
        self._dump_npz()

    def _dump_npz(self) -> None:
        """Write per-tick traces to /tmp. Idempotent — fires from post-eval and atexit."""
        if self._dumped or not self._t:
            return
        out_path = Path("/tmp") / f"noise_predictor_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez(
            out_path,
            t=np.array(self._t),
            pred=np.array(self._pred),
            vz_l=np.array(self._vz_l),
            vz_r=np.array(self._vz_r),
            fz_l=np.array(self._fz_l),
            fz_r=np.array(self._fz_r),
        )
        n_impacts = sum(1 for v in self._pred if v != 0.0)
        logger.info(f"[noise] saved {len(self._t)} ticks ({n_impacts} impacts) to {out_path}")
        self._dumped = True

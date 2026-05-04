"""Eval callback that runs the noise predictor and logs predictions on foot impact."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import NoisePredictorConfig
from holosoma.utils.noise_predictor import NoisePredictor


class NoisePredictorCallback(RLEvalCallback):
    """Run the predictor every tick; record/plot only on foot-contact rising edges.

    The buffer fills implicitly as predict() is called every step. A prediction
    is recorded when either foot transitions from no-contact to contact.
    """

    POLICY_HZ = 50
    PLOT_WINDOW_S = 10.0
    CONTACT_THRESHOLD_N = 1.0

    def __init__(self, config: NoisePredictorConfig, training_loop: Any = None) -> None:
        super().__init__(config, training_loop)
        self.env_id = config.env_id
        self.checkpoint_path = config.checkpoint_path
        self._predictor: NoisePredictor | None = None
        self._prev_contact = None
        self._times: list[float] = []
        self._preds: list[float] = []
        self._step = 0
        self._fig = None
        self._ax = None
        self._line = None

    def _env(self):
        return self.training_loop._unwrap_env()

    def on_pre_evaluate_policy(self) -> None:
        self._predictor = NoisePredictor(self.checkpoint_path, device=self.device)
        logger.info(f"Noise predictor loaded from {self.checkpoint_path}")
        self._times.clear()
        self._preds.clear()
        self._step = 0
        self._prev_contact = None
        self._init_plot()

    def _init_plot(self) -> None:
        import matplotlib.pyplot as plt

        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1, figsize=(7, 3))
        self._line, = self._ax.plot([], [], "ko", markersize=4)
        self._ax.set_xlabel("Time (s)")
        self._ax.set_ylabel("Predicted impact loudness")
        self._ax.set_title("Noise predictor (impact-gated)")
        self._fig.tight_layout()
        self._fig.show()

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        if self._predictor is None:
            return actor_state

        env = self._env()
        eid = self.env_id

        pred = self._predictor.predict(env, eid)

        contact = env.simulator.contact_forces[eid, env.feet_indices, 2] > self.CONTACT_THRESHOLD_N
        rising = contact if self._prev_contact is None else contact & ~self._prev_contact
        self._prev_contact = contact.clone()

        if pred is not None and bool(rising.any()):
            t = self._step / self.POLICY_HZ
            value = float(pred[0])
            self._times.append(t)
            self._preds.append(value)
            logger.info(f"[noise] step {self._step:>6d} t={t:6.2f}s pred={value:.4f}")
            self._update_plot(t)

        self._step += 1
        return actor_state

    def _update_plot(self, t_now: float) -> None:
        window_start = max(0.0, t_now - self.PLOT_WINDOW_S)
        while self._times and self._times[0] < window_start:
            self._times.pop(0)
            self._preds.pop(0)
        self._line.set_data(self._times, self._preds)
        self._ax.set_xlim(window_start, t_now + 0.5)
        if self._preds:
            lo, hi = min(self._preds), max(self._preds)
            pad = 0.1 * (hi - lo) if hi > lo else max(abs(hi) * 0.1, 1e-3)
            self._ax.set_ylim(lo - pad, hi + pad)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def on_post_evaluate_policy(self) -> None:
        if not self._preds:
            return
        a = np.array(self._preds)
        logger.info(f"[noise] {len(a)} impacts | mean={a.mean():.4f} min={a.min():.4f} max={a.max():.4f}")

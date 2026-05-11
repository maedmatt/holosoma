"""Scripted policy eval: drive a fixed velocity-command schedule and log
per-tick foot state to NPZ so different policies can be compared offline.
Edit SCHEDULE to change what the test exercises.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import PolicyEvalConfig
from holosoma.utils.timeseries_log import TimeseriesLogger


# (duration_s, vx, vy, vyaw)
# Translation-only schedule with idle stops between motions. Yaw is excluded
# because MuJoCo's plane friction (mu=1.0) saturates during sustained yaw and
# the stance foot slips, which is not representative of real-robot behavior.
SCHEDULE: list[tuple[float, float, float, float]] = [
    (3.0, 0.0, 0.0, 0.0),    # settle
    (5.0, 0.5, 0.0, 0.0),    # forward
    (2.0, 0.0, 0.0, 0.0),    # stop
    (5.0, -0.4, 0.0, 0.0),   # backward
    (2.0, 0.0, 0.0, 0.0),    # stop
    (5.0, 0.0, 0.3, 0.0),    # left strafe
    (2.0, 0.0, 0.0, 0.0),    # stop
    (5.0, 0.0, -0.3, 0.0),   # right strafe
    (3.0, 0.0, 0.0, 0.0),    # final settle
]

OUTPUT_NAME = "policy_eval.npz"
ENV_ID = 0
POLICY_HZ = 50


class PolicyEvalCallback(RLEvalCallback):
    """Drive locomotion commands from SCHEDULE and dump per-tick foot state."""

    def __init__(self, config: PolicyEvalConfig, training_loop: Any = None) -> None:
        super().__init__(config, training_loop)

        self._boundaries: list[int] = []
        cum = 0
        for d, *_ in SCHEDULE:
            cum += int(round(d * POLICY_HZ))
            self._boundaries.append(cum)
        self._total_steps = cum

        self._log = TimeseriesLogger()
        self._step = 0
        self._prev_foot_vel: torch.Tensor | None = None
        self._dumped = False

    def _env(self):
        return self.training_loop._unwrap_env()

    def _cmd_for_step(self, step: int) -> tuple[float, float, float]:
        for i, b in enumerate(self._boundaries):
            if step < b:
                _, vx, vy, vyaw = SCHEDULE[i]
                return vx, vy, vyaw
        return 0.0, 0.0, 0.0

    def on_pre_evaluate_policy(self) -> None:
        self._log.clear()
        self._step = 0
        self._prev_foot_vel = None
        self._dumped = False
        atexit.register(self._dump)
        logger.info(
            f"[policy_eval] schedule {self._total_steps} steps "
            f"({self._total_steps / POLICY_HZ:.1f}s); "
            f"pass --training.max-eval-steps {self._total_steps} to cover it"
        )

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        vx, vy, vyaw = self._cmd_for_step(self._step)
        cmd = self._env().command_manager.commands
        cmd[ENV_ID, 0] = vx
        cmd[ENV_ID, 1] = vy
        cmd[ENV_ID, 2] = vyaw
        return actor_state

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        env = self._env()
        dt = 1.0 / POLICY_HZ

        foot_vel = env.simulator._rigid_body_vel[ENV_ID, env.feet_indices, :]
        foot_pos = env.simulator._rigid_body_pos[ENV_ID, env.feet_indices, :]
        foot_f = env.simulator.contact_forces[ENV_ID, env.feet_indices, :]

        if self._prev_foot_vel is None:
            foot_acc = torch.zeros_like(foot_vel)
        else:
            foot_acc = (foot_vel - self._prev_foot_vel) / dt
        self._prev_foot_vel = foot_vel.clone()

        root = env.simulator.robot_root_states[ENV_ID]
        gait = env.command_manager.get_state("locomotion_gait")

        self._log.append(
            t=self._step / POLICY_HZ,
            cmd=env.command_manager.commands[ENV_ID],
            foot_pos=foot_pos,
            foot_vel=foot_vel,
            foot_acc=foot_acc,
            foot_force=foot_f,
            base_pos=root[:3],
            base_quat=root[3:7],
            base_lin_vel=root[7:10],
            base_ang_vel=root[10:13],
            dof_pos=env.simulator.dof_pos[ENV_ID],
            dof_vel=env.simulator.dof_vel[ENV_ID],
            gait_phase=gait.phase[ENV_ID],
            action=actor_state["actions"][ENV_ID],
        )

        self._step += 1
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        self._dump()

    def _dump(self) -> None:
        if self._dumped:
            return
        out_path = Path(self.training_loop.log_dir) / OUTPUT_NAME
        n = self._log.save(out_path)
        logger.info(f"[policy_eval] saved {n} steps to {out_path}")
        self._dumped = True

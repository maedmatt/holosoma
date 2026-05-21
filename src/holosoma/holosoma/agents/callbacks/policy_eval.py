"""Run a fixed schedule of velocity commands and save one NPZ per scenario.

Each scenario follows the same shape:

    [idle SETTLE_S] -> [active ACTIVE_S] -> [idle SETTLE_S] -> reset

The idle windows let the robot settle before and after the command. Rows
inside the active window are marked is_active=True; metrics should use
only those.

Output goes to <log_dir>/policy_eval/ as one NPZ per scenario plus a
manifest.json describing the schedule.

To start each scenario from the same pose, the callback sets
actor_state["eval_reset_request"] = True at the end of a scenario.
FastSACAgent.evaluate_policy calls env.reset() when it sees this flag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import PolicyEvalConfig
from holosoma.utils.timeseries_log import TimeseriesLogger


SETTLE_S = 1.0  # idle bracket before and after each scenario's active window
ACTIVE_S = 5.0  # duration of the velocity command in each scenario

ENV_ID = 0
OUTPUT_DIR = "policy_eval"
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class Scenario:
    """One velocity-command scenario in the eval schedule."""

    name: str
    vx: float
    vy: float
    vyaw: float


SCHEDULE: list[Scenario] = [
    Scenario(name="forward",      vx= 0.5, vy= 0.0, vyaw=0.0),
    Scenario(name="backward",     vx=-0.4, vy= 0.0, vyaw=0.0),
    Scenario(name="strafe_left",  vx= 0.0, vy= 0.3, vyaw=0.0),
    Scenario(name="strafe_right", vx= 0.0, vy=-0.3, vyaw=0.0),
]


class PolicyEvalCallback(RLEvalCallback):
    """Drive a fixed schedule of velocity commands, log one NPZ per scenario."""

    def __init__(self, config: PolicyEvalConfig, training_loop: Any = None) -> None:
        super().__init__(config, training_loop)
        self._scenario_idx = 0
        self._step_in_scenario = 0
        self._log = TimeseriesLogger()
        self._prev_foot_vel: torch.Tensor | None = None
        self._out_dir: Path | None = None
        # Derived from env.dt at the start of evaluation.
        self._dt = 0.0
        self._settle_steps = 0
        self._active_steps = 0
        self._scenario_steps = 0

    def _env(self):
        return self.training_loop._unwrap_env()

    def _scenario(self) -> Scenario:
        return SCHEDULE[self._scenario_idx]

    def _is_active_step(self) -> bool:
        """True while we are inside the active velocity-command window."""
        return self._settle_steps <= self._step_in_scenario < self._settle_steps + self._active_steps

    def on_pre_evaluate_policy(self) -> None:
        self._scenario_idx = 0
        self._step_in_scenario = 0
        self._log.clear()
        self._prev_foot_vel = None

        self._dt = self._env().dt
        self._settle_steps = int(round(SETTLE_S / self._dt))
        self._active_steps = int(round(ACTIVE_S / self._dt))
        self._scenario_steps = self._settle_steps + self._active_steps + self._settle_steps
        total_steps = self._scenario_steps * len(SCHEDULE)

        self._out_dir = Path(self.training_loop.log_dir) / OUTPUT_DIR
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

        logger.info(
            f"[policy_eval] {len(SCHEDULE)} scenarios "
            f"x {self._scenario_steps} steps = {total_steps} total steps "
            f"({total_steps * self._dt:.1f}s); "
            f"pass --training.max-eval-steps {total_steps} to cover it"
        )

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        scenario = self._scenario()
        if self._is_active_step():
            vx, vy, vyaw = scenario.vx, scenario.vy, scenario.vyaw
        else:
            vx, vy, vyaw = 0.0, 0.0, 0.0  # idle bracket

        cmd = self._env().command_manager.commands
        cmd[ENV_ID, 0] = vx
        cmd[ENV_ID, 1] = vy
        cmd[ENV_ID, 2] = vyaw
        return actor_state

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        # Eval loop may run longer than the schedule; stop logging when done.
        if self._scenario_idx >= len(SCHEDULE):
            return actor_state

        self._log_step(actor_state)
        self._step_in_scenario += 1

        if self._step_in_scenario >= self._scenario_steps:
            self._dump_scenario()
            self._scenario_idx += 1
            self._step_in_scenario = 0
            self._prev_foot_vel = None
            self._log.clear()
            # Ask the eval loop to reset so the next scenario starts from
            # the same pose as every other scenario.
            if self._scenario_idx < len(SCHEDULE):
                actor_state["eval_reset_request"] = True

        return actor_state

    def _log_step(self, actor_state: dict) -> None:
        env = self._env()

        foot_vel = env.simulator._rigid_body_vel[ENV_ID, env.feet_indices, :]
        foot_pos = env.simulator._rigid_body_pos[ENV_ID, env.feet_indices, :]
        foot_f = env.simulator.contact_forces[ENV_ID, env.feet_indices, :]

        if self._prev_foot_vel is None:
            foot_acc = torch.zeros_like(foot_vel)
        else:
            foot_acc = (foot_vel - self._prev_foot_vel) / self._dt
        self._prev_foot_vel = foot_vel.clone()

        root = env.simulator.robot_root_states[ENV_ID]
        gait = env.command_manager.get_state("locomotion_gait")

        self._log.append(
            t=self._step_in_scenario * self._dt,
            is_active=self._is_active_step(),
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

    def _dump_scenario(self) -> None:
        scenario = self._scenario()
        out_path = self._out_dir / f"scenario_{self._scenario_idx:02d}_{scenario.name}.npz"
        n = self._log.save(out_path)
        logger.info(
            f"[policy_eval] saved scenario {self._scenario_idx} ({scenario.name}): "
            f"{n} steps to {out_path}"
        )

    def _write_manifest(self) -> None:
        manifest = {
            "policy_eval_version": MANIFEST_VERSION,
            "simulator": type(self._env().simulator).__name__,
            "policy_hz": 1.0 / self._dt,
            "settle_s": SETTLE_S,
            "active_s": ACTIVE_S,
            "scenarios": [
                {
                    "index": i,
                    "name": s.name,
                    "vx": s.vx,
                    "vy": s.vy,
                    "vyaw": s.vyaw,
                    "n_steps": self._scenario_steps,
                    "settle_steps": self._settle_steps,
                    "active_steps": self._active_steps,
                    "tail_steps": self._settle_steps,
                }
                for i, s in enumerate(SCHEDULE)
            ],
        }
        out_path = self._out_dir / MANIFEST_NAME
        with out_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"[policy_eval] wrote manifest to {out_path}")

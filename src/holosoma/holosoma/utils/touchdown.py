"""Per-foot touchdown event detector, vectorized over [num_envs, num_feet].

Touchdown is a discrete event ("foot just hit the ground"), distinct from
contact state ("foot is on the ground right now"). Reward terms use a
contact gate (F_z > threshold); touchdown is what you want for impact
metrics and per-strike accounting.

Three algorithms. Default "hybrid" combines vz-based arming with F_z
rising-edge firing -- rejects intra-stance force chatter (vz arm) and
catches the actual contact instant (fz edge).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

Algo = Literal["vz_armed", "fz_edge", "hybrid"]


class TouchdownDetector:
    """Stateful touchdown detector. step() advances state; reset(env_ids) clears it."""

    def __init__(
        self,
        num_envs: int,
        num_feet: int = 2,
        algo: Algo = "hybrid",
        *,
        vz_arm_thr: float = 0.3,
        fz_fire_thr: float = 5.0,
        fz_release_thr: float = 1.0,
        refractory_steps: int = 3,
    ) -> None:
        """
        vz_arm_thr:       |vz| in m/s required to arm a descent (vz_armed, hybrid).
        fz_fire_thr:      Rising-edge F_z threshold in N (fz_edge, hybrid).
        fz_release_thr:   F_z must drop below this to re-arm fz_edge.
        refractory_steps: Min steps between fires per foot (fz_edge, hybrid).
        """
        self._shape = (num_envs, num_feet)
        self._vz_arm_thr = vz_arm_thr
        self._fz_fire_thr = fz_fire_thr
        self._fz_release_thr = fz_release_thr
        self._refractory_steps = refractory_steps
        self._step_fn = {
            "vz_armed": self._step_vz_armed,
            "fz_edge": self._step_fz_edge,
            "hybrid": self._step_hybrid,
        }[algo]
        self._prev_vz: Tensor | None = None
        self._prev_fz: Tensor | None = None
        self._armed: Tensor | None = None
        self._cooldown: Tensor | None = None

    def step(self, foot_vz: Tensor, foot_fz: Tensor) -> Tensor:
        """Advance one step. Returns bool mask [E, F] of touchdowns fired this step."""
        if self._prev_vz is None:
            device = foot_vz.device
            dtype = foot_vz.dtype
            self._prev_vz = torch.zeros(self._shape, device=device, dtype=dtype)
            self._prev_fz = torch.zeros(self._shape, device=device, dtype=dtype)
            self._armed = torch.zeros(self._shape, device=device, dtype=torch.bool)
            self._cooldown = torch.zeros(self._shape, device=device, dtype=torch.int32)
        return self._step_fn(foot_vz, foot_fz)

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Clear per-foot state. None = all envs; tensor = those envs only."""
        if self._prev_vz is None:
            return
        if env_ids is None:
            self._prev_vz.zero_()
            self._prev_fz.zero_()
            self._armed.zero_()
            self._cooldown.zero_()
        else:
            self._prev_vz[env_ids] = 0
            self._prev_fz[env_ids] = 0
            self._armed[env_ids] = False
            self._cooldown[env_ids] = 0

    def _step_vz_armed(self, vz: Tensor, fz: Tensor) -> Tensor:
        """Arm on descent (vz < -thr), fire on vz rising through zero."""
        self._armed |= vz < -self._vz_arm_thr
        fired = self._armed & (self._prev_vz < 0) & (vz >= 0)
        self._armed &= ~fired
        self._prev_vz.copy_(vz)
        return fired

    def _step_fz_edge(self, vz: Tensor, fz: Tensor) -> Tensor:
        """Arm when F_z drops below release_thr, fire on F_z rising through fire_thr."""
        self._armed |= self._prev_fz < self._fz_release_thr
        rising = (self._prev_fz < self._fz_fire_thr) & (fz >= self._fz_fire_thr)
        fired = self._armed & rising & (self._cooldown == 0)
        self._armed &= ~fired
        self._cooldown = torch.where(
            fired,
            torch.full_like(self._cooldown, self._refractory_steps),
            torch.clamp(self._cooldown - 1, min=0),
        )
        self._prev_fz.copy_(fz)
        return fired

    def _step_hybrid(self, vz: Tensor, fz: Tensor) -> Tensor:
        """vz_armed's arming AND fz_edge's firing. Fires at first-contact, no chatter."""
        self._armed |= vz < -self._vz_arm_thr
        rising = (self._prev_fz < self._fz_fire_thr) & (fz >= self._fz_fire_thr)
        fired = self._armed & rising & (self._cooldown == 0)
        self._armed &= ~fired
        self._cooldown = torch.where(
            fired,
            torch.full_like(self._cooldown, self._refractory_steps),
            torch.clamp(self._cooldown - 1, min=0),
        )
        self._prev_vz.copy_(vz)
        self._prev_fz.copy_(fz)
        return fired

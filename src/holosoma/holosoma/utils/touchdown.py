"""Per-foot touchdown event detector, vectorized over [num_envs, num_feet].

Touchdown is a discrete event ("foot just hit the ground"), distinct from
contact state ("foot is on the ground right now"). Reward terms use a
contact gate (F_z > threshold); touchdown is what you want for impact
metrics and per-strike accounting.

Algorithm: arm on descent (vz < -arm_thr), fire on F_z rising edge through
fire_thr. The vz arm rejects intra-stance force chatter; the F_z rising
edge catches the actual contact instant.
"""

from __future__ import annotations

import torch
from torch import Tensor


class TouchdownDetector:
    """Stateful touchdown detector. step() advances state; reset(env_ids) clears it."""

    def __init__(
        self,
        num_envs: int,
        num_feet: int = 2,
        *,
        vz_arm_thr: float = 0.3,
        fz_fire_thr: float = 1.0,
    ) -> None:
        """
        vz_arm_thr:  |vz| in m/s required to arm a descent.
        fz_fire_thr: rising-edge F_z threshold in N.
        """
        self._shape = (num_envs, num_feet)
        self._vz_arm_thr = vz_arm_thr
        self._fz_fire_thr = fz_fire_thr
        self._prev_fz: Tensor | None = None
        self._armed: Tensor | None = None

    def step(self, foot_vz: Tensor, foot_fz: Tensor) -> Tensor:
        """Advance one step. Returns bool mask [E, F] of touchdowns fired this step."""
        if self._prev_fz is None:
            device = foot_vz.device
            dtype = foot_vz.dtype
            self._prev_fz = torch.zeros(self._shape, device=device, dtype=dtype)
            self._armed = torch.zeros(self._shape, device=device, dtype=torch.bool)
        self._armed |= foot_vz < -self._vz_arm_thr
        rising = (self._prev_fz < self._fz_fire_thr) & (foot_fz >= self._fz_fire_thr)
        fired = self._armed & rising
        self._armed &= ~fired
        self._prev_fz.copy_(foot_fz)
        return fired

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Clear per-foot state. None = all envs; tensor = those envs only."""
        if self._prev_fz is None:
            return
        if env_ids is None:
            self._prev_fz.zero_()
            self._armed.zero_()
        else:
            self._prev_fz[env_ids] = 0
            self._armed[env_ids] = False

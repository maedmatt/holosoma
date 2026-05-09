"""Inline locosonic NoisePredictor inference, sized for v9_11-style checkpoints.

Loads a `best.pt` produced by `locosonic train` and runs it against simulator
state assembled from leg q/dq and the locomotion velocity command.

Only past temporal context is supported (`@t-k` / `@t`). Future context
(`@t+k`) would require knowing the next state and is rejected.

The buffer is a deque of length pre+1; appending implicitly drops the oldest
sample. On any tick, calling predict() runs the model on the current window
or returns None while the buffer is still warming up.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor, nn

# Leg joint name -> 23-DOF index. Upper-body joints are not used by v9_xx.
_JOINT_INDEX: dict[str, int] = {
    "left_hip_pitch": 0, "left_hip_roll": 1, "left_hip_yaw": 2,
    "left_knee": 3, "left_ankle_pitch": 4, "left_ankle_roll": 5,
    "right_hip_pitch": 6, "right_hip_roll": 7, "right_hip_yaw": 8,
    "right_knee": 9, "right_ankle_pitch": 10, "right_ankle_roll": 11,
}
_CMD_INDEX: dict[str, int] = {"vx": 0, "vy": 1, "vyaw": 2}
_SIGNAL_SOURCE: dict[str, str] = {"q": "dof_pos", "dq": "dof_vel"}


class _Model(nn.Module):
    """MLP with internal z-score normalization. Mirrors locosonic.NoisePredictor."""

    def __init__(self, n_features: int, n_outputs: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_outputs)
        self.register_buffer("input_mean", torch.zeros(n_features))
        self.register_buffer("input_std", torch.ones(n_features))

    def forward(self, x: Tensor) -> Tensor:
        x = ((x - self.input_mean) / self.input_std).clamp(-5.0, 5.0)
        return self.head(self.hidden(x))


def _parse_name(name: str) -> tuple[str, int, int]:
    """Parse 'joints.left_knee.q@t-2' -> ('dof_vel', 3, -2)."""
    base, _, suffix = name.partition("@")
    if not suffix or suffix == "t":
        offset = 0
    elif suffix.startswith("t-"):
        offset = -int(suffix[2:])
    else:
        raise ValueError(f"Unsupported time suffix in {name!r}: only @t and @t-k are supported")

    if base.startswith("joints."):
        _, joint, signal = base.split(".")
        if joint not in _JOINT_INDEX or signal not in _SIGNAL_SOURCE:
            raise ValueError(f"Unsupported joint feature {name!r} (only legs q/dq)")
        return _SIGNAL_SOURCE[signal], _JOINT_INDEX[joint], offset

    if base.startswith("joystick."):
        cmd = base.split(".", 1)[1]
        if cmd not in _CMD_INDEX:
            raise ValueError(f"Unknown joystick axis in {name!r}")
        return "cmd", _CMD_INDEX[cmd], offset

    raise ValueError(f"Cannot parse feature name {name!r}")


class NoisePredictor:
    """Past-context predictor with a sliding-window buffer of base features."""

    def __init__(self, checkpoint_path: str | Path, device: torch.device) -> None:
        self.device = device
        ckpt = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        names: list[str] = ckpt["input_feature_names"]
        parsed = [_parse_name(n) for n in names]

        # Locosonic packs all `n_steps` consecutive entries for one base feature
        # together: [feat0@t-k, ..., feat0@t, feat1@t-k, ...]. n_steps is the
        # length of the leading run with the same (source, src_idx).
        first_key = parsed[0][:2]
        n_steps = 1
        while n_steps < len(parsed) and parsed[n_steps][:2] == first_key:
            n_steps += 1
        n_base = len(parsed) // n_steps

        # Per-source gather: which destination slots in the base buffer come
        # from each sim tensor, and at which indices.
        sources = [parsed[b * n_steps][0] for b in range(n_base)]
        src_idx_all = [parsed[b * n_steps][1] for b in range(n_base)]
        self._gather: dict[str, tuple[Tensor, Tensor]] = {}
        for src in ("dof_pos", "dof_vel", "cmd"):
            slots = [b for b, s in enumerate(sources) if s == src]
            if slots:
                self._gather[src] = (
                    torch.tensor(slots, dtype=torch.long, device=device),
                    torch.tensor([src_idx_all[b] for b in slots], dtype=torch.long, device=device),
                )

        model_cfg = ckpt["experiment"]["model"]
        self.model = _Model(
            n_features=len(names),
            n_outputs=ckpt["model_state_dict"]["head.bias"].shape[0],
            hidden_dims=tuple(model_cfg["hidden_dims"]),
            dropout=float(model_cfg["dropout"]),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device).eval()

        self._n_base = n_base
        self._n_steps = n_steps
        self._buffer: deque[Tensor] = deque(maxlen=n_steps)

        logger.info(f"Noise predictor: {n_base} base features × {n_steps} steps = {len(names)} inputs")

    def predict(self, env: Any, env_id: int) -> Tensor | None:
        """Append the current base feature row, then forward if the buffer is full."""
        sim = env.simulator
        # Mirror the locosonic recorder's 0.1 stick deadzone so cmd stays in
        # the same distribution the predictor was trained on.
        cmd = env.command_manager.commands[env_id]
        sources = {
            "dof_pos": sim.dof_pos[env_id],
            "dof_vel": sim.dof_vel[env_id],
            "cmd": cmd * (cmd.abs() > 0.1),
        }
        row = torch.empty(self._n_base, device=self.device)
        for src, (dst, src_idx) in self._gather.items():
            row[dst] = sources[src][src_idx]
        self._buffer.append(row)

        if len(self._buffer) < self._n_steps:
            return None

        # Stack [n_steps, n_base] -> transpose -> flatten matches locosonic's
        # [feat0@t-k, ..., feat0@t, feat1@t-k, ...] packing.
        x = torch.stack(list(self._buffer)).T.reshape(-1).unsqueeze(0)
        with torch.no_grad():
            return self.model(x)[0]

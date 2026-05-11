"""Locosonic noise-predictor inference for sim consumers.

NoisePredictor: single-env, deque buffer (eval callback).
BatchedNoisePredictor: many-env, tensor buffer (training reward).

Only past temporal context (`@t-k`, `@t`) is supported.
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


def _load_predictor(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[_Model, int, int, dict[str, tuple[Tensor, Tensor]]]:
    """Load a locosonic predictor checkpoint and return (model, n_steps, n_base, gather).

    `gather` maps a source name ("dof_pos", "dof_vel", "cmd") to a (dst_slots, src_idx)
    pair: which base-feature slots come from that source, and at which source indices.
    """
    ckpt = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    names: list[str] = ckpt["input_feature_names"]
    parsed = [_parse_name(n) for n in names]

    # Locosonic packs all `n_steps` consecutive entries for one base feature
    # together: [feat0@t-k, ..., feat0@t, feat1@t-k, ...]. n_steps is the length
    # of the leading run with the same (source, src_idx).
    first_key = parsed[0][:2]
    n_steps = 1
    while n_steps < len(parsed) and parsed[n_steps][:2] == first_key:
        n_steps += 1
    n_base = len(parsed) // n_steps

    sources = [parsed[b * n_steps][0] for b in range(n_base)]
    src_idx_all = [parsed[b * n_steps][1] for b in range(n_base)]
    gather: dict[str, tuple[Tensor, Tensor]] = {}
    for src in ("dof_pos", "dof_vel", "cmd"):
        slots = [b for b, s in enumerate(sources) if s == src]
        if slots:
            gather[src] = (
                torch.tensor(slots, dtype=torch.long, device=device),
                torch.tensor([src_idx_all[b] for b in slots], dtype=torch.long, device=device),
            )

    model_cfg = ckpt["experiment"]["model"]
    model = _Model(
        n_features=len(names),
        n_outputs=ckpt["model_state_dict"]["head.bias"].shape[0],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, n_steps, n_base, gather


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
        self.model, self._n_steps, self._n_base, self._gather = _load_predictor(checkpoint_path, device)
        self._buffer: deque[Tensor] = deque(maxlen=self._n_steps)
        logger.info(
            f"Noise predictor: {self._n_base} base features × {self._n_steps} steps "
            f"= {self._n_base * self._n_steps} inputs"
        )

    def update_buffer(self, env: Any, env_id: int) -> bool:
        """Append current base features to the rolling buffer. Returns True once full."""
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
        return len(self._buffer) >= self._n_steps

    def forward(self) -> Tensor:
        """Run the model on the current buffer. Caller must ensure buffer is full."""
        # Stack [n_steps, n_base] -> transpose -> flatten matches locosonic's
        # [feat0@t-k, ..., feat0@t, feat1@t-k, ...] packing.
        x = torch.stack(list(self._buffer)).T.reshape(-1).unsqueeze(0)
        with torch.no_grad():
            return self.model(x)[0]


class BatchedNoisePredictor:
    """Vectorized predictor over `num_envs` parallel envs.

    Buffer is a tensor `[num_envs, n_steps, n_base]`. Each `update_buffer(env)` call
    rolls the time axis and writes the current row for every env in one shot. A
    per-env counter tracks warm-up so callers can mask outputs for envs whose
    buffer is still filling (e.g. right after a reset).
    """

    def __init__(self, checkpoint_path: str | Path, num_envs: int, device: torch.device) -> None:
        self.device = device
        self.num_envs = num_envs
        self.model, self._n_steps, self._n_base, self._gather = _load_predictor(checkpoint_path, device)
        self.n_outputs = int(self.model.head.bias.shape[0])
        self._buffer = torch.zeros((num_envs, self._n_steps, self._n_base), device=device)
        self._n_filled = torch.zeros(num_envs, dtype=torch.long, device=device)
        logger.info(
            f"Batched noise predictor: {num_envs} envs × {self._n_base} base × {self._n_steps} steps "
            f"= {self._n_base * self._n_steps} inputs, {self.n_outputs} outputs"
        )

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Clear buffer rows so warm-up restarts. None = all envs."""
        if env_ids is None:
            self._buffer.zero_()
            self._n_filled.zero_()
        else:
            self._buffer[env_ids] = 0
            self._n_filled[env_ids] = 0

    def update_buffer(self, env: Any) -> Tensor:
        """Roll the time axis and write the current row for every env. Returns [E] ready mask."""
        sim = env.simulator
        cmd = env.command_manager.commands
        sources = {
            "dof_pos": sim.dof_pos,
            "dof_vel": sim.dof_vel,
            "cmd": cmd * (cmd.abs() > 0.1),
        }
        row = torch.empty((self.num_envs, self._n_base), device=self.device)
        for src, (dst, src_idx) in self._gather.items():
            row[:, dst] = sources[src][:, src_idx]
        self._buffer = torch.roll(self._buffer, shifts=-1, dims=1)
        self._buffer[:, -1, :] = row
        self._n_filled = torch.clamp(self._n_filled + 1, max=self._n_steps)
        return self._n_filled >= self._n_steps

    def forward(self) -> Tensor:
        """Run the model on every env's buffer. Returns [E, n_outputs]."""
        # [E, n_steps, n_base] -> [E, n_base, n_steps] -> [E, n_base*n_steps] matches
        # locosonic's [feat0@t-k, ..., feat0@t, feat1@t-k, ...] packing.
        x = self._buffer.transpose(1, 2).reshape(self.num_envs, -1)
        with torch.no_grad():
            return self.model(x)

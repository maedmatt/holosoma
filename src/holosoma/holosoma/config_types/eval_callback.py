"""Config types for eval callbacks."""

from __future__ import annotations

import dataclasses

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RecordingConfig:
    """Settings for trajectory recording during evaluation."""

    enabled: bool = False
    """Whether to enable trajectory recording."""

    output_path: str = "eval_recording.npz"
    """Path to save NPZ recording."""

    env_id: int = 0
    """Environment ID to record."""


@dataclass(frozen=True)
class NoisePredictorConfig:
    """Settings for noise predictor during evaluation."""

    enabled: bool = False
    """Whether to enable noise prediction."""

    checkpoint_path: str = ""
    """Path to predictor checkpoint (.pt)."""

    env_id: int = 0
    """Environment ID to predict for."""


@dataclass(frozen=True)
class RecordingCallbackConfig:
    """Instantiation config for EvalRecordingCallback."""

    _target_: str = "holosoma.agents.callbacks.recording.EvalRecordingCallback"
    """Class to instantiate."""

    config: RecordingConfig = RecordingConfig()
    """Recording settings."""


@dataclass(frozen=True)
class NoisePredictorCallbackConfig:
    """Instantiation config for NoisePredictorCallback."""

    _target_: str = "holosoma.agents.callbacks.noise_predictor.NoisePredictorCallback"
    """Class to instantiate."""

    config: NoisePredictorConfig = NoisePredictorConfig()
    """Noise predictor settings."""


@dataclass(frozen=True)
class PolicyEvalConfig:
    """Settings for scripted policy evaluation."""

    enabled: bool = False
    """Whether to enable scripted policy evaluation (drives commands from SCHEDULE)."""


@dataclass(frozen=True)
class PolicyEvalCallbackConfig:
    """Instantiation config for PolicyEvalCallback."""

    _target_: str = "holosoma.agents.callbacks.policy_eval.PolicyEvalCallback"
    """Class to instantiate."""

    config: PolicyEvalConfig = PolicyEvalConfig()
    """Scripted policy evaluation settings."""


@dataclass(frozen=True)
class EvalCallbacksConfig:
    """Container for all eval callback configs.

    To add a new callback, add a field here with its config type.
    Each field's value is passed to instantiate() if it has a _target_.
    """

    recording: RecordingCallbackConfig = RecordingCallbackConfig()
    """Trajectory recording callback."""

    noise_predictor: NoisePredictorCallbackConfig = NoisePredictorCallbackConfig()
    """Noise predictor callback."""

    policy_eval: PolicyEvalCallbackConfig = PolicyEvalCallbackConfig()
    """Scripted policy evaluation callback."""

    def collect_active_callbacks(self) -> dict:
        """Collect callback configs where config.enabled is True."""
        cb_configs = {}
        for f in dataclasses.fields(self):
            cfg = getattr(self, f.name)
            if not hasattr(cfg, "_target_"):
                raise ValueError(f"Callback config '{f.name}' missing _target_ field")
            if not hasattr(cfg.config, "enabled"):
                raise ValueError(f"Callback config '{f.name}' missing config.enabled field")
            if cfg.config.enabled:
                cb_configs[f.name] = cfg
        return cb_configs

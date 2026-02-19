"""DDS message types for policy telemetry, recorded by NoisePrint."""

from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence


@dataclass
class PolicyTelemetry(IdlStruct):
    """Flat float array published over DDS for NoisePrint recording."""

    timestamp: float = 0.0
    data: sequence[float] = ()  # type: ignore[assignment]

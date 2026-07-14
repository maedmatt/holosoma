"""Replay a fixed schedule of velocity commands instead of the joystick.

Hold the deadman combo (X+down) on the wireless controller and the robot
walks through SCHEDULE below; release and you are back on normal stick control
with the schedule frozen where it paused. This mirrors the sim benchmark
(holosoma/agents/callbacks/policy_eval.py) so the same path can be replayed
across policies and compared.

Tweak SCHEDULE / SETTLE_S to change the path. Each scenario runs for its own
duration; a short idle bracket is inserted before each one so the robot settles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from termcolor import colored


# X+down wireless-controller bitmask (base_interface keymap: X=1024, down=16384).
DEADMAN_KEYS = 17408

SETTLE_S = 1.0  # idle pause before each scenario; set to 0 to chain directly


@dataclass(frozen=True)
class Scenario:
    """One velocity-command segment. Edit these to change the path."""

    name: str
    vx: float
    vy: float
    vyaw: float
    duration_s: float


SCHEDULE: list[Scenario] = [
    Scenario(name="forward",  vx= 0.3, vy=0.0, vyaw=0.0, duration_s=5.0),
    Scenario(name="backward", vx=-0.3, vy=0.0, vyaw=0.0, duration_s=5.0),
]


class ScriptRunner:
    """Tick-driven player over SCHEDULE; advances only when step() is called."""

    def __init__(self, schedule: list[Scenario], dt: float, logger=None) -> None:
        self.logger = logger
        settle = max(0, round(SETTLE_S / dt))
        # Flatten to (vx, vy, vyaw, n_steps, label) phases: idle bracket then command.
        self._phases: list[tuple[float, float, float, int, str]] = []
        for s in schedule:
            if settle:
                self._phases.append((0.0, 0.0, 0.0, settle, f"settle->{s.name}"))
            self._phases.append((s.vx, s.vy, s.vyaw, max(1, round(s.duration_s / dt)), s.name))
        self._i = 0  # phase index
        self._k = 0  # step within phase
        self._t0: float | None = None
        self._done_logged = False

    def reset(self) -> None:
        """Rewind to the start of the schedule (used when relaunching from the keyboard)."""
        self._i = 0
        self._k = 0
        self._t0 = None
        self._done_logged = False

    def step(self) -> tuple[float, float, float] | None:
        """Advance one control tick. Returns (vx, vy, vyaw), or None when finished."""
        if self._i >= len(self._phases):
            if not self._done_logged:
                self._log("[eval_script] schedule complete")
                self._done_logged = True
            return None

        vx, vy, vyaw, n, label = self._phases[self._i]
        if self._k == 0:
            if self._t0 is None:
                self._t0 = time.perf_counter()
            t = time.perf_counter() - self._t0
            self._log(f"[eval_script] t={t:5.1f}s  {label}: vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}")

        self._k += 1
        if self._k >= n:
            self._i += 1
            self._k = 0
        return vx, vy, vyaw

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(colored(msg, "cyan"))
        else:
            print(msg)

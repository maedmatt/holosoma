from __future__ import annotations

import math
import time

import numpy as np
from termcolor import colored

from .base import BasePolicy

# (duration_s, lin_vel_x, ang_vel_z)
_SCRIPTED_SEGMENTS: list[tuple[float, float, float]] = [
    (2.0, 0.0, 0.0),  # wait
    (5.0, 0.5, 0.0),  # walk forward
    (5.0, -0.5, 0.0),  # walk backward
]


class LocomotionPolicy(BasePolicy):
    def __init__(self, config):
        super().__init__(config)
        self.is_standing = False

        # Scripted path state (only used when use_scripted_path is set)
        self._scripted_enabled = config.task.use_scripted_path
        self._scripted_start: float | None = None
        self._scripted_done = True
        self._scripted_segment_index = -1

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_buffer_dict["actions"] = self.last_policy_action
        current_obs_buffer_dict["command_lin_vel"] = self.lin_vel_command
        current_obs_buffer_dict["command_ang_vel"] = self.ang_vel_command
        current_obs_buffer_dict["command_stand"] = self.stand_command

        # Add phase observations only if they are configured
        if "sin_phase" in self.obs_dict.get("actor_obs", []):
            current_obs_buffer_dict["sin_phase"] = self._get_obs_sin_phase()
        if "cos_phase" in self.obs_dict.get("actor_obs", []):
            current_obs_buffer_dict["cos_phase"] = self._get_obs_cos_phase()

        return current_obs_buffer_dict

    def _get_obs_sin_phase(self):
        """Calculate sin phase for gait."""
        return np.array([np.sin(self.phase[0, :])])

    def _get_obs_cos_phase(self):
        """Calculate cos phase for gait."""
        return np.array([np.cos(self.phase[0, :])])

    def update_phase_time(self):
        """Update phase time."""
        phase_tp1 = self.phase + self.phase_dt
        self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        if np.linalg.norm(self.lin_vel_command[0]) < 0.01 and np.linalg.norm(self.ang_vel_command[0]) < 0.01:
            # Robot should stand still - set both feet to same phase
            self.phase[0, :] = np.pi * np.ones(2)
            self.is_standing = True
        elif self.is_standing:
            # When the robot starts to move, reset the phase to initial state
            self.phase = np.array([[0.0, np.pi]])
            self.is_standing = False

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses for locomotion."""
        # Call parent handler for common commands
        super().handle_keyboard_button(keycode)

        # Locomotion-specific commands
        if keycode in ["w", "s", "a", "d"]:
            self._handle_velocity_control(keycode)
        elif keycode in ["q", "e"]:
            self._handle_angular_velocity_control(keycode)
        elif keycode == "=":
            self._handle_stand_command()
        elif keycode == "z":
            self._handle_zero_velocity()
        elif keycode == "p" and self._scripted_enabled:
            self._handle_start_scripted()

        self._print_control_status()

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses for locomotion."""
        # Call parent handler for common commands
        super().handle_joystick_button(cur_key)

        # Locomotion-specific commands
        if cur_key == "start":
            self._handle_stand_command()
        elif cur_key == "L2":
            self._handle_zero_velocity()

    def _handle_velocity_control(self, keycode):
        """Handle linear velocity control."""
        if not self.stand_command[0, 0]:
            return

        if keycode == "w":
            self.lin_vel_command[0, 0] += 0.1
        elif keycode == "s":
            self.lin_vel_command[0, 0] -= 0.1
        elif keycode == "a":
            self.lin_vel_command[0, 1] += 0.1
        elif keycode == "d":
            self.lin_vel_command[0, 1] -= 0.1

    def _handle_angular_velocity_control(self, keycode):
        """Handle angular velocity control."""
        if keycode == "q":
            self.ang_vel_command[0, 0] -= 0.1
        elif keycode == "e":
            self.ang_vel_command[0, 0] += 0.1

    def _handle_stand_command(self):
        """Handle stand command toggle."""
        self.stand_command[0, 0] = 1 - self.stand_command[0, 0]
        if self.stand_command[0, 0] == 0:
            self.ang_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 1] = 0.0
            self.logger.info(colored("Stance command", "blue"))
        else:
            self.base_height_command[0, 0] = self.desired_base_height
            self.logger.info(colored("Walk command", "blue"))

    def _handle_stop_policy(self):
        """Handle stop policy action — also aborts any scripted path."""
        self._scripted_done = True
        super()._handle_stop_policy()

    def _handle_zero_velocity(self):
        """Handle zero velocity command — also aborts any scripted path."""
        self._scripted_done = True
        self.ang_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 1] = 0.0
        self.logger.info(colored("Velocities set to zero", "blue"))

    def _print_control_status(self):
        """Print current control status."""
        super()._print_control_status()

        # Extract values for better formatting
        lin_vel_x = self.lin_vel_command[0, 0]
        lin_vel_y = self.lin_vel_command[0, 1]
        ang_vel_z = self.ang_vel_command[0, 0]
        is_walking = self.stand_command[0, 0] == 1

        # Print with clear labels and units
        mode = "Walking" if is_walking else "Standing"
        status = "✓ applied" if is_walking else "✗ not applied"
        print(f"Linear velocity: x={lin_vel_x:+.2f} m/s, y={lin_vel_y:+.2f} m/s")
        print(f"Angular velocity: {ang_vel_z:+.2f} rad/s")
        print(f"Mode: {mode} ({status})")
        print("💡 Terminal keys: W/A/S/D (lin) | Q/E (ang) | = (toggle mode)")
        print("🎬 MuJoCo keys (in simulator only): 7/8 (band) | 9 (toggle) | BACKSPACE (reset)")

    # ============================================================================
    # Scripted Path
    # ============================================================================

    def _handle_start_scripted(self):
        """Activate scripted velocity sequence. Policy must already be running in walk mode."""
        if not self.use_policy_action:
            self.logger.warning("[scripted] policy not active — press ] first")
            return
        if not self.stand_command[0, 0]:
            self.logger.warning("[scripted] not in walk mode — press = first")
            return
        self._scripted_start = time.monotonic()
        self._scripted_done = False
        self._scripted_segment_index = -1
        self.logger.info(colored("[scripted] sequence started — press o or z to abort", "cyan"))

    def _update_scripted_commands(self) -> None:
        """Overwrite velocity commands based on elapsed time in the scripted sequence."""
        if self._scripted_done or self._scripted_start is None:
            return

        elapsed = time.monotonic() - self._scripted_start

        cumulative = 0.0
        for i, (duration, lin_x, ang_z) in enumerate(_SCRIPTED_SEGMENTS):
            if elapsed < cumulative + duration:
                if i != self._scripted_segment_index:
                    self._scripted_segment_index = i
                    self.logger.info(
                        colored(
                            f"[scripted] segment {i}: lin_vel={lin_x:.2f} ang_vel={ang_z:.2f} "
                            f"t={elapsed:.1f}s",
                            "cyan",
                        )
                    )
                self.lin_vel_command[0, 0] = lin_x
                self.lin_vel_command[0, 1] = 0.0
                self.ang_vel_command[0, 0] = ang_z
                return
            cumulative += duration

        # All segments finished — zero velocities, stay in walk mode
        self.lin_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 1] = 0.0
        self.ang_vel_command[0, 0] = 0.0
        self._scripted_done = True
        total = sum(d for d, _, _ in _SCRIPTED_SEGMENTS)
        self.logger.info(colored(f"[scripted] sequence complete ({total:.1f}s)", "green"))

    def policy_action(self):
        """Inject scripted commands before running the normal policy action."""
        self._update_scripted_commands()
        super().policy_action()

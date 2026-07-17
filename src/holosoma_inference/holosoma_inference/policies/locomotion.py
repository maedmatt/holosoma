import numpy as np
from termcolor import colored

from .base import BasePolicy
from .eval_script import DEADMAN_KEYS, SCHEDULE, ScriptRunner


class LocomotionPolicy(BasePolicy):
    def __init__(self, config):
        super().__init__(config)
        self.is_standing = False
        # Scripted command replay: joystick holds X+down (deadman); keyboard latches with 'p'.
        self._script = ScriptRunner(SCHEDULE, 1.0 / self.rl_rate, logger=self.logger)
        self._script_active = False  # keyboard 'p' toggle; joystick uses the deadman instead

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
        elif keycode == "p":
            self._toggle_script()

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

    def _drive_script(self):
        """Set the velocity command from the next schedule tick (stand still when finished)."""
        cmd = self._script.step()
        if cmd is None:  # schedule finished: stand still
            self.lin_vel_command[:] = 0.0
            self.ang_vel_command[:] = 0.0
        else:
            self.lin_vel_command[0, 0], self.lin_vel_command[0, 1] = cmd[0], cmd[1]
            self.ang_vel_command[0, 0] = cmd[2]
        self.stand_command[0, 0] = 1

    def _toggle_script(self):
        """Keyboard 'p': start the scripted path from the top, or stop it and stand still."""
        self._script_active = not self._script_active
        if self._script_active:
            self._script.reset()
            self.logger.info(colored("Scripted path: START (press p again to stop)", "cyan"))
        else:
            self._handle_zero_velocity()
            self.logger.info(colored("Scripted path: STOP", "cyan"))

    def process_joystick_input(self):
        """Deadman replay: hold X+down to drive commands from SCHEDULE, else manual."""
        msg = self.interface.get_joystick_msg()
        keys = getattr(msg, "keys", 0)
        if keys == DEADMAN_KEYS:
            self._drive_script()
            return
        if keys & DEADMAN_KEYS:
            # Only one of the two deadman buttons is held (mid press/release):
            # ignore this frame so X/down don't trigger their normal actions
            # (down would lower kp_level).
            return
        super().process_joystick_input()

    def policy_action(self):
        """Keyboard mode: drive the scripted path each tick while 'p' is latched on."""
        if self._script_active:
            self._drive_script()
        super().policy_action()

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

    def _handle_zero_velocity(self):
        """Handle zero velocity command."""
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

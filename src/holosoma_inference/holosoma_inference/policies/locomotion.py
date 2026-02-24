from __future__ import annotations

from pathlib import Path
from typing_extensions import override

import numpy as np
from termcolor import colored

from .base import BasePolicy


class LocomotionPolicy(BasePolicy):
    def __init__(self, config):
        super().__init__(config)
        self.is_standing = False

        # Velocity command replay
        self._replay_commands: np.ndarray | None = None
        self._replay_step: int = 0
        self._replay_active: bool = False

        if config.task.replay_commands_path is not None:
            self._load_replay_commands(config.task.replay_commands_path)

    def _load_replay_commands(self, path: str) -> None:
        """Load velocity commands from parquet. Expects a 'joystick' column (FixedSizeList[3]: vx, vy, vyaw)."""
        import pyarrow.parquet as pq

        table = pq.read_table(Path(path), columns=["joystick"])
        # FixedSizeList column: flatten to (N*3,) then reshape to (N, 3)
        flat = table.column("joystick").combine_chunks().values.to_numpy(zero_copy_only=False)
        self._replay_commands = flat.reshape(-1, 3).astype(np.float32)

        n = len(self._replay_commands)
        duration = n / self.rl_rate
        self.logger.info(f"Loaded replay commands: {n} steps ({duration:.1f}s at {self.rl_rate} Hz) from {path}")

    def _start_replay(self) -> None:
        self._replay_active = True
        self._replay_step = 0
        self.logger.info(colored("Velocity replay started", "green"))

    def _stop_replay(self, reason: str) -> None:
        if not self._replay_active:
            return
        self._replay_active = False
        self.lin_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 1] = 0.0
        self.ang_vel_command[0, 0] = 0.0
        self.logger.info(colored(f"Velocity replay stopped ({reason})", "yellow"))

    @override
    def policy_action(self):
        if self._replay_active and self._replay_commands is not None:
            step = self._replay_step
            total = len(self._replay_commands)

            if step >= total:
                self._stop_replay("completed")
            else:
                vx, vy, vyaw = self._replay_commands[step]
                self.lin_vel_command[0, 0] = vx
                self.lin_vel_command[0, 1] = vy
                self.ang_vel_command[0, 0] = vyaw
                self._replay_step += 1

                if step % 50 == 0:
                    self.logger.info(f"Replay step {step}/{total} (vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f})")

        super().policy_action()

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
            self._stop_replay("aborted")
        elif keycode == "p":
            self._handle_replay_key()

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

    def _handle_zero_velocity(self):
        """Handle zero velocity command."""
        self.ang_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 1] = 0.0
        self.logger.info(colored("Velocities set to zero", "blue"))

    def _handle_replay_key(self) -> None:
        """Start velocity replay if configured and preconditions are met."""
        if self._replay_commands is None:
            return
        if not self.use_policy_action:
            self.logger.warning("Start policy first (]) before replay")
            return
        if not self.stand_command[0, 0]:
            self.logger.warning("Enable walk mode (=) before replay")
            return
        self._start_replay()

    @override
    def _handle_stop_policy(self):
        self._stop_replay("aborted")
        super()._handle_stop_policy()

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

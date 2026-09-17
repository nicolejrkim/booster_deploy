from typing import Optional
import evdev
import threading
from dataclasses import dataclass
import time
import termios
import tty
import select
import atexit
import sys

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from booster_interface.msg import RemoteControllerState


@dataclass
class JoystickConfig:
    max_vx: float = 1.
    max_vy: float = 1.
    max_vyaw: float = 1.
    control_threshold: float = 0.1
    # logitech
    custom_mode_button: evdev.ecodes = evdev.ecodes.BTN_X
    rl_gait_button: evdev.ecodes = evdev.ecodes.BTN_A
    back_button: evdev.ecodes = evdev.ecodes.BTN_Y
    estop_button: evdev.ecodes = evdev.ecodes.BTN_B
    x_axis: evdev.ecodes = evdev.ecodes.ABS_Y
    y_axis: evdev.ecodes = evdev.ecodes.ABS_X
    yaw_axis: evdev.ecodes = evdev.ecodes.ABS_Z

    # xiaoji (use the same X=Custom, A=RL mode mapping)
    # x_axis: evdev.ecodes = evdev.ecodes.ABS_Y
    # y_axis: evdev.ecodes = evdev.ecodes.ABS_X
    # yaw_axis: evdev.ecodes = evdev.ecodes.ABS_RX


class RemoteControlService:
    # Logical buttons: X (custom / STAND), A (RL / forward), Y (back), B (ESTOP)
    BUTTONS = ("x", "a", "y", "b")
    # Keyboard keys mapped to the logical buttons.
    KEYBOARD_BUTTONS = {"x": "x", "r": "a", "n": "y", "b": "b"}
    """Service for handling joystick remote control input without display dependencies."""

    def __init__(self, config: Optional[JoystickConfig] = None):
        """Initialize remote control service with optional configuration."""
        self.config = config or JoystickConfig()
        self._lock = threading.Lock()
        self._running = True
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

        self.joystick = None
        self.joystick_runner = None
        self.keyboard_runner = None
        self.ros_node = None
        self.ros_executor = None
        self.ros_runner = None
        self._topic_custom_mode = False
        self._topic_rl_gait = False
        # Latched presses by logical button name (x, a, y, b) from the
        # keyboard and the remote-controller topic; joystick presses are
        # edge-detected in consume_press().
        self._latched: dict[str, bool] = {n: False for n in self.BUTTONS}
        self._joystick_was_down = {n: False for n in self.BUTTONS}
        self._topic_was_down = {n: False for n in self.BUTTONS}

        self._init_keyboard_control()
        self._start_keyboard_thread()

        try:
            self._init_joystick()
            self._start_joystick_thread()
        except Exception as e:
            print(f"{e}, trying /remote_controller_state topic fallback...")
            try:
                self._init_topic_fallback()
            except Exception as topic_error:
                print(f"{topic_error}, downgrade to keyboard control")

    def get_operation_hint(self) -> str:
        if hasattr(self, "joystick") and getattr(self, "joystick") is not None:
            return "Joystick left axis for forward/backward/left/right, right axis for rotation left/right"
        if self.ros_node is not None:
            return "Remote controller topic or keyboard 'w'/'s'/'a'/'d'/'q'/'e' controls velocity, press 'Space' to stop."
        return "Press keyboard 'w'/'s' to increase/decrease vx; Press 'a'/'d' to increase/decrease vy; Press 'q'/'e' to increase/decrease vyaw, press 'Space' to stop."

    def get_custom_mode_operation_hint(self) -> str:
        if hasattr(self, "joystick") and getattr(self, "joystick") is not None:
            return "Press joystick button X to start custom mode."
        if self.ros_node is not None:
            return "Press remote controller button X or keyboard 'x' to start custom mode."
        return "Press keyboard 'x' to start custom mode."

    def get_fsm_operation_hint(self) -> str:
        if self.joystick is not None:
            return ("Remote: X=STAND, A=forward (WALK/TASK), Y=back, "
                    "B=ESTOP; keyboard: x, r, n, b. Ctrl+C exits.")
        return ("Keyboard: x=STAND, r=forward (WALK/TASK), n=back, b=ESTOP "
                "(remote X/A/Y/B on the topic). Ctrl+C exits.")

    def get_rl_gait_operation_hint(self) -> str:
        # Keep the mode-switch prompt consistent across joystick, ROS topic,
        # and keyboard input paths.
        return "Press remote controller button A or keyboard 'r' to start RL mode."

    def _init_keyboard_control(self):
        self.keyboard_start_custom_mode = False
        self.keyboard_start_rl_gait = False

    def _start_keyboard_thread(self):
        # Start a thread that reads stdin in cbreak mode and dispatches presses.
        self.keyboard_runner = threading.Thread(target=self._keyboard_listener, daemon=True)
        # Save original terminal attrs so we can restore later
        try:
            if sys.stdin.isatty():
                self._stdin_tty = True
                self._old_termios = termios.tcgetattr(sys.stdin.fileno())
            else:
                self._stdin_tty = False
                self._old_termios = None
        except Exception:
            self._stdin_tty = False
            self._old_termios = None

        # Ensure we attempt to clean up terminal on process exit
        try:
            atexit.register(self.close)
        except Exception:
            pass

        self.keyboard_runner.start()

    def _keyboard_listener(self):
        # Use cbreak mode so we can read key presses without requiring Enter.
        fd = None
        try:
            if not getattr(self, "_stdin_tty", False):
                # stdin is not a tty; nothing to do
                return
            fd = sys.stdin.fileno()
            tty.setcbreak(fd)
            while self._running:
                # small timeout to allow clean shutdown
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == "\x03":  # Ctrl-C
                        # Let the main program handle KeyboardInterrupt
                        continue
                    if ch == " ":
                        key = "space"
                    else:
                        key = ch
                    try:
                        self._handle_keyboard_press(key)
                    except Exception:
                        # swallow handler errors to keep listener alive
                        pass
        finally:
            # restore terminal settings if we changed them
            try:
                if fd is not None and getattr(self, "_old_termios", None) is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass

    def _handle_keyboard_press(self, key):
        message = None
        with self._lock:
            if key == "x":
                self.keyboard_start_custom_mode = True
            if key == "r":
                self.keyboard_start_rl_gait = True
            if key in self.KEYBOARD_BUTTONS:
                self._latched[self.KEYBOARD_BUTTONS[key]] = True
            if key == "w":
                old_x = self.vx
                self.vx = min(self.vx + 0.1, self.config.max_vx)
                message = f"VX: {old_x:.1f} => {self.vx:.1f}"
            if key == "s":
                old_x = self.vx
                self.vx = max(self.vx - 0.1, -self.config.max_vx)
                message = f"VX: {old_x:.1f} => {self.vx:.1f}"
            if key == "a":
                old_y = self.vy
                self.vy = min(self.vy + 0.1, self.config.max_vy)
                message = f"VY: {old_y:.1f} => {self.vy:.1f}"
            if key == "d":
                old_y = self.vy
                self.vy = max(self.vy - 0.1, -self.config.max_vy)
                message = f"VY: {old_y:.1f} => {self.vy:.1f}"
            if key == "q":
                old_yaw = self.vyaw
                self.vyaw = min(self.vyaw + 0.1, self.config.max_vyaw)
                message = f"VYaw: {old_yaw:.1f} => {self.vyaw:.1f}"
            if key == "e":
                old_yaw = self.vyaw
                self.vyaw = max(self.vyaw - 0.1, -self.config.max_vyaw)
                message = f"VYaw: {old_yaw:.1f} => {self.vyaw:.1f}"
            if key == "space":
                self.vx = 0
                self.vy = 0
                self.vyaw = 0
                message = "FULL STOP"
        if message is not None:
            print(message)

    def _init_topic_fallback(self) -> None:
        """Subscribe to the ROS2 remote-controller state as a fallback input."""
        if not rclpy.ok():
            raise RuntimeError("rclpy is not initialized")

        self.ros_node = rclpy.create_node("remote_control_topic_sub")
        self.ros_executor = SingleThreadedExecutor()

        def topic_callback(msg: RemoteControllerState):
            with self._lock:
                self.vx = -float(msg.ly) * self.config.max_vx
                self.vy = -float(msg.lx) * self.config.max_vy
                self.vyaw = -float(msg.rx) * self.config.max_vyaw

                if msg.hat_r:
                    self.vyaw = self.config.max_vyaw * 0.5
                elif msg.hat_l:
                    self.vyaw = -self.config.max_vyaw * 0.5
                if msg.hat_u:
                    self.vx = self.config.max_vx * 0.5
                elif msg.hat_d:
                    self.vx = -self.config.max_vx * 0.5

                # Keep the topic mapping aligned with the physical controller:
                # X starts Custom mode and A starts RL mode. ``getattr`` keeps
                # the callback compatible with older message definitions.
                # Button messages can be followed by an axis update before the
                # controller's 10 Hz polling loop runs. Latch the press until
                # it is consumed so short X/A presses are not missed.
                # The topic repeats the button state while held: latch
                # only on the press edge so one press is one event.
                for name in self.BUTTONS:
                    down = bool(getattr(msg, name, False))
                    if down and not self._topic_was_down[name]:
                        self._latched[name] = True
                        if name == "x":
                            self._topic_custom_mode = True
                        elif name == "a":
                            self._topic_rl_gait = True
                    self._topic_was_down[name] = down

        self.ros_node.create_subscription(
            RemoteControllerState,
            "/remote_controller_state",
            topic_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.ros_executor.add_node(self.ros_node)

        def ros_spinner():
            while self._running and rclpy.ok():
                self.ros_executor.spin_once(timeout_sec=0.1)

        self.ros_runner = threading.Thread(target=ros_spinner, daemon=True)
        self.ros_runner.start()
        print("Subscribed to /remote_controller_state topic as fallback")

    def _init_joystick(self) -> None:
        """Initialize and validate joystick connection using evdev."""
        try:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            joystick = None

            for device in devices:
                caps = device.capabilities()
                # print(f"Device {device.name}:")
                # print(f"Capabilities: {device.capabilities(verbose=True)}")

                # Check for both absolute axes and keys
                if evdev.ecodes.EV_ABS in caps and evdev.ecodes.EV_KEY in caps:
                    abs_info = caps.get(evdev.ecodes.EV_ABS, [])
                    # Look for typical gamepad axes
                    axes = [code for (code, info) in abs_info]
                    keys = caps.get(evdev.ecodes.EV_KEY, [])
                    # Require the gamepad buttons too: some non-gamepad
                    # devices (e.g. motherboard LED controllers) advertise
                    # joystick axes without any buttons.
                    has_axes = all(code in axes for code in [
                        self.config.x_axis, self.config.y_axis,
                        self.config.yaw_axis])
                    has_buttons = all(code in keys for code in [
                        self.config.custom_mode_button,
                        self.config.rl_gait_button])
                    if has_axes and has_buttons:
                        absinfo = {}
                        for code, info in abs_info:
                            absinfo[code] = info
                        self.axis_ranges = {
                            self.config.x_axis: absinfo[self.config.x_axis],
                            self.config.y_axis: absinfo[self.config.y_axis],
                            self.config.yaw_axis: absinfo[self.config.yaw_axis],
                        }
                        print(f"Found suitable joystick: {device.name}")
                        joystick = device
                        break

            if not joystick:
                raise RuntimeError("No suitable joystick found")

            self.joystick = joystick
            print(f"Selected joystick: {joystick.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize joystick: {e}")

    def _start_joystick_thread(self):
        """Start joystick polling thread."""
        self.joystick_runner = threading.Thread(target=self._run_joystick)
        self.joystick_runner.daemon = True
        self.joystick_runner.start()

    def _joystick_button(self, name: str):
        return {
            "x": self.config.custom_mode_button,
            "a": self.config.rl_gait_button,
            "y": self.config.back_button,
            "b": self.config.estop_button,
        }[name]

    def consume_press(self) -> Optional[str]:
        """Return one pending logical button press (x, a, y, b) or None.

        Keyboard and topic presses are latched until consumed; joystick
        buttons trigger once per press (edge-detected while polling).
        """
        active = set()
        if self.joystick is not None:
            active = set(self.joystick.active_keys())
        with self._lock:
            for name in self.BUTTONS:
                down = self._joystick_button(name) in active
                edge = down and not self._joystick_was_down[name]
                self._joystick_was_down[name] = down
                if edge or self._latched[name]:
                    self._latched[name] = False
                    return name
        return None

    def start_custom_mode(self) -> bool:
        """Check if custom mode button is pressed."""
        joystick_triggered = (
            self.joystick is not None
            and self.config.custom_mode_button in self.joystick.active_keys()
        )
        with self._lock:
            triggered = (
                joystick_triggered
                or self._topic_custom_mode
                or self.keyboard_start_custom_mode
            )
            self._topic_custom_mode = False
            self.keyboard_start_custom_mode = False
            return triggered

    def start_rl_gait(self) -> bool:
        """Check if gait button is pressed."""
        joystick_triggered = (
            self.joystick is not None
            and self.config.rl_gait_button in self.joystick.active_keys()
        )
        with self._lock:
            triggered = (
                joystick_triggered
                or self._topic_rl_gait
                or self.keyboard_start_rl_gait
            )
            self._topic_rl_gait = False
            self.keyboard_start_rl_gait = False
            return triggered

    def _run_joystick(self):
        """Poll joystick events."""
        while self._running:
            try:
                # read one event
                event = self.joystick.read_one()
                if event:
                    if event.type == evdev.ecodes.EV_ABS:
                        self._handle_axis(event.code, event.value)
                else:
                    time.sleep(0.01)
            except Exception as e:
                if not self._running:  # If the exception was caused by shutdown, no need to log
                    break
                print(f"Error in joystick polling loop: {e}")
                time.sleep(0.05)

    def _handle_axis(self, code: int, value: int):
        """Handle axis events."""
        with self._lock:
            if code == self.config.x_axis:
                self.vx = self._scale(value, self.config.max_vx, self.config.control_threshold, code)
                # print("value x:", self.vx)
            elif code == self.config.y_axis:
                self.vy = self._scale(value, self.config.max_vy, self.config.control_threshold, code)
                # print("value y:", self.vy)
            elif code == self.config.yaw_axis:
                self.vyaw = self._scale(value, self.config.max_vyaw, self.config.control_threshold, code)
                # print("value yaw:", self.vyaw)

    def _scale(self, value: float, max: float, threshold: float, axis_code: int) -> float:
        """Scale joystick input to velocity command using actual axis ranges."""
        absinfo = self.axis_ranges[axis_code]
        min_in = absinfo.min
        max_in = absinfo.max

        mapped_value = ((value - min_in) / (max_in - min_in) * 2 - 1) * max
        # print(f"Axis {axis_code}, value {value} min_in {min_in}, max_in {max_in}: {value} => {mapped_value}")

        if abs(mapped_value) < threshold:
            return 0.0
        return -mapped_value

    def get_vx_cmd(self) -> float:
        """Get forward velocity command."""
        with self._lock:
            return self.vx

    def get_vy_cmd(self) -> float:
        """Get lateral velocity command."""
        with self._lock:
            return self.vy

    def get_vyaw_cmd(self) -> float:
        """Get yaw velocity command."""
        with self._lock:
            return self.vyaw

    def close(self):
        """Clean up resources."""
        self._running = False
        # try restore stdin terminal settings if we changed them
        try:
            if getattr(self, "_stdin_tty", False) and getattr(self, "_old_termios", None) is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
        except Exception:
            pass
        if hasattr(self, "joystick") and getattr(self, "joystick") is not None:
            try:
                self.joystick.close()
            except Exception as e:
                print(f"Error closing joystick: {e}")
        if hasattr(self, "joystick_runner") and getattr(self, "joystick_runner") is not None:
            try:
                self.joystick_runner.join(timeout=1.0)
                if self.joystick_runner.is_alive():
                    print("Joystick thread didn't exit within the time limit")
            except Exception as e:
                print(f"Error waiting for joystick thread to end: {e}")
        if hasattr(self, "keyboard_runner") and getattr(self, "keyboard_runner") is not None:
            try:
                self.keyboard_runner.join(timeout=1.0)
                if self.keyboard_runner.is_alive():
                    print("Keyboard thread didn't exit within the time limit")
            except Exception as e:
                print(f"Error waiting for keyboard thread to end: {e}")
        if hasattr(self, "ros_runner") and getattr(self, "ros_runner") is not None:
            try:
                self.ros_runner.join(timeout=1.0)
                if self.ros_runner.is_alive():
                    print("ROS topic thread didn't exit within the time limit")
            except Exception as e:
                print(f"Error waiting for ROS topic thread to end: {e}")
        if hasattr(self, "ros_node") and getattr(self, "ros_node") is not None:
            try:
                self.ros_node.destroy_node()
            except Exception as e:
                print(f"Error destroying ROS topic node: {e}")
        if hasattr(self, "ros_executor") and getattr(self, "ros_executor") is not None:
            try:
                self.ros_executor.shutdown()
            except Exception as e:
                print(f"Error shutting down ROS topic executor: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

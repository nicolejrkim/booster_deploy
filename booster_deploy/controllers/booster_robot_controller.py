from __future__ import annotations
import json
import logging
import signal
import time
import threading
import multiprocessing as mp
from copy import deepcopy
from multiprocessing import synchronize

import numpy as np
import torch

import rclpy
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from booster_interface.msg import BoosterApiReqMsg, LowState, LowCmd, MotorCmd
from booster_interface.srv import RpcService
from rclpy.qos import DurabilityPolicy
from std_msgs.msg import String


class _RobotModeInt:
    """Mode values used by the Booster Loco RPC API."""

    kDamping = 0
    kWalking = 2
    kCustom = 3


_LOC_API_CHANGE_MODE = 2000
_LOC_API_GET_STATUS = 2018
_LOC_API_GET_UP_WITH_MODE = 2025

from .controller_cfg import ControllerCfg
from .base_controller import BaseController, BoosterRobot
from ..utils.synced_array import SyncedArray
from ..utils.metrics import SyncedMetrics
from ..utils.isaaclab import math as lab_math
from ..utils.remote_control_service import RemoteControlService


logger = logging.getLogger("booster_deploy")
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


class BoosterRobotPortal:
    synced_state: SyncedArray
    synced_command: SyncedArray
    synced_action: SyncedArray
    exit_event: synchronize.Event

    def __init__(self, cfg: ControllerCfg) -> None:
        self.cfg = cfg

        self.robot = BoosterRobot(cfg.robot)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if not rclpy.ok():
            rclpy.init()
        self.remoteControlService = RemoteControlService()
        # Use multiprocessing.Event for inter-process communication
        self.exit_event = mp.Event()
        self.velocity_commands_enabled_event = mp.Event()
        self.task_start_event = mp.Event()
        self.low_state_received_event = threading.Event()
        # Deployment state machine (see booster_deploy.fsm): the portal
        # requests states, the executor process acknowledges the active one
        # and may request follow-up states itself (policy stop/finish).
        self.fsm_requested = mp.Value("i", 0)
        self.fsm_active = mp.Value("i", 0)
        self.fsm_executor_request = mp.Value("i", -1)
        self._fsm = None
        self._fsm_walk_first = False
        # State requested over the booster_deploy/fsm_request topic (monitor,
        # `ros2 topic pub`), consumed by the supervisor loop.
        self._fsm_topic_request: str | None = None
        self._fsm_topic_lock = threading.Lock()
        self.is_running = True
        def signal_handler(sig, frame):
            if mp.current_process().name == "MainProcess":
                print("\nKeyboard interrupt received. Shutting down...")
            self.exit_event.set()

        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)

        self._init_synced_buffer()
        self._init_metrics()

        self._cleanup_done = False
        self._safety_abort = False
        self.inference_process = None  # Inference process reference
        self.low_cmd_publisher: rclpy.publisher.Publisher = None
        self.fsm_state_publisher = None
        self.fsm_result_publisher = None
        self.low_state_thread = None
        self.low_cmd_process: mp.Process | None = None

        # Initialize communication. Callbacks may start immediately and
        # reference `is_running` and `exit_event`, so ensure those are set.
        self._init_communication()

    def _init_synced_buffer(self):
        action_dtype = np.dtype(
            [
                ("dof_target", float, (self.robot.num_joints,)),
                ("stiffness", float, (self.robot.num_joints,)),
                ("damping", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_action = SyncedArray(
            "action",
            shape=(1,),
            dtype=action_dtype,
        )
        self._action_buf = np.ndarray((1,), dtype=action_dtype)

        state_dtype = np.dtype(
            [
                ("root_rpy_w", float, (3,)),
                ("root_ang_vel_b", float, (3,)),
                ("root_pos_w", float, (3,)),
                ("root_lin_vel_w", float, (3,)),
                ("joint_pos", float, (self.robot.num_joints,)),
                ("joint_vel", float, (self.robot.num_joints,)),
                ("feedback_torque", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_state = SyncedArray(
            "state",
            shape=(1,),
            dtype=state_dtype
        )
        self._state_buf = np.zeros((1,), dtype=state_dtype)

        command_dtype = np.dtype(
            [
                ("vx", float),
                ("vy", float),
                ("vyaw", float),
            ]
        )
        self.synced_command = SyncedArray(
            "command",
            shape=(1,),
            dtype=command_dtype,
        )

    def _init_metrics(self):
        # initialize cross-process synced metrics
        max_events = self.cfg.booster.metrics_max_events
        self.metrics = {
            "low_state_handler": SyncedMetrics(
                "low_state_handler", max_events=max_events
            ),
            "policy_step": SyncedMetrics(
                "policy_step", max_events=max_events
            ),
        }

    def _init_communication(self) -> None:
        try:
            self.create_low_cmd_publisher("booster_deploy_low_cmd_pub")
            self._start_low_state_subscription()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _start_low_state_subscription(self) -> None:
        """Start ROS 2 subscription loop on a dedicated thread.

        The subscription is run on a dedicated thread and spins a
        SingleThreadedExecutor for the `/low_state` topic.
        """

        def low_state_service_executor():
            self.logger.info("Low state subscription started")
            low_state_node = rclpy.create_node("booster_deploy_low_state_sub")
            low_state_node.create_subscription(
                LowState,
                "/low_state",
                self._low_state_handler,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                ),
            )
            low_state_node.create_subscription(
                String,
                "booster_deploy/fsm_request",
                self._fsm_request_handler,
                QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE,
                           history=HistoryPolicy.KEEP_LAST),
            )

            executor = SingleThreadedExecutor()
            executor.add_node(low_state_node)

            try:
                # loop: check exit_event and rclpy.ok()
                while rclpy.ok() and not self.exit_event.is_set():
                    executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                pass
            except Exception as exc:
                # Suppress RCLError if we are shutting down
                is_rcl_error = "RCLError" in type(exc).__name__
                is_shutting_down = self.exit_event.is_set() or not rclpy.ok()

                if is_rcl_error and is_shutting_down:
                    pass
                else:
                    self.logger.error(
                        "Low state subscription executor stopped: %s",
                        exc,
                        exc_info=True
                    )
            finally:
                executor.shutdown()
                low_state_node.destroy_node()
            self.logger.info("Low state subscription stopped")

        self.low_state_thread = threading.Thread(
            target=low_state_service_executor,
            name="low_state_executor",
            daemon=True,
        )
        self.low_state_thread.start()

    def _low_state_handler(self, low_state_msg: LowState):
        self.metrics["low_state_handler"].mark()
        try:
            if not self.is_running or self.exit_event.is_set():
                return

            # collect state data
            rpy = np.array(low_state_msg.imu_state.rpy, dtype=np.float32)
            gyro = np.array(low_state_msg.imu_state.gyro, dtype=np.float32)
            dof_pos = np.zeros(self.robot.num_joints, dtype=np.float32)
            dof_vel = np.zeros(self.robot.num_joints, dtype=np.float32)
            fb_torque = np.zeros(self.robot.num_joints, dtype=np.float32)

            for i, motor in enumerate(low_state_msg.motor_state_serial):
                dof_pos[i] = motor.q
                dof_vel[i] = motor.dq
                fb_torque[i] = motor.tau_est

            self._state_buf[0]["root_rpy_w"][:] = rpy
            self._state_buf[0]["root_ang_vel_b"][:] = gyro
            self._state_buf[0]["root_pos_w"][:] = np.zeros(
                3, dtype=np.float32
            )
            self._state_buf[0]["root_lin_vel_w"][:] = np.zeros(
                3, dtype=np.float32
            )
            self._state_buf[0]["joint_pos"][:] = dof_pos
            self._state_buf[0]["joint_vel"][:] = dof_vel
            self._state_buf[0]["feedback_torque"][:] = fb_torque
            self.synced_state.write(self._state_buf)
            self.low_state_received_event.set()

            # update velocity commands to synced_command
            cmd = np.zeros((1,), dtype=self.synced_command.dtype)
            cmd[0]["vx"] = self.remoteControlService.get_vx_cmd()
            cmd[0]["vy"] = self.remoteControlService.get_vy_cmd()
            cmd[0]["vyaw"] = self.remoteControlService.get_vyaw_cmd()
            self.synced_command.write(cmd)

        except Exception as e:
            self.logger.error(f"Error in _low_state_handler: {e}")
            self.running = False
            self.exit_event.set()

    def _fsm_request_handler(self, msg: String) -> None:
        with self._fsm_topic_lock:
            self._fsm_topic_request = msg.data.strip().upper()

    def _get_up(self) -> bool:
        """Booster's built-in get-up into Walking mode (blocking, polled)."""
        version = int(self.cfg.booster.getup_version)
        self.logger.info("Requesting get-up (version %d)...", version)
        ok, _ = self._call_booster_rpc(
            _LOC_API_GET_UP_WITH_MODE,
            {"mode": _RobotModeInt.kWalking, "version": version},
        )
        if not ok:
            self.logger.error("Get-up request was refused")
            return False
        deadline = time.perf_counter() + self.cfg.booster.getup_timeout_s
        while time.perf_counter() < deadline and not self.exit_event.is_set():
            time.sleep(0.5)
            status_ok, status = self._call_booster_rpc(_LOC_API_GET_STATUS)
            if (
                status_ok and status is not None
                and int(status.get("current_mode", -1))
                == _RobotModeInt.kWalking
                and self._posture_is_upright()
            ):
                self.logger.info("Get-up finished; robot is in Walking mode")
                return True
        self.logger.error("Get-up did not finish within %.0fs",
                          self.cfg.booster.getup_timeout_s)
        return False

    def create_low_cmd_publisher(self, name):
        self.publish_node = rclpy.create_node(name)
        publisher = self.publish_node.create_publisher(
            LowCmd,
            "joint_ctrl",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST
            )
        )
        self.low_cmd_publisher = publisher

        # construct low_cmd struct
        self.low_cmd = LowCmd()  # type: ignore
        self.low_cmd.cmd_type = LowCmd.CMD_TYPE_SERIAL   # type: ignore
        motor_cmd_buf = [
            MotorCmd() for _ in range(self.robot.num_joints)
        ]  # type: ignore
        for i in range(self.robot.num_joints):
            motor_cmd_buf[i].q = 0.0
            motor_cmd_buf[i].dq = 0.0
            motor_cmd_buf[i].tau = 0.0
            motor_cmd_buf[i].kp = 0.0
            motor_cmd_buf[i].kd = 0.0
            motor_cmd_buf[i].weight = 0.0
        self.low_cmd.motor_cmd.extend(motor_cmd_buf)
        self.motor_cmd = self.low_cmd.motor_cmd

        self.rpc_service_client = self.publish_node.create_client(
            RpcService, "booster_rpc_service"
        )
        # Current FSM state for monitors (latched so late joiners get it).
        self.fsm_state_publisher = self.publish_node.create_publisher(
            String,
            "booster_deploy/fsm_state",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )
        # Verdict on each transition request, for the monitor: "OK STAND",
        # "REJECTED TASK: not allowed from IDLE", "FAILED STAND: <why>".
        self.fsm_result_publisher = self.publish_node.create_publisher(
            String,
            "booster_deploy/fsm_result",
            QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST),
        )

        return publisher

    def _call_booster_rpc(
        self,
        api_id: int,
        body: dict[str, object] | None = None,
    ) -> tuple[bool, dict | None]:
        """Call Booster's ROS2 Loco RPC and parse its JSON response."""
        if self.rpc_service_client is None:
            self.logger.error("booster_rpc_service client is not initialized")
            return False, None

        try:
            request = RpcService.Request()
            request.msg = BoosterApiReqMsg()
            request.msg.api_id = api_id
            request.msg.body = json.dumps(body) if body is not None else ""

            future = self.rpc_service_client.call_async(request)
            rclpy.spin_until_future_complete(
                self.publish_node, future, timeout_sec=3.0)
            if not future.done():
                self.rpc_service_client.remove_pending_request(future)
                self.logger.error("booster_rpc_service call timed out")
                return False, None
            result = future.result()
        except Exception as exc:
            self.logger.error("booster_rpc_service call failed: %s", exc)
            return False, None

        if result is None:
            self.logger.error("booster_rpc_service returned no result")
            return False, None

        parsed = None
        if result.msg.body:
            try:
                parsed = json.loads(result.msg.body)
            except json.JSONDecodeError:
                self.logger.warning(
                    "booster_rpc_service returned non-JSON body: %s",
                    result.msg.body,
                )

        if result.msg.status != 0:
            self.logger.warning(
                "booster_rpc_service status=%s body=%s",
                result.msg.status,
                result.msg.body,
            )
            return False, parsed
        return True, parsed

    def _change_robot_mode(self, mode_name: str) -> bool:
        mode_values = {
            "damping": _RobotModeInt.kDamping,
            "walking": _RobotModeInt.kWalking,
            "walk": _RobotModeInt.kWalking,
            "custom": _RobotModeInt.kCustom,
        }
        normalized = mode_name.strip().lower()
        if normalized not in mode_values:
            raise ValueError(
                f"Unsupported robot mode {mode_name!r}; "
                "expected 'damping', 'walking' (or 'walk'), or 'custom'"
            )

        if not self.rpc_service_client.wait_for_service(timeout_sec=15.0):
            self.logger.error("booster_rpc_service is unavailable")
            return False

        expected_mode = mode_values[normalized]
        display_name = (
            "Walking" if expected_mode == _RobotModeInt.kWalking
            else normalized.capitalize()
        )
        for _ in range(20):
            ok, _ = self._call_booster_rpc(
                _LOC_API_CHANGE_MODE,
                {"mode": expected_mode},
            )
            if ok:
                status_ok, status = self._call_booster_rpc(
                    _LOC_API_GET_STATUS
                )
                if (
                    status_ok
                    and status is not None
                    and int(status.get("current_mode", -1)) == expected_mode
                ):
                    self.logger.info("Robot switched to %s mode", display_name)
                    return True
            time.sleep(0.5)

        self.logger.error("Failed to switch robot to %s mode", display_name)
        return False

    def _prime_custom_command(self) -> None:
        """Preload the PVT-style PD hold command before leaving PVT."""
        state = self.synced_state.read()[0]
        joint_pos = state["joint_pos"]
        prepare_state = self.robot.cfg.prepare_state
        for i in range(self.robot.num_joints):
            self.motor_cmd[i].q = float(joint_pos[i])
            self.motor_cmd[i].kp = float(prepare_state.stiffness[i])
            self.motor_cmd[i].kd = float(prepare_state.damping[i])
            self.motor_cmd[i].tau = 0.0
        # The low-level controller keeps this command when Custom is entered,
        # until the locomotion policy publishes its first action.
        self.low_cmd_publisher.publish(self.low_cmd)

    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while not self.exit_event.is_set():
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)

        if self.exit_event.is_set():
            return False

        while (
            rclpy.ok()
            and not self.exit_event.is_set()
            and not self.low_state_received_event.wait(timeout=0.5)
        ):
            self.logger.info("Waiting for first '/low_state' message")

        if not self.low_state_received_event.is_set():
            self.logger.error("No valid '/low_state'; refusing Custom mode")
            return False

        # Python-side walking preparation is allowed only from an upright
        # posture.  This is a single check at the X trigger; interpolation
        # itself intentionally has no additional posture checks.
        if self.cfg.robot.prepare_mode.strip().lower() == "walking":
            state = self.synced_state.read()[0]
            rpy = torch.from_numpy(state["root_rpy_w"]).to(dtype=torch.float32)
            projected_gravity = lab_math.quat_apply_inverse(
                lab_math.quat_from_euler_xyz(*rpy).squeeze(),
                torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            )
            if projected_gravity[2] > -0.5:
                self.logger.error(
                    "Refusing walk preparation: robot posture is unsafe "
                    "(projected_gravity[2]=%.3f). Switching to damping mode.",
                    float(projected_gravity[2]),
                )
                self._safety_abort = True
                self._change_robot_mode("damping")
                return False

            while rclpy.ok() and self.low_cmd_publisher.get_subscription_count() == 0:
                self.logger.info("Waiting for '/joint_ctrl' subscriber, retry in 0.5s")
                time.sleep(0.5)

            # Match standing preparation's hold position and PVT-style PD
            # gains.  Locomotion takes over only after Custom is entered.
            # Send exactly one command while PVT still owns the robot; the
            # low-level controller retains it across the Custom transition.
            self._prime_custom_command()
            # Give the low-level/PVT path time to receive and retain the
            # non-zero-gain hold command before relinquishing control.
            time.sleep(0.1)

            # The Python locomotion policy publishes joint_ctrl commands.  The
            # robot must be in Custom mode for those commands to be accepted;
            # merely starting the inference process leaves a robot that was in
            # PVT mode under the previous controller.
            if not self._change_robot_mode("custom"):
                self.logger.error(
                    "Failed to switch to Custom mode for walking preparation"
                )
                return False

            # Walking preparation uses the Python locomotion policy directly;
            # do not run the PVT interpolation.
            self.logger.info(
                "Walking preparation accepted; starting zero-command locomotion"
            )
            return True

        while rclpy.ok() and self.low_cmd_publisher.get_subscription_count() == 0:
            self.logger.info("Waiting for '/joint_ctrl' subscriber, retry in 0.5s")
            time.sleep(0.5)

        self.logger.info("Subscriber found, starting control loop")        

        prepare_state = self.robot.cfg.prepare_state
        init_joint_pos = self.synced_state.read()[0]['joint_pos']
        for i in range(self.robot.num_joints):
            self.motor_cmd[i].q = init_joint_pos[i]
            self.motor_cmd[i].kp = float(prepare_state.stiffness[i])
            self.motor_cmd[i].kd = float(prepare_state.damping[i])

        self.low_cmd_publisher.publish(self.low_cmd)
        time.sleep(0.1)

        if not self._change_robot_mode("custom"):
            self.logger.error("Failed to switch to Custom mode")
            return False

        trans = np.linspace(init_joint_pos, prepare_state.joint_pos, num=500)
        start_time = time.perf_counter()
        for i in range(500):
            for j in range(self.robot.num_joints):
                self.motor_cmd[j].q = trans[i][j]
            self.low_cmd_publisher.publish(self.low_cmd)
            while time.perf_counter() < start_time + (i + 1) * 0.002:
                time.sleep(0.0002)
        if self.cfg.robot.prepare_mode.strip().lower() == "walking":
            self.logger.info(
                "Prepare pose reached; starting Python RL walking with zero commands"
            )
        self.logger.info("Custom mode started, initialized with prepare pose")
        return True

    def _build_prepare_cfg(self):
        """Build the matching robot locomotion config for walking preparation."""
        robot_name = self.cfg.robot.name.lower()
        if "t2" in robot_name:
            from tasks.locomotion.robots.t2 import T2WalkTaskCfg
            walk_cfg = T2WalkTaskCfg()
        elif "t1" in robot_name:
            from tasks.locomotion.robots.t1 import T1WalkControllerCfg1
            walk_cfg = T1WalkControllerCfg1()
        elif "k1" in robot_name:
            from tasks.locomotion.robots.k1 import K1WalkTaskCfg
            walk_cfg = K1WalkTaskCfg()
        else:
            raise RuntimeError(f"No locomotion preparation policy for robot {self.cfg.robot.name!r}")
        prepare_cfg = deepcopy(self.cfg)
        # The hand-off command and the prepare controller must use the
        # matching locomotion robot gains, not gains inherited from a task
        # policy that happens to run on the same robot.
        prepare_cfg.robot = deepcopy(walk_cfg.robot)
        prepare_cfg.policy = deepcopy(walk_cfg.policy)
        prepare_cfg.vel_command = deepcopy(walk_cfg.vel_command)
        prepare_cfg.robot.prepare_mode = "walking"
        return prepare_cfg

    def start_rl_gait_conditionally(self, wait_for_trigger: bool = True):
        """Start RL mode and spawn inference process and publisher thread."""
        if wait_for_trigger:
            print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
            while not self.exit_event.is_set():
                if self.remoteControlService.start_rl_gait():
                    break
                time.sleep(0.1)

        if self.exit_event.is_set():
            return False

        # In walking preparation, start the matching zero-command locomotion
        # policy immediately; the child switches to the task policy after A/r.
        process_cfg = self.cfg
        prepare_then_task = (
            self.cfg.robot.prepare_mode.strip().lower() == "walking"
        )
        if prepare_then_task:
            process_cfg = self._build_prepare_cfg()
        self.inference_process = mp.Process(
            target=BoosterRobotPortal.inference_process_func,
            args=(process_cfg, self, prepare_then_task),
            daemon=True,
        )
        self.inference_process.start()
        self.logger.info("Inference process started")

        # In walking preparation the locomotion policy is command-free, so do
        # not advertise velocity controls until the task policy is enabled.
        if not prepare_then_task and self.cfg.vel_command is not None:
            print(f"{self.remoteControlService.get_operation_hint()}")
        return True

    def cleanup(self) -> None:
        """Clean up resources (idempotent)."""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        self.logger.info("Doing cleanup...")

        # stop threads and processes
        self.is_running = False
        self.exit_event.set()

        # wait for inference process
        if (
            self.inference_process is not None
            and self.inference_process.is_alive()
        ):
            self.logger.info("Waiting for inference process...")
            self.inference_process.join(timeout=2.0)
            if self.inference_process.is_alive():
                self.logger.warning(
                    "Inference process did not stop, terminating...")
                self.inference_process.terminate()
                self.inference_process.join(timeout=1.0)

        # close communications
        try:
            self.remoteControlService.close()
        except Exception as e:
            self.logger.error(f"Error closing remote control: {e}")

        if self.low_cmd_process is not None and self.low_cmd_process.is_alive():
            self.logger.info("Waiting for low cmd publisher process...")
            self.low_cmd_process.join(timeout=2.0)
            if self.low_cmd_process.is_alive():
                self.logger.warning(
                    "Low cmd publisher process did not stop, terminating...")
                self.low_cmd_process.terminate()
                self.low_cmd_process.join(timeout=1.0)

        try:
            thread = self.low_state_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

        except Exception as e:
            self.logger.error(f"Error waiting for low state thread: {e}")

        if rclpy.ok():
            rclpy.shutdown()

        self.logger.info("Cleanup complete")

        # Print synced metrics summary to stdout
        for name, metric in self.metrics.items():
            stats = metric.compute()
            print(
                f"METRICS {name}: count={stats['count']}, "
                f"freq={stats['freq_hz']:.3f}Hz, "
                f"mean_period={stats['mean_period_s']}, "
                f"min={stats['min_period_s']}, max={stats['max_period_s']}"
            )
        # Release the shared-memory segments explicitly: deploy.py exits
        # with os._exit afterwards, which skips the atexit cleanup.
        for arr in (self.synced_state, self.synced_command, self.synced_action):
            arr.cleanup()
        for metric in self.metrics.values():
            metric._arr.cleanup()

    # ------------------------------------------------------------------ FSM
    def _posture_is_upright(self) -> bool:
        state = self.synced_state.read()[0]
        rpy = torch.from_numpy(state["root_rpy_w"]).to(dtype=torch.float32)
        projected_gravity = lab_math.quat_apply_inverse(
            lab_math.quat_from_euler_xyz(*rpy).squeeze(),
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
        )
        if projected_gravity[2] > -0.5:
            self.logger.error(
                "Robot posture is unsafe (projected_gravity[2]=%.3f)",
                float(projected_gravity[2]),
            )
            return False
        return True

    def _enter_custom_mode(self) -> str:
        """Prime a PD hold of the current pose and switch to Custom mode.

        Returns ``"ok"``, ``"unsafe"`` (posture check failed) or ``"failed"``.
        """
        if not self._posture_is_upright():
            return "unsafe"
        while (rclpy.ok()
               and self.low_cmd_publisher.get_subscription_count() == 0):
            if self.exit_event.is_set():
                return "failed"
            self.logger.info(
                "Waiting for '/joint_ctrl' subscriber, retry in 0.5s")
            time.sleep(0.5)
        # Send exactly one hold command while the built-in controller still
        # owns the robot; the low-level controller retains it across the
        # Custom transition until the executor publishes.
        self._prime_custom_command()
        time.sleep(0.1)
        if not self._change_robot_mode("custom"):
            return "failed"
        return "ok"

    def _build_walk_cfg(self):
        """Locomotion config for the WALK state, or None if the task is it."""
        try:
            walk_cfg = self._build_prepare_cfg()
        except RuntimeError as exc:
            self.logger.warning("No WALK state: %s", exc)
            return None
        if (walk_cfg.policy.checkpoint_path
                == self.cfg.policy.checkpoint_path):
            return None
        return walk_cfg

    def _fsm_request(self, target: str, wait_ack: bool = True,
                     timeout_s: float = 3.0) -> bool:
        from ..fsm import state_index
        index = state_index(target)
        self.fsm_requested.value = index
        if not wait_ack:
            return True
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self.fsm_active.value == index:
                return True
            if self.exit_event.is_set():
                return False
            time.sleep(0.01)
        self.logger.error(
            "Executor did not enter %s within %.1fs", target, timeout_s)
        return False

    def fsm_transition(self, target: str) -> bool:
        """Perform a validated state transition (mode RPCs + executor)."""
        from ..fsm import ESTOP, IDLE, STAND
        fsm = self._fsm
        if not fsm.can(target):
            self.logger.warning(
                "Transition %s -> %s is not allowed (allowed: %s)",
                fsm.current, target, ", ".join(fsm.targets()) or "none")
            self._fsm_publish_result(
                "REJECTED", target, f"not allowed from {fsm.current}")
            return False
        current = fsm.current
        ok = True
        if current == IDLE and target == STAND:
            status = self._enter_custom_mode()
            if status == "unsafe":
                self.logger.error("Refusing STAND; switching to Damping mode")
                self._safety_abort = True
                self._fsm_request(ESTOP, wait_ack=False)
                self._change_robot_mode("damping")
                fsm.switch(ESTOP)
                self._fsm_print_state()
                self._fsm_publish_result(
                    "FAILED", target, "unsafe posture, switched to Damping")
                return False
            ok = status == "ok" and self._fsm_request(STAND)
            if not ok:
                self._change_robot_mode("walking")
        elif target == ESTOP:
            # The executor publishes one damping command and stops publishing
            # before the firmware leaves Custom mode.
            self._fsm_request(ESTOP)
            ok = self._change_robot_mode("damping")
        elif target == IDLE:
            self._fsm_request(IDLE)
            if current == ESTOP and not self._posture_is_upright():
                # A limp robot is usually on the floor: use the firmware's
                # get-up, which ends in Walking mode.
                ok = self._get_up()
            else:
                ok = self._change_robot_mode("walking")
        else:
            ok = self._fsm_request(target)
        if not ok:
            self.logger.error("Transition %s -> %s failed", current, target)
            self._fsm_publish_result(
                "FAILED", target, f"{current} -> {target} failed")
            return False
        fsm.switch(target)
        self._fsm_print_state()
        self._fsm_publish_result("OK", target)
        return True

    def _fsm_publish_result(self, verdict: str, target: str,
                            reason: str = "") -> None:
        """Report the outcome of a transition request on
        ``booster_deploy/fsm_result`` (the monitor shows it)."""
        text = f"{verdict} {target}" + (f": {reason}" if reason else "")
        if self.fsm_result_publisher is not None:
            self.fsm_result_publisher.publish(String(data=text))

    def _fsm_map_press(self, press: str):
        """Map a logical button (x, a, y, b) to a target state, or None."""
        from ..fsm import ESTOP, IDLE, STAND, TASK, WALK
        current = self._fsm.current
        if press == "x":
            return STAND if current == IDLE else None
        if press == "b":
            return ESTOP if current != ESTOP else None
        if press == "a":  # forward
            if current == STAND:
                return WALK if self._fsm_walk_first else TASK
            if current == WALK:
                return TASK
            return None
        if press == "y":  # back
            if current == TASK:
                return WALK if self._fsm_walk_first else STAND
            if current == WALK:
                return STAND
            if current in (STAND, ESTOP):
                return IDLE
        return None

    def _fsm_sync_from_executor(self) -> None:
        """Apply a transition initiated by the executor (policy stop/finish)."""
        from ..fsm import ESTOP, state_name
        index = self.fsm_executor_request.value
        if index < 0:
            return
        self.fsm_executor_request.value = -1
        target = state_name(index)
        self.logger.info("Executor switched to %s", target)
        self.fsm_requested.value = index
        if target == ESTOP:
            self._safety_abort = True
            self._change_robot_mode("damping")
        if self._fsm.current != target:
            self._fsm.switch(target)
            self._fsm_print_state()

    def _fsm_check_robot_mode(self) -> None:
        """Mode watchdog: the firmware can leave Custom mode on its own (a
        fall protection, a restart, an operator app).  If it did while a
        Custom state is active, stop publishing and follow the robot."""
        from ..fsm import CUSTOM_STATES, ESTOP, IDLE
        if self._fsm.current not in CUSTOM_STATES:
            return
        ok, status = self._call_booster_rpc(_LOC_API_GET_STATUS)
        if not ok or status is None:
            return
        mode = int(status.get("current_mode", -1))
        if mode == _RobotModeInt.kCustom:
            return
        target = ESTOP if mode == _RobotModeInt.kDamping else IDLE
        self.logger.error(
            "Robot left Custom mode (current_mode=%d) while in %s; "
            "commands are being ignored. Switching to %s.",
            mode, self._fsm.current, target)
        self._fsm_request(target)
        self._fsm.switch(target)
        self._fsm_print_state()

    def _fsm_print_state(self) -> None:
        hints = []
        labels = (("x", "x/X"), ("a", "r/A"), ("y", "n/Y"), ("b", "b/B"))
        for press, label in labels:
            target = self._fsm_map_press(press)
            if target is not None:
                hints.append(f"{label}: {target}")
        print(f"[FSM] state: {self._fsm.current}   "
              f"({', '.join(hints)}; Ctrl+C: exit)")
        if self.fsm_state_publisher is not None:
            self.fsm_state_publisher.publish(String(data=self._fsm.current))

    def run(self):
        """Main loop: supervise the deployment state machine (10 Hz).

        States and transitions are described in ``booster_deploy.fsm``.
        """
        from ..fsm import (
            CUSTOM_STATES, ESTOP, IDLE, STATES, TRANSITIONS, StateMachine,
        )
        from ..fsm.executor import fsm_process_func

        print("Initialization complete.")

        prepare_mode = self.cfg.robot.prepare_mode.strip().lower()
        if prepare_mode not in ("walking", "standing"):
            raise ValueError(
                f"Unsupported prepare_mode {self.cfg.robot.prepare_mode!r}; "
                "expected 'walking' or 'standing'"
            )
        exit_mode = self.cfg.booster.exit_mode.strip().lower()
        exit_mode = "walking" if exit_mode == "walk" else exit_mode
        if exit_mode not in ("walking", "damping"):
            raise ValueError(
                f"Unsupported exit_mode {self.cfg.booster.exit_mode!r}; "
                "expected 'walking' or 'damping'"
            )

        while (
            rclpy.ok()
            and not self.exit_event.is_set()
            and not self.low_state_received_event.wait(timeout=0.5)
        ):
            self.logger.info("Waiting for first '/low_state' message")
        if not self.low_state_received_event.is_set():
            self.logger.error("No valid '/low_state'; not starting")
            return

        walk_cfg = self._build_walk_cfg()
        self._fsm = StateMachine(STATES, TRANSITIONS, IDLE)
        self._fsm_walk_first = (
            prepare_mode == "walking" and walk_cfg is not None)

        self.inference_process = mp.Process(
            target=fsm_process_func,
            args=(self, self.cfg, walk_cfg),
            daemon=True,
        )
        self.inference_process.start()
        self.logger.info("FSM executor process started")
        print(self.remoteControlService.get_fsm_operation_hint())
        self._fsm_print_state()

        next_mode_check = time.perf_counter()
        while self.is_running and not self.exit_event.is_set():
            self._fsm_sync_from_executor()
            if time.perf_counter() >= next_mode_check:
                next_mode_check += self.cfg.booster.mode_check_period_s
                self._fsm_check_robot_mode()
            press = self.remoteControlService.consume_press()
            if press is not None:
                target = self._fsm_map_press(press)
                if target is None:
                    self.logger.warning(
                        "Button '%s' has no effect in %s",
                        press, self._fsm.current)
                else:
                    self.fsm_transition(target)
            with self._fsm_topic_lock:
                requested, self._fsm_topic_request = (
                    self._fsm_topic_request, None)
            if requested is not None:
                if requested not in self._fsm.states:
                    self.logger.warning(
                        "Ignoring unknown state request %r", requested)
                    self._fsm_publish_result(
                        "REJECTED", requested, "unknown state")
                elif requested == self._fsm.current:
                    self._fsm_publish_result(
                        "IGNORED", requested, "already the current state")
                else:
                    self.logger.info("State %s requested over topic", requested)
                    self.fsm_transition(requested)
            if (
                self.inference_process is not None
                and not self.inference_process.is_alive()
            ):
                self.logger.error("FSM executor process died unexpectedly")
                self.is_running = False
                self.exit_event.set()
                break
            time.sleep(0.1)

        # Hand the robot back before exiting.  The executor has stopped
        # publishing (exit_event), so only the firmware mode is switched.
        self.exit_event.set()
        if self.inference_process is not None:
            self.inference_process.join(timeout=2.0)
        current = self._fsm.current
        if current in CUSTOM_STATES:
            damping = self._safety_abort or exit_mode == "damping"
            target_mode = "damping" if damping else "walking"
            self.logger.info("Exiting from %s; switching to %s mode...",
                             current, target_mode.capitalize())
            if not self._change_robot_mode(target_mode):
                self.logger.error("Switching to %s mode failed",
                                  target_mode.capitalize())
            self._fsm.switch(ESTOP if target_mode == "damping" else IDLE)

    def __enter__(self) -> BoosterRobotPortal:
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()

    @staticmethod
    def inference_process_func(
        cfg: ControllerCfg,
        portal: BoosterRobotPortal,
        prepare_then_task: bool = False,
    ) -> None:
        controller = BoosterRobotController(cfg, portal)
        controller.run(stop_event=portal.task_start_event if prepare_then_task else None)
        if prepare_then_task and not portal.exit_event.is_set():
            BoosterRobotController(portal.cfg, portal).run()
        portal.logger.info("Inference process stopped.")


class BoosterRobotController(BaseController):
    '''Controller for Booster robots. Note that this controller runs in a
    separate process forked by BoosterRobotPortal.
    '''
    def __init__(self, cfg: ControllerCfg, portal: BoosterRobotPortal) -> None:
        super().__init__(cfg)
        self.portal = portal
        self._torque_warn_time = -1.0
        # Walking preparation starts inference before A/r, but keeps velocity
        # commands masked until that trigger is received.
        self._velocity_commands_enabled = not (
            cfg.robot.prepare_mode.strip().lower() == "walking"
            and cfg.vel_command is not None
        )

    def update_vel_command(self):
        if not self._velocity_commands_enabled:
            self.vel_command.lin_vel_x = 0.0
            self.vel_command.lin_vel_y = 0.0
            self.vel_command.ang_vel_yaw = 0.0
            return
        cmd = self.portal.synced_command.read()[0]

        self.vel_command.lin_vel_x = self.vel_command.scale_vx(cmd["vx"])
        self.vel_command.lin_vel_y = cmd["vy"] * self.vel_command.vy_max
        self.vel_command.ang_vel_yaw = cmd["vyaw"] * self.vel_command.vyaw_max

    def update_state(self) -> None:
        state = self.portal.synced_state.read()[0]

        self.robot.data.joint_pos = torch.from_numpy(
            state["joint_pos"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            state["joint_vel"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            state["feedback_torque"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            state["root_pos_w"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        rpy_t = torch.from_numpy(state["root_rpy_w"]).to(
            dtype=torch.float32).to(self.robot.data.device)
        self.robot.data.root_quat_w = lab_math.quat_from_euler_xyz(
            *rpy_t
        ).squeeze()
        self.robot.data.root_lin_vel_b = lab_math.quat_apply_inverse(
            self.robot.data.root_quat_w,
            torch.from_numpy(
                state["root_lin_vel_w"]).to(dtype=torch.float32).to(
                    self.robot.data.device)
        )
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            state["root_ang_vel_b"]).to(dtype=torch.float32).to(
                self.robot.data.device)

    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        for i in range(self.robot.num_joints):
            self.portal.motor_cmd[i].q = float(dof_targets[i].item())
            kp_val = float(self.robot.joint_stiffness[i].item())
            kd_val = float(self.robot.joint_damping[i].item())
            self.portal.motor_cmd[i].kp = kp_val
            self.portal.motor_cmd[i].kd = kd_val
        self.portal.low_cmd_publisher.publish(self.portal.low_cmd)
        self._check_torque_limits(dof_targets)

    def _check_torque_limits(self, dof_targets: torch.Tensor) -> None:
        """Warn (throttled) when the PD torque the firmware will compute
        from this command exceeds the joint's effort limit."""
        tau = (self.robot.joint_stiffness
               * (dof_targets - self.robot.data.joint_pos)
               - self.robot.joint_damping * self.robot.data.joint_vel)
        ratio = tau.abs() / self.robot.effort_limit
        if not bool((ratio > 1.0).any()):
            return
        now = time.perf_counter()
        if now - self._torque_warn_time < 1.0:
            return
        self._torque_warn_time = now
        i = int(torch.argmax(ratio))
        self.portal.logger.warning(
            "predicted torque exceeds limit on %d joint(s); worst %s: "
            "%.1f > %.1f Nm",
            int((ratio > 1.0).sum()), self.robot.cfg.joint_names[i],
            float(tau[i].abs()), float(self.robot.effort_limit[i]))

    def stop(self):
        super().stop()
        self.portal.exit_event.set()

    def run(self, stop_event=None):
        self.update_state()
        if self.vel_command is not None:
            self.update_vel_command()
        self.start()
        next_inference_time = time.perf_counter()
        while (
            self.is_running
            and not self.portal.exit_event.is_set()
            and not (stop_event is not None and stop_event.is_set())
        ):
            if (
                not self._velocity_commands_enabled
                and self.portal.velocity_commands_enabled_event.is_set()
            ):
                self._velocity_commands_enabled = True
            if time.perf_counter() < next_inference_time:
                time.sleep(0.0002)
                continue
            next_inference_time += self.cfg.policy_dt

            self.update_state()
            if self.vel_command is not None:
                self.update_vel_command()
            self.portal.metrics["policy_step"].mark()
            dof_targets = self.policy_step()
            self.ctrl_step(dof_targets)

        if stop_event is None:
            self.portal.exit_event.set()

from __future__ import annotations
import json
import logging
import signal
import time
import threading
import multiprocessing as mp
from multiprocessing import synchronize
from copy import deepcopy

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
    """Mode values used by the Booster Loco RPC API (``RobotMode``)."""

    kDamping = 0
    kPrepare = 1
    kWalking = 2
    kCustom = 3


_MODE_VALUES = {
    "damping": _RobotModeInt.kDamping,
    "prepare": _RobotModeInt.kPrepare,
    "walking": _RobotModeInt.kWalking,
    "walk": _RobotModeInt.kWalking,
    "custom": _RobotModeInt.kCustom,
}
_MODE_NAMES = {
    _RobotModeInt.kDamping: "Damping",
    _RobotModeInt.kPrepare: "Prepare",
    _RobotModeInt.kWalking: "Walking",
    _RobotModeInt.kCustom: "Custom",
}

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
        self.low_state_received_event = threading.Event()
        # Deployment state machine (see booster_deploy.fsm): the portal
        # requests states, the executor process acknowledges the active one
        # and may request follow-up states itself (policy stop/finish).
        self.fsm_requested = mp.Value("i", 0)
        self.fsm_active = mp.Value("i", 0)
        self.fsm_executor_request = mp.Value("i", -1)
        # Set by the executor once both policy controllers are built; no
        # transition is performed before that.
        self.fsm_executor_ready = mp.Value("i", 0)
        self._fsm = None
        self._fsm_walk_first = False
        # Robot mode last seen by the mode watchdog, to log changes once.
        self._seen_mode: int | None = None
        self._walk_available = False
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
        """Booster's built-in get-up (blocking, polled).  The firmware gets
        up into Walking mode; the caller requests the mode it wants after."""
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
        """ChangeMode RPC ("damping", "prepare", "walking" or "custom"),
        confirmed with GetStatus."""
        normalized = mode_name.strip().lower()
        if normalized not in _MODE_VALUES:
            raise ValueError(
                f"Unsupported robot mode {mode_name!r}; expected "
                "'damping', 'prepare', 'walking' (or 'walk'), or 'custom'"
            )

        if not self.rpc_service_client.wait_for_service(timeout_sec=15.0):
            self.logger.error("booster_rpc_service is unavailable")
            return False

        expected_mode = _MODE_VALUES[normalized]
        display_name = _MODE_NAMES[expected_mode]
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
        """Config of the WALK state: the robot's locomotion policy from
        ``tasks/locomotion`` with its own gains and stick limits, or None
        when the robot has none or the task itself is that policy."""
        robot_name = self.cfg.robot.name.lower()
        if "t2" in robot_name:
            from tasks.locomotion.robots.t2 import T2WalkTaskCfg
            walk = T2WalkTaskCfg()
        elif "t1" in robot_name:
            from tasks.locomotion.robots.t1 import T1WalkControllerCfg1
            walk = T1WalkControllerCfg1()
        elif "k1" in robot_name:
            from tasks.locomotion.robots.k1 import K1WalkTaskCfg
            walk = K1WalkTaskCfg()
        else:
            self.logger.warning("No WALK state: no locomotion policy for "
                                "robot %r", self.cfg.robot.name)
            return None
        if walk.policy.checkpoint_path == self.cfg.policy.checkpoint_path:
            return None
        walk_cfg = deepcopy(self.cfg)
        walk_cfg.robot = deepcopy(walk.robot)
        walk_cfg.policy = deepcopy(walk.policy)
        walk_cfg.vel_command = deepcopy(walk.vel_command)
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
        """Perform a validated state transition: hand the executor over and
        switch the firmware to the state's mode (``fsm.ROBOT_MODE``)."""
        from ..fsm import (
            CUSTOM_STATES, ESTOP, IDLE, ROBOT_MODE, STAND, WALK,
        )
        fsm = self._fsm
        if not fsm.can(target):
            self.logger.warning(
                "Transition %s -> %s is not allowed (allowed: %s)",
                fsm.current, target, ", ".join(fsm.targets()) or "none")
            self._fsm_publish_result(
                "REJECTED", target, f"not allowed from {fsm.current}")
            return False
        if target == WALK and not self._walk_available:
            self.logger.warning("No locomotion policy: WALK is not available")
            self._fsm_publish_result(
                "REJECTED", target, "no locomotion policy for this robot")
            return False
        if not self.fsm_executor_ready.value:
            # Never switch the firmware on behalf of an executor that cannot
            # take over (still loading, or stuck in its start-up).
            self.logger.warning(
                "FSM executor is not ready yet; %s refused", target)
            self._fsm_publish_result(
                "REJECTED", target, "executor not ready (still starting)")
            return False
        current = fsm.current
        if target == WALK:
            # A latched keyboard velocity must not make the robot walk off.
            self.remoteControlService.reset_velocity()
        reached = target
        if target in CUSTOM_STATES and current not in CUSTOM_STATES:
            # STAND -> WALK / TASK: prime a hold, enter Custom mode, then
            # the executor starts the policy.
            status = self._enter_custom_mode()
            if status == "unsafe":
                self.logger.error("Refusing %s; switching to Damping mode",
                                  target)
                self._safety_abort = True
                self._fsm_request(ESTOP, wait_ack=False)
                self._change_robot_mode("damping")
                fsm.switch(ESTOP)
                self._fsm_print_state()
                self._fsm_publish_result(
                    "FAILED", target, "unsafe posture, switched to Damping")
                return False
            if status != "ok":
                reached = None
            elif not self._fsm_request(target):
                # Custom mode holds the primed pose; give the robot back.
                self._fsm_request(current, wait_ack=False)
                reached = self._hand_back(ROBOT_MODE[current])
        elif target in CUSTOM_STATES:
            # WALK <-> TASK: a policy switch in Custom mode.
            if not self._fsm_request(target):
                reached = None
        elif target == ESTOP:
            # The executor publishes one damping command and stops
            # publishing before the firmware leaves Custom mode.
            self._fsm_request(ESTOP)
            reached = self._hand_back("damping")
        elif current in CUSTOM_STATES:
            # The executor stops publishing first; the firmware keeps the
            # last command until it has switched.
            self._fsm_request(target)
            reached = self._hand_back(ROBOT_MODE[target])
        elif current == ESTOP:
            # ESTOP -> IDLE: both are Damping mode; nothing to switch.
            self._fsm_request(IDLE)
            self._safety_abort = False
        else:
            self._fsm_request(target)
            ok = True
            if (target == STAND and current == IDLE
                    and not self._posture_is_upright()):
                # A limp robot is usually on the floor: the firmware's
                # get-up ends in Walking mode, Prepare is requested after.
                ok = self._get_up()
            if not (ok and self._change_robot_mode(ROBOT_MODE[target])):
                reached = None
        if reached != target:
            self.logger.error("Transition %s -> %s failed", current, target)
            reason = f"{current} -> {target} failed"
            if reached is not None and reached != current:
                fsm.switch(reached)
                self._fsm_print_state()
                reason += f", robot is in {reached}"
            self._fsm_publish_result("FAILED", target, reason)
            return False
        fsm.switch(target)
        self._fsm_print_state()
        self._fsm_publish_result("OK", target)
        return True

    def _hand_back(self, mode: str) -> str | None:
        """Leave Custom mode for ``mode`` ("prepare", "walking" or
        "damping").  When the firmware refuses, the next mode in the
        fallback chain is tried, damping last.  Returns the state reached,
        or None when every switch failed (the firmware then keeps the last
        command in Custom mode)."""
        from ..fsm import ESTOP, IDLE, STAND
        chains = {
            "prepare": ("prepare", "walking", "damping"),
            "walking": ("walking", "prepare", "damping"),
            "damping": ("damping",),
        }
        states = {"prepare": STAND, "walking": IDLE, "damping": ESTOP}
        for candidate in chains[mode]:
            if self._change_robot_mode(candidate):
                return states[candidate]
            self.logger.error("Switching to %s mode failed",
                              _MODE_NAMES[_MODE_VALUES[candidate]])
        return None

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
        """Apply a transition initiated by the executor (policy stop/finish):
        it has stopped publishing already; switch the firmware to the
        state's mode."""
        from ..fsm import ESTOP, ROBOT_MODE, state_name
        index = self.fsm_executor_request.value
        if index < 0:
            return
        self.fsm_executor_request.value = -1
        target = state_name(index)
        self.logger.info("Executor switched to %s", target)
        self.fsm_requested.value = index
        if target == ESTOP:
            self._safety_abort = True
        if ROBOT_MODE[target] == "custom":
            reached = target  # WALK: a policy switch, the mode is unchanged
        else:
            reached = self._hand_back(ROBOT_MODE[target])
        if reached is None:
            self.logger.error(
                "Handing the robot back failed; it keeps the last command "
                "in Custom mode")
        elif reached != target:
            self.logger.warning("Robot is in %s instead of %s", reached, target)
            target = reached
        if self._fsm.current != target:
            self._fsm.switch(target)
            self._fsm_print_state()

    def _fsm_follow_robot(self, initial: bool = False) -> None:
        """Mode watchdog: keep the state machine in step with the firmware.

        Every state has a firmware mode (``fsm.ROBOT_MODE``).  The firmware
        changes mode on its own (fall protection, a restart, the operator
        app or the Booster remote); the state matching the new mode is then
        entered without a transition check and the executor stops
        publishing if it was.  ``initial`` picks the start state.
        """
        from ..fsm import CUSTOM_STATES, ESTOP, IDLE, ROBOT_MODE, STAND
        ok, status = self._call_booster_rpc(_LOC_API_GET_STATUS)
        if not ok or status is None:
            return
        mode = int(status.get("current_mode", -1))
        current = self._fsm.current
        if mode == _MODE_VALUES[ROBOT_MODE[current]]:
            self._seen_mode = mode
            return
        if mode == _RobotModeInt.kPrepare:
            target = STAND
        elif mode == _RobotModeInt.kWalking:
            target = IDLE  # Booster's own controller
        elif mode == _RobotModeInt.kDamping:
            target = IDLE if initial else ESTOP
        else:
            # Custom mode under another controller (or ours, after a
            # crash), Soccer, unknown: hands off.
            target = IDLE
        if mode != self._seen_mode:
            self._seen_mode = mode
            name = _MODE_NAMES.get(mode, f"mode {mode}")
            if mode == _RobotModeInt.kCustom and current not in CUSTOM_STATES:
                self.logger.warning(
                    "Robot is in Custom mode under another controller; "
                    "%s publishes nothing (X switches it to Prepare)",
                    target)
            elif initial or target == current:
                self.logger.info("Robot is in %s mode: %s", name, target)
            else:
                self.logger.error(
                    "Robot switched to %s mode on its own while in %s; "
                    "following to %s", name, current, target)
        if target == current:
            return
        if target != ESTOP:
            self._safety_abort = False
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
            CUSTOM_STATES, IDLE, STATES, TRANSITIONS, StateMachine,
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
        self._walk_available = walk_cfg is not None
        self._fsm = StateMachine(STATES, TRANSITIONS, IDLE)
        self._fsm_walk_first = (
            prepare_mode == "walking" and self._walk_available)

        self.inference_process = mp.Process(
            target=fsm_process_func,
            args=(self, self.cfg, walk_cfg),
            daemon=True,
        )
        self.inference_process.start()
        self.logger.info("FSM executor process started")
        print(self.remoteControlService.get_fsm_operation_hint())
        # Start in the state matching the robot's current mode.
        self._fsm_follow_robot(initial=True)
        self._fsm_print_state()

        next_mode_check = time.perf_counter()
        executor_started = time.perf_counter()
        next_ready_warn = executor_started + 5.0
        while self.is_running and not self.exit_event.is_set():
            if (not self.fsm_executor_ready.value
                    and time.perf_counter() >= next_ready_warn):
                pid = self.inference_process.pid
                self.logger.warning(
                    "FSM executor (pid %s) not ready after %.0fs: still "
                    "loading the policies, or stuck starting up "
                    "(py-spy dump --pid %s shows where); transitions are "
                    "refused until it reports ready",
                    pid, time.perf_counter() - executor_started, pid)
                next_ready_warn += 10.0
            self._fsm_sync_from_executor()
            if time.perf_counter() >= next_mode_check:
                next_mode_check += self.cfg.booster.mode_check_period_s
                self._fsm_follow_robot()
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
        # publishing (exit_event), so only the firmware mode is switched;
        # in the other states the firmware controls the robot already.
        self.exit_event.set()
        if self.inference_process is not None:
            self.inference_process.join(timeout=2.0)
        current = self._fsm.current
        if current in CUSTOM_STATES:
            damping = self._safety_abort or exit_mode == "damping"
            target_mode = "damping" if damping else "walking"
            self.logger.info("Exiting from %s; switching to %s mode...",
                             current, target_mode.capitalize())
            reached = self._hand_back(target_mode)
            if reached is None:
                self.logger.error(
                    "Handing the robot back failed; it keeps the last "
                    "command in Custom mode")
            else:
                self._fsm.switch(reached)

    def __enter__(self) -> BoosterRobotPortal:
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


class BoosterRobotController(BaseController):
    '''Controller for Booster robots. Note that this controller runs in a
    separate process forked by BoosterRobotPortal.
    '''
    def __init__(self, cfg: ControllerCfg, portal: BoosterRobotPortal) -> None:
        super().__init__(cfg)
        self.portal = portal
        self._torque_warn_time = -1.0
        self._velocity_commands_enabled = True

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

    def run(self) -> None:
        """Run the policy until it stops (the FSM executor ticks it
        instead; see ``booster_deploy.fsm.executor``)."""
        self.update_state()
        if self.vel_command is not None:
            self.update_vel_command()
        self.start()
        next_inference_time = time.perf_counter()
        while self.is_running and not self.portal.exit_event.is_set():
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

        self.portal.exit_event.set()

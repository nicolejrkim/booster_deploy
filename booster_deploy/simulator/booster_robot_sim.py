"""MuJoCo stand-in for a Booster robot's ROS 2 low-level interface.

The node behaves like the robot's firmware as seen from ``deploy.py``:

- publishes ``/low_state`` (IMU roll/pitch/yaw and gyro, serial motor states),
  ``/odometer_state`` (ground-truth trunk x, y and heading) and
  ``booster_sim/contact_forces`` (ground-truth normal force under each foot);
- consumes ``/joint_ctrl`` (``LowCmd`` with per-motor PD targets);
- serves ``booster_rpc_service`` for the Loco RPC ``ChangeMode`` and
  ``GetStatus`` calls used to enter and leave Custom mode.

In Damping mode the motors only damp, in Prepare/Walking mode a built-in PD
stand controller holds the robot's prepare pose (the firmware's locomotion
controller is not emulated), and in Custom mode the latest ``/joint_ctrl``
command is applied.  As on the robot, commands received before entering
Custom mode are retained and applied at the switch.

The IMU is taken as the floating base frame (the K1/T1/T2 MJCF ``imu`` site
sits at the trunk origin with identity orientation).

An optional *elastic band* (as in crl-humanoid-ros) is a slack rope on the
trunk: it applies no force while the trunk is at or above its anchor height
(the spawn height by default), and catches the robot with a spring-damper
when it drops below, so policies can be tried without falls while standing
and walking stay unaffected.  Toggle it with the ``elastic_band``
``std_srvs/SetBool`` service or the ``E`` key in the viewer.
"""
from __future__ import annotations

import json
import logging
import signal
import threading
import time
from typing import Optional

import mujoco
import numpy as np
import rclpy
from booster_assets import BOOSTER_ASSETS_DIR
from booster_interface.msg import LowCmd, LowState, MotorState, Odometer
from booster_interface.srv import RpcService
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from std_msgs.msg import Bool, Float32MultiArray
from std_srvs.srv import SetBool

from ..controllers.controller_cfg import RobotCfg

logger = logging.getLogger("booster_deploy.simulator")


class RobotMode:
    """Values of the Loco RPC ``RobotMode`` enum."""

    kDamping = 0
    kPrepare = 1
    kWalking = 2
    kCustom = 3

    NAMES = {0: "damping", 1: "prepare", 2: "walking", 3: "custom"}

    @classmethod
    def from_name(cls, name: str) -> int:
        lookup = {v: k for k, v in cls.NAMES.items()}
        lookup["walk"] = cls.kWalking
        key = name.strip().lower()
        if key not in lookup:
            raise ValueError(
                f"unknown robot mode {name!r}; expected one of "
                f"{sorted(lookup)}")
        return lookup[key]


_LOC_API_CHANGE_MODE = 2000
_LOC_API_GET_UP = 2008
_LOC_API_GET_STATUS = 2018
_LOC_API_GET_UP_WITH_MODE = 2025


def euler_xyz_from_quat(quat: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw of a (w, x, y, z) quaternion.

    Same convention as ``isaaclab.math.euler_xyz_from_quat`` and therefore
    the inverse of the ``quat_from_euler_xyz`` used by the deployment
    controller to rebuild the base orientation from ``imu_state.rpy``.
    """
    w, x, y, z = quat
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def lowest_collision_point(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """World z of the lowest point of the robot's collision geometry."""
    zmin = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue  # world geoms (ground plane)
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        pos = data.geom_xpos[g]
        rot = data.geom_xmat[g].reshape(3, 3)
        gtype = model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[adr:adr + num]
            zmin = min(zmin, float((verts @ rot.T + pos)[:, 2].min()))
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            half = model.geom_size[g]
            corners = np.array([
                [sx * half[0], sy * half[1], sz * half[2]]
                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
            ])
            zmin = min(zmin, float((corners @ rot.T + pos)[:, 2].min()))
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            zmin = min(zmin, float(pos[2] - model.geom_size[g][0]))
        else:  # capsule / cylinder / ellipsoid: conservative bound
            zmin = min(zmin, float(pos[2] - model.geom_size[g][:2].sum()))
    return zmin


class BoosterRobotSim(Node):
    """ROS 2 node that simulates a Booster robot's low-level interface."""

    def __init__(
        self,
        robot_cfg: RobotCfg,
        *,
        physics_dt: Optional[float] = None,
        state_rate_hz: float = 500.0,
        real_time_factor: float = 1.0,
        initial_mode: str = "walking",
        init_pos: Optional[list[float]] = None,
        init_joint_pos: Optional[list[float]] = None,
        mode_transition_s: float = 1.0,
        log_states: Optional[str] = None,
        elastic_band: bool = False,
        band_height: Optional[float] = None,
        band_stiffness: float = 2000.0,
        band_damping: float = 100.0,
        node_name: str = "booster_robot_sim",
    ) -> None:
        super().__init__(node_name)
        self.robot_cfg = robot_cfg
        self.num_joints = len(robot_cfg.joint_names)

        mjcf_path = robot_cfg.mjcf_path.replace(
            "{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        if physics_dt is not None:
            self.model.opt.timestep = physics_dt
        self.physics_dt = float(self.model.opt.timestep)

        self._check_model_layout()

        self.state_rate_hz = float(state_rate_hz)
        self.real_time_factor = float(real_time_factor)
        self.mode_transition_s = float(mode_transition_s)
        self.force_limit = self.model.actuator_forcerange[:, 1].copy()

        prepare = robot_cfg.prepare_state
        self.prepare_kp = np.asarray(prepare.stiffness, dtype=np.float64)
        self.prepare_kd = np.asarray(prepare.damping, dtype=np.float64)
        self.hold_pose = np.asarray(
            prepare.joint_pos if init_joint_pos is None else init_joint_pos,
            dtype=np.float64,
        )

        # --- shared state (guarded by _lock or written atomically) ---
        self._lock = threading.Lock()
        self._stop = threading.Event()  # stop requested (signal / viewer)
        self._stopped = False           # stop() already ran
        self._cmd: Optional[np.ndarray] = None  # (n, 5): q dq tau kp kd
        self._cmd_count = 0
        self._cmd_warned = False
        self.mode = RobotMode.from_name(initial_mode)
        self._ramp_start_pose = self.hold_pose.copy()
        self._ramp_start_time = 0.0
        self._custom_entry_pose = self.hold_pose.copy()

        self._reset_pose(init_pos)

        # --- elastic band (virtual harness on the trunk) ---
        self.band_enabled = bool(elastic_band)
        self.band_stiffness = float(band_stiffness)
        self.band_damping = float(band_damping)
        self._band_body = 1  # trunk: first body after the world
        # Anchor height: the rope is slack above it.  Default: the spawn
        # height, so a standing or walking robot never feels the band.
        self.band_height = (float(self.data.qpos[2]) if band_height is None
                            else float(band_height))

        # --- ROS 2 interface (names match the robot firmware) ---
        self._low_state_pub = self.create_publisher(
            LowState, "/low_state",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST),
        )
        self._odom_pub = self.create_publisher(
            Odometer, "/odometer_state",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST),
        )
        self._odom = Odometer()
        # Ground-truth foot contact forces (normal component, N), one entry
        # per foot body in ``self.foot_bodies`` order.
        self.foot_bodies = [
            i for i in range(1, self.model.nbody)
            if "ankle_roll" in self.model.body(i).name
            or "foot" in self.model.body(i).name.lower()]
        self.foot_names = [self.model.body(i).name for i in self.foot_bodies]
        self._contact_pub = self.create_publisher(
            Float32MultiArray, "booster_sim/contact_forces",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST),
        )
        self._contact_msg = Float32MultiArray()
        self._contact_buf = np.zeros(6)
        # Throttled torque-limit warning (as in crl-humanoid-ros' simulator).
        self._torque_warn_time = -1.0
        self._torque_warn_period = 1.0
        self.create_subscription(
            LowCmd, "/joint_ctrl", self._on_joint_ctrl,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST),
        )
        self.create_service(RpcService, "booster_rpc_service", self._on_rpc)
        self.create_service(SetBool, "elastic_band", self._on_elastic_band)
        self._band_pub = self.create_publisher(
            Bool, "booster_sim/elastic_band",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       history=HistoryPolicy.KEEP_LAST))
        self._band_pub.publish(Bool(data=self.band_enabled))

        self._low_state = LowState()
        self._low_state.motor_state_serial = [
            MotorState() for _ in range(self.num_joints)]
        for motor in self._low_state.motor_state_serial:
            motor.temperature = 30
            motor.reserve = np.array([0, int(self.state_rate_hz)],
                                     dtype=np.uint32)
        self._prev_base_vel_w = np.zeros(3)

        # --- logging ---
        self._log_path = log_states
        self._log: dict[str, list] = {
            "time": [], "qpos": [], "qvel": [], "ctrl": [], "mode": []}
        self._wall_start = time.perf_counter()
        self._lag_steps = 0

        self._physics_thread = threading.Thread(
            target=self._physics_loop, name="physics", daemon=True)

        self.get_logger().info(
            f"{robot_cfg.name}: {self.num_joints} joints, physics dt "
            f"{self.physics_dt * 1e3:.1f} ms, /low_state at "
            f"{self.state_rate_hz:.0f} Hz, initial mode "
            f"'{RobotMode.NAMES[self.mode]}'")

    # ------------------------------------------------------------------ setup
    def _check_model_layout(self) -> None:
        joint_names = [self.model.joint(i).name
                       for i in range(1, self.model.njnt)]
        if joint_names != list(self.robot_cfg.joint_names):
            raise RuntimeError(
                "MJCF joint order does not match RobotCfg.joint_names:\n"
                f"  mjcf: {joint_names}\n"
                f"  cfg:  {list(self.robot_cfg.joint_names)}")
        if self.model.nu != self.num_joints:
            raise RuntimeError(
                f"MJCF has {self.model.nu} actuators, "
                f"expected {self.num_joints}")
        if self.model.nq != 7 + self.num_joints:
            raise RuntimeError("MJCF must have a single free joint followed "
                               "by the robot's hinge joints")

    def _reset_pose(self, init_pos: Optional[list[float]]) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[3] = 1.0
        self.data.qpos[7:] = self.hold_pose
        if init_pos is not None:
            self.data.qpos[:3] = init_pos
        else:
            self.data.qpos[2] = 1.0
            mujoco.mj_forward(self.model, self.data)
            zmin = lowest_collision_point(self.model, self.data)
            self.data.qpos[2] += -zmin + 0.003
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info(
            f"spawned at z={self.data.qpos[2]:.3f} m in the prepare pose")

    # -------------------------------------------------------------- ROS 2 I/O
    def _on_joint_ctrl(self, msg: LowCmd) -> None:
        if msg.cmd_type != LowCmd.CMD_TYPE_SERIAL:
            if not self._cmd_warned:
                self._cmd_warned = True
                self.get_logger().warning(
                    f"ignoring LowCmd with cmd_type={msg.cmd_type}; only "
                    "CMD_TYPE_SERIAL is simulated")
            return
        if len(msg.motor_cmd) != self.num_joints:
            if not self._cmd_warned:
                self._cmd_warned = True
                self.get_logger().warning(
                    f"ignoring LowCmd with {len(msg.motor_cmd)} motors, "
                    f"expected {self.num_joints}")
            return
        cmd = np.array(
            [[m.q, m.dq, m.tau, m.kp, m.kd] for m in msg.motor_cmd],
            dtype=np.float64,
        )
        self._cmd = cmd  # reference swap is atomic
        self._cmd_count += 1

    def _on_rpc(self, request, response):
        api_id = int(request.msg.api_id)
        try:
            body = json.loads(request.msg.body) if request.msg.body else {}
        except json.JSONDecodeError:
            body = {}
        response.msg.status = 0
        response.msg.body = ""
        if api_id == _LOC_API_CHANGE_MODE:
            mode = int(body.get("mode", -1))
            if mode in RobotMode.NAMES:
                self.set_mode(mode)
            else:
                self.get_logger().error(f"ChangeMode: unknown mode {mode}")
                response.msg.status = -1
        elif api_id in (_LOC_API_GET_UP, _LOC_API_GET_UP_WITH_MODE):
            # The firmware's get-up motion is not simulated: the robot is
            # put back upright in the prepare pose and handed to the stand
            # controller of the requested mode (Walking by default).
            mode = int(body.get("mode", RobotMode.kWalking))
            self._teleport_upright()
            if mode not in RobotMode.NAMES:
                mode = RobotMode.kWalking
            self.set_mode(mode)
            self.get_logger().info("get-up: teleported upright (not simulated)")
        elif api_id == _LOC_API_GET_STATUS:
            response.msg.body = json.dumps({
                "current_mode": self.mode,
                "current_body_control": 0,
                "current_actions": [],
            })
        else:
            self.get_logger().warning(
                f"RPC api_id={api_id} is not simulated; replying status 0")
        return response

    def _on_elastic_band(self, request, response):
        self.set_elastic_band(bool(request.data))
        response.success = True
        response.message = (
            f"elastic band {'enabled' if self.band_enabled else 'disabled'}")
        return response

    def set_elastic_band(self, enabled: bool) -> None:
        with self._lock:
            self.band_enabled = enabled
            if not enabled:
                self.data.xfrc_applied[self._band_body, :] = 0.0
        self._band_pub.publish(Bool(data=enabled))
        self.get_logger().info(
            f"elastic band {'ON' if enabled else 'OFF'} "
            f"(slack above z={self.band_height:.2f} m, "
            f"k={self.band_stiffness:.0f} N/m, c={self.band_damping:.0f} Ns/m)")

    def _apply_elastic_band(self) -> None:
        """Slack rope: upward spring-damper only below the anchor height."""
        pos = self.data.xpos[self._band_body]
        vel = self.data.cvel[self._band_body, 3:6]  # linear part, world
        drop = self.band_height - pos[2]
        force = 0.0
        if drop > 0.0:
            force = self.band_stiffness * drop - self.band_damping * vel[2]
            force = max(force, 0.0)  # a rope cannot push down
        self.data.xfrc_applied[self._band_body, :] = 0.0
        self.data.xfrc_applied[self._band_body, 2] = force

    def _teleport_upright(self) -> None:
        with self._lock:
            xy = self.data.qpos[:2].copy()
            self._reset_pose(None)
            self.data.qpos[:2] = xy
            self._ramp_start_pose = self.hold_pose.copy()
            self._ramp_start_time = self.data.time
            mujoco.mj_forward(self.model, self.data)

    def set_mode(self, mode: int) -> None:
        with self._lock:
            if mode == self.mode:
                return
            old = self.mode
            q = self.data.qpos[7:].copy()
            if mode in (RobotMode.kPrepare, RobotMode.kWalking):
                self._ramp_start_pose = q
                self._ramp_start_time = self.data.time
            elif mode == RobotMode.kCustom:
                self._custom_entry_pose = q
            self.mode = mode
        self.get_logger().info(
            f"mode {RobotMode.NAMES[old]} -> {RobotMode.NAMES[mode]}"
            + (f" (retained joint_ctrl #{self._cmd_count})"
               if mode == RobotMode.kCustom and self._cmd is not None else ""))

    # ---------------------------------------------------------------- physics
    def _control_torque(self) -> np.ndarray:
        q = self.data.qpos[7:]
        dq = self.data.qvel[6:]
        mode = self.mode
        if mode == RobotMode.kDamping:
            return -self.prepare_kd * dq
        if mode in (RobotMode.kPrepare, RobotMode.kWalking):
            if self.mode_transition_s > 0.0:
                alpha = min(1.0, (self.data.time - self._ramp_start_time)
                            / self.mode_transition_s)
            else:
                alpha = 1.0
            target = self._ramp_start_pose + alpha * (
                self.hold_pose - self._ramp_start_pose)
            return self.prepare_kp * (target - q) - self.prepare_kd * dq
        cmd = self._cmd
        if cmd is None:
            return (self.prepare_kp * (self._custom_entry_pose - q)
                    - self.prepare_kd * dq)
        return (cmd[:, 3] * (cmd[:, 0] - q) + cmd[:, 4] * (cmd[:, 1] - dq)
                + cmd[:, 2])

    def _foot_contact_forces(self) -> list[float]:
        """Sum of normal contact forces on each foot body, in Newtons."""
        forces = [0.0] * len(self.foot_bodies)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            for k, body in enumerate(self.foot_bodies):
                if body in (b1, b2):
                    mujoco.mj_contactForce(self.model, self.data, i,
                                           self._contact_buf)
                    forces[k] += abs(float(self._contact_buf[0]))
        return forces

    def _check_torque_limits(self, tau: np.ndarray) -> None:
        """Warn (throttled) when a PD torque exceeds the joint limit."""
        over = np.abs(tau) > self.force_limit
        if not over.any():
            return
        now = self.data.time
        if now - self._torque_warn_time < self._torque_warn_period:
            return
        self._torque_warn_time = now
        i = int(np.argmax(np.abs(tau) / self.force_limit))
        self.get_logger().warning(
            f"torque command exceeds limit on {int(over.sum())} joint(s); "
            f"worst {self.robot_cfg.joint_names[i]}: "
            f"{abs(tau[i]):.1f} > {self.force_limit[i]:.1f} Nm (clamped)")

    def _publish_low_state(self) -> None:
        msg = self._low_state
        quat = self.data.qpos[3:7]
        msg.imu_state.rpy = euler_xyz_from_quat(quat)
        msg.imu_state.gyro = self.data.qvel[3:6].astype(np.float32)
        # proper acceleration in the base frame (gravity included)
        base_vel_w = self.data.qvel[:3].copy()
        acc_w = (base_vel_w - self._prev_base_vel_w) / self.physics_dt
        acc_w[2] += 9.81
        self._prev_base_vel_w = base_vel_w
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        msg.imu_state.acc = (rot.reshape(3, 3).T @ acc_w).astype(np.float32)

        q = self.data.qpos[7:]
        dq = self.data.qvel[6:]
        ddq = self.data.qacc[6:]
        tau = self.data.actuator_force
        for i, motor in enumerate(msg.motor_state_serial):
            motor.q = float(q[i])
            motor.dq = float(dq[i])
            motor.ddq = float(ddq[i])
            motor.tau_est = float(tau[i])
        self._low_state_pub.publish(msg)
        self._odom.x = float(self.data.qpos[0])
        self._odom.y = float(self.data.qpos[1])
        self._odom.theta = float(msg.imu_state.rpy[2])
        self._odom_pub.publish(self._odom)
        self._contact_msg.data = self._foot_contact_forces()
        self._contact_pub.publish(self._contact_msg)

    def _physics_loop(self) -> None:
        dt = self.physics_dt
        steps_per_state = max(1, int(round(1.0 / (self.state_rate_hz * dt))))
        wall_dt = dt / self.real_time_factor
        next_t = time.perf_counter()
        step = 0
        last_report = time.perf_counter()
        report_steps = 0
        while not self._stop.is_set():
            with self._lock:
                tau = self._control_torque()
                self._check_torque_limits(tau)
                self.data.ctrl[:] = np.clip(
                    tau, -self.force_limit, self.force_limit)
                if self.band_enabled:
                    self._apply_elastic_band()
                mujoco.mj_step(self.model, self.data)
                step += 1
                report_steps += 1
                if step % steps_per_state == 0 and not self._stop.is_set():
                    self._publish_low_state()
                    if self._log_path is not None:
                        self._log["time"].append(self.data.time)
                        self._log["qpos"].append(self.data.qpos.copy())
                        self._log["qvel"].append(self.data.qvel.copy())
                        self._log["ctrl"].append(self.data.ctrl.copy())
                        self._log["mode"].append(self.mode)
            next_t += wall_dt
            now = time.perf_counter()
            remaining = next_t - now
            if remaining > 3e-4:
                time.sleep(remaining - 2e-4)
            elif remaining < -0.05:
                self._lag_steps += 1
                next_t = now  # far behind: drop the backlog instead of racing
            while time.perf_counter() < next_t:
                pass
            if now - last_report >= 2.0:
                rtf = report_steps * dt / (now - last_report)
                self.get_logger().info(
                    f"t={self.data.time:7.2f}s "
                    f"mode={RobotMode.NAMES[self.mode]:8s} "
                    f"z={self.data.qpos[2]:.3f} rtf={rtf:.2f} "
                    f"joint_ctrl#={self._cmd_count}"
                    + (" band=ON" if self.band_enabled else "")
                    + (f" lag_resets={self._lag_steps}"
                       if self._lag_steps else ""))
                last_report = now
                report_steps = 0

    # -------------------------------------------------------------- lifecycle
    def run(self, viewer: bool = False) -> None:
        """Run until Ctrl+C / SIGTERM (or the viewer window is closed).

        Signals only raise the stop flag; the physics thread, the executor
        and the viewer are then shut down in order, so no thread ever touches
        a torn-down ROS context.  ``rclpy.init`` should be called with
        ``SignalHandlerOptions.NO`` (see ``scripts/sim_robot.py``).
        """
        self._physics_thread.start()
        executor = SingleThreadedExecutor()
        executor.add_node(self)

        def on_signal(signum, frame):  # noqa: ARG001
            self._stop.set()

        previous = {s: signal.signal(s, on_signal)
                    for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            if viewer:
                spin_thread = threading.Thread(
                    target=self._spin, args=(executor,), name="rclpy",
                    daemon=True)
                spin_thread.start()
                self._run_viewer()
                spin_thread.join(timeout=2.0)
            else:
                self._spin(executor)
        finally:
            self.stop()
            executor.shutdown(timeout_sec=1.0)
            for s, handler in previous.items():
                signal.signal(s, handler)

    def _spin(self, executor: SingleThreadedExecutor) -> None:
        while not self._stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    def _run_viewer(self) -> None:
        import mujoco.viewer

        def on_key(keycode: int) -> None:
            if keycode == ord("E"):
                self.set_elastic_band(not self.band_enabled)

        with mujoco.viewer.launch_passive(
            self.model, self.data,
            show_left_ui=False, show_right_ui=False,
            key_callback=on_key,
        ) as v:
            v.cam.elevation = -20
            v.cam.distance = 2.0
            self.get_logger().info("viewer: press E to toggle the elastic band")
            while v.is_running() and not self._stop.is_set():
                with self._lock:
                    v.cam.lookat[:] = self.data.qpos[:3]
                    v.sync()
                time.sleep(1.0 / 60.0)
            # stop stepping before the viewer's render thread is torn down
            self._stop.set()
            if self._physics_thread.is_alive():
                self._physics_thread.join(timeout=1.0)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._physics_thread.is_alive():
            self._physics_thread.join(timeout=1.0)
        if self._log_path is not None and self._log["time"]:
            path = self._log_path
            if not path.endswith(".npz"):
                path += ".npz"
            np.savez(path, **{k: np.asarray(v) for k, v in self._log.items()})
            # rclpy's context may already be shut down here (SIGINT).
            print(f"[booster_robot_sim] saved {len(self._log['time'])} "
                  f"state samples to {path}", flush=True)

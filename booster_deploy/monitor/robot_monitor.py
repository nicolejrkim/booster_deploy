"""Live monitor: pose a MuJoCo model from the robot's published state.

Subscribes to the topics a deployment uses (``/low_state``, ``/joint_ctrl``,
``booster_deploy/fsm_state``) and shows, in a MuJoCo viewer:

- the robot posed from the joint encoders and the IMU orientation, with the
  feet kept on the floor; its position and heading come from
  ``/odometer_state`` when that is published (robot firmware, simulator),
  otherwise it stays at the origin;
- a translucent ghost at the commanded joint targets from ``/joint_ctrl``;
- a floating label with the deployment's FSM state;
- a state-machine panel (key ``M``) listing the states, the current one and
  the transitions allowed from it; ``Up``/``Down`` select and ``Enter``
  requests the transition on ``booster_deploy/fsm_request``;
- the simulator's elastic band state (key ``B`` toggles it);
- the simulator's ground-truth foot contact forces, and warnings when a
  joint runs at its torque limit or outside its angle range.

A status line (topic rates, largest joint tracking error, largest torque
relative to the effort limit, trunk tilt) is printed to the terminal, which
also works without a display (``viewer=False``, e.g. over SSH).  The
received stream can be recorded to an ``.npz`` file for offline analysis.

Works against the real robot (same ROS 2 domain) and against
``scripts/sim_robot.py``.
"""
from __future__ import annotations

import signal
import threading
import time
from collections import deque
from typing import Optional

import mujoco
import numpy as np
import rclpy
from booster_assets import BOOSTER_ASSETS_DIR
from booster_interface.msg import LowCmd, LowState, Odometer
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from std_msgs.msg import Bool, Float32MultiArray, String
from std_srvs.srv import SetBool

from ..controllers.controller_cfg import RobotCfg
from ..fsm import STATES, TRANSITIONS
from ..simulator.booster_robot_sim import lowest_collision_point

# GLFW key codes used by the viewer's key callback.
_KEY_ENTER, _KEY_UP, _KEY_DOWN = 257, 265, 264


def quat_from_euler_xyz(rpy: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of roll/pitch/yaw, inverse of the IMU convention."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
    ])


class _RateMeter:
    def __init__(self, window: int = 200) -> None:
        self.stamps: deque[float] = deque(maxlen=window)

    def mark(self) -> None:
        self.stamps.append(time.perf_counter())

    def hz(self) -> float:
        if len(self.stamps) < 2:
            return 0.0
        span = self.stamps[-1] - self.stamps[0]
        if time.perf_counter() - self.stamps[-1] > 1.0:
            return 0.0  # stale
        return (len(self.stamps) - 1) / span if span > 0 else 0.0


class RobotMonitor(Node):
    def __init__(
        self,
        robot_cfg: RobotCfg,
        *,
        log_path: Optional[str] = None,
        ghost_rgba=(0.2, 0.8, 0.2, 0.3),
        node_name: str = "booster_robot_monitor",
    ) -> None:
        super().__init__(node_name)
        self.robot_cfg = robot_cfg
        self.num_joints = len(robot_cfg.joint_names)
        mjcf_path = robot_cfg.mjcf_path.replace(
            "{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.ghost_data = mujoco.MjData(self.model)
        self.effort_limit = np.asarray(robot_cfg.effort_limit, dtype=np.float64)
        self.ghost_rgba = np.asarray(ghost_rgba, dtype=np.float32)
        self._ghost_option = mujoco.MjvOption()

        joint_names = [self.model.joint(i).name
                       for i in range(1, self.model.njnt)]
        if joint_names != list(robot_cfg.joint_names):
            raise RuntimeError("MJCF joint order does not match "
                               "RobotCfg.joint_names")

        # latest data (guarded by _lock)
        self._lock = threading.Lock()
        self.q = np.zeros(self.num_joints)
        self.dq = np.zeros(self.num_joints)
        self.tau = np.zeros(self.num_joints)
        self.rpy = np.zeros(3)
        self.gyro = np.zeros(3)
        self.q_cmd: Optional[np.ndarray] = None
        self.kp_cmd = np.zeros(self.num_joints)
        self.kd_cmd = np.zeros(self.num_joints)
        self.fsm_state = "?"
        self.band_state: Optional[bool] = None  # None: no simulator band
        self.odom: Optional[np.ndarray] = None  # x, y, theta
        self._odom_rate = _RateMeter()
        self.contact: Optional[np.ndarray] = None  # N per foot (simulator)
        self._contact_rate = _RateMeter()
        self.joint_range = self.model.jnt_range[1:].copy()  # hinge joints
        self._range_limited = self.model.jnt_limited[1:].astype(bool)
        self._state_rate = _RateMeter()
        self._cmd_rate = _RateMeter()
        self._stop = threading.Event()
        self._stopped = False
        # viewer UI state (written by the GUI thread's key callback)
        self._show_panel = True
        self._show_info = True
        self._follow = True
        self._selected = 0
        self._pending_request: Optional[str] = None
        self._pending_band: Optional[bool] = None

        self._log_path = log_path
        self._log: dict[str, list] = {
            k: [] for k in ("time", "q", "dq", "tau", "rpy", "gyro",
                            "q_cmd", "kp", "kd")}
        self._t0 = time.perf_counter()

        best_effort = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LowState, "/low_state",
                                 self._on_low_state, best_effort)
        self.create_subscription(Odometer, "/odometer_state",
                                 self._on_odometer, best_effort)
        self.create_subscription(Float32MultiArray,
                                 "booster_sim/contact_forces",
                                 self._on_contact, best_effort)
        self.create_subscription(
            LowCmd, "/joint_ctrl", self._on_joint_ctrl,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST))
        latched = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(
            String, "booster_deploy/fsm_state", self._on_fsm_state, latched)
        self.create_subscription(
            Bool, "booster_sim/elastic_band", self._on_band_state, latched)
        self._request_pub = self.create_publisher(
            String, "booster_deploy/fsm_request",
            QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST))
        self._band_client = self.create_client(SetBool, "elastic_band")
        self.get_logger().info(
            f"monitoring {robot_cfg.name}: /low_state, /joint_ctrl, "
            "booster_deploy/fsm_state")

    # ------------------------------------------------------------ callbacks
    def _on_low_state(self, msg: LowState) -> None:
        n = len(msg.motor_state_serial)
        if n != self.num_joints:
            return
        with self._lock:
            for i, m in enumerate(msg.motor_state_serial):
                self.q[i] = m.q
                self.dq[i] = m.dq
                self.tau[i] = m.tau_est
            self.rpy[:] = msg.imu_state.rpy
            self.gyro[:] = msg.imu_state.gyro
            self._state_rate.mark()
            if self._log_path is not None:
                self._log["time"].append(time.perf_counter() - self._t0)
                self._log["q"].append(self.q.copy())
                self._log["dq"].append(self.dq.copy())
                self._log["tau"].append(self.tau.copy())
                self._log["rpy"].append(self.rpy.copy())
                self._log["gyro"].append(self.gyro.copy())
                q_cmd = self.q_cmd if self.q_cmd is not None else np.full(
                    self.num_joints, np.nan)
                self._log["q_cmd"].append(q_cmd.copy())
                self._log["kp"].append(self.kp_cmd.copy())
                self._log["kd"].append(self.kd_cmd.copy())

    def _on_joint_ctrl(self, msg: LowCmd) -> None:
        if len(msg.motor_cmd) != self.num_joints:
            return
        with self._lock:
            if self.q_cmd is None:
                self.q_cmd = np.zeros(self.num_joints)
            for i, m in enumerate(msg.motor_cmd):
                self.q_cmd[i] = m.q
                self.kp_cmd[i] = m.kp
                self.kd_cmd[i] = m.kd
            self._cmd_rate.mark()

    def _on_odometer(self, msg: Odometer) -> None:
        with self._lock:
            self.odom = np.array([msg.x, msg.y, msg.theta])
            self._odom_rate.mark()

    def _on_contact(self, msg: Float32MultiArray) -> None:
        with self._lock:
            self.contact = np.asarray(msg.data, dtype=np.float64)
            self._contact_rate.mark()

    def _on_fsm_state(self, msg: String) -> None:
        with self._lock:
            self.fsm_state = msg.data

    def _on_band_state(self, msg: Bool) -> None:
        with self._lock:
            self.band_state = bool(msg.data)

    # ------------------------------------------------------------ requests
    def request_state(self, target: str) -> None:
        """Ask the deployment for a transition (validated on its side)."""
        self._request_pub.publish(String(data=target))
        self.get_logger().info(f"requested state {target}")

    def toggle_band(self) -> None:
        if not self._band_client.service_is_ready():
            self.get_logger().warning(
                "no elastic_band service (not running against the simulator)")
            return
        with self._lock:
            enable = not bool(self.band_state)
        self._band_client.call_async(SetBool.Request(data=enable))
        self.get_logger().info(
            f"elastic band {'on' if enable else 'off'} requested")

    def _on_key(self, keycode: int) -> None:
        """Viewer key callback (GUI thread): only flips flags."""
        if keycode == ord("M"):
            self._show_panel = not self._show_panel
        elif keycode == ord("N"):
            self._show_info = not self._show_info
        elif keycode == ord("V"):
            self._follow = not self._follow
        elif keycode == ord("B"):
            self._pending_band = True
        elif keycode == _KEY_UP and self._show_panel:
            self._selected = (self._selected - 1) % len(STATES)
        elif keycode == _KEY_DOWN and self._show_panel:
            self._selected = (self._selected + 1) % len(STATES)
        elif keycode == _KEY_ENTER and self._show_panel:
            self._pending_request = STATES[self._selected]

    def _service_pending(self) -> None:
        request, self._pending_request = self._pending_request, None
        if request is not None:
            self.request_state(request)
        if self._pending_band:
            self._pending_band = None
            self.toggle_band()

    def _overlay_texts(self, state: str) -> list:
        texts = []
        if self._show_info:
            with self._lock:
                band = self.band_state
            band_text = ("n/a" if band is None else
                         ("ON" if band else "OFF"))
            with self._lock:
                odom = None if self.odom is None else self.odom.copy()
                odom_live = self._odom_rate.hz() > 0
            odom_text = (f"{odom[0]:+.2f} {odom[1]:+.2f} m "
                         f"{np.degrees(odom[2]):+.0f} deg"
                         if odom is not None and odom_live else "none")
            with self._lock:
                contact = None if self.contact is None else self.contact.copy()
                contact_live = self._contact_rate.hz() > 0
                tau = self.tau.copy()
            contact_text = ("/".join(f"{c:.0f}" for c in contact) + " N"
                            if contact is not None and contact_live
                            else "n/a")
            tau_frac = np.abs(tau) / self.effort_limit
            i_tau = int(np.argmax(tau_frac))
            worst = self.robot_cfg.joint_names[i_tau].replace("_joint", "")
            tau_text = (f"{tau_frac[i_tau]:.2f} {worst}"
                        + ("  LIMIT!" if tau_frac[i_tau] >= 0.95 else ""))
            left = ("booster_deploy monitor\n"
                    "state\nlow_state\njoint_ctrl\nodometry\n"
                    "contact L/R\ntorque/limit\nelastic band\n\n"
                    "M panel  B band  N info  V follow")
            right = ("\n" f"{state}\n{self._state_rate.hz():.0f} Hz\n"
                     f"{self._cmd_rate.hz():.0f} Hz\n{odom_text}\n"
                     f"{contact_text}\n{tau_text}\n{band_text}\n")
            texts.append((int(mujoco.mjtFontScale.mjFONTSCALE_150),
                          int(mujoco.mjtGridPos.mjGRID_TOPLEFT), left, right))
        if self._show_panel:
            names, status = ["FINITE STATE MACHINE"], [""]
            for i, name in enumerate(STATES):
                marker = ">" if i == self._selected else " "
                current = name == state
                allowed = (state, name) in TRANSITIONS
                names.append(f"{marker} {'*' if current else ' '} {name}")
                status.append("current" if current else
                              ("allowed" if allowed else "-"))
            names.append("Up/Down select, Enter switch")
            status.append("")
            texts.append((int(mujoco.mjtFontScale.mjFONTSCALE_150),
                          int(mujoco.mjtGridPos.mjGRID_TOPRIGHT),
                          "\n".join(names), "\n".join(status)))
        return texts

    # ------------------------------------------------------------- status
    def status_line(self) -> str:
        with self._lock:
            q, tau, rpy = self.q.copy(), self.tau.copy(), self.rpy.copy()
            q_cmd = None if self.q_cmd is None else self.q_cmd.copy()
            kp = self.kp_cmd.copy()
            state = self.fsm_state
            state_hz, cmd_hz = self._state_rate.hz(), self._cmd_rate.hz()
            odom = None if self.odom is None else self.odom.copy()
            odom_live = self._odom_rate.hz() > 0
            contact = None if self.contact is None else self.contact.copy()
            contact_live = self._contact_rate.hz() > 0
        tilt = np.degrees(np.arccos(np.clip(
            np.cos(rpy[0]) * np.cos(rpy[1]), -1.0, 1.0)))
        tau_frac = np.abs(tau) / self.effort_limit
        i_tau = int(np.argmax(tau_frac))
        parts = [
            f"state={state:6s}",
            f"low_state={state_hz:5.0f}Hz",
            f"joint_ctrl={cmd_hz:4.0f}Hz",
            f"tilt={tilt:4.1f}deg",
            f"tau/lim={tau_frac[i_tau]:.2f}"
            f"({self.robot_cfg.joint_names[i_tau].replace('_joint', '')})",
        ]
        if tau_frac[i_tau] >= 0.95:
            parts.append("TORQUE LIMIT")
        out_of_range = self._range_limited & (
            (q < self.joint_range[:, 0]) | (q > self.joint_range[:, 1]))
        if out_of_range.any():
            j = int(np.argmax(out_of_range))
            parts.append(
                "JOINT RANGE "
                f"{self.robot_cfg.joint_names[j].replace('_joint', '')}")
        if contact is not None and contact_live:
            parts.append(
                "contact=" + "/".join(f"{c:.0f}" for c in contact) + "N")
        if q_cmd is not None and kp.max() > 0:
            err = np.abs(q - q_cmd)
            i = int(np.argmax(err))
            parts.append(
                f"|q-q_cmd|max={err[i]:.3f}"
                f"({self.robot_cfg.joint_names[i].replace('_joint', '')})")
        if odom is not None and odom_live:
            parts.append(f"odom=({odom[0]:+.2f},{odom[1]:+.2f}) "
                         f"{np.degrees(odom[2]):+.0f}deg")
        else:
            parts.append("odom=none")
        return "  ".join(parts)

    # -------------------------------------------------------------- viewer
    def _update_pose(self) -> None:
        with self._lock:
            q, rpy = self.q.copy(), self.rpy.copy()
            q_cmd = None if self.q_cmd is None else self.q_cmd.copy()
            state = self.fsm_state
            odom = None if self.odom is None else self.odom.copy()
        self.data.qpos[:3] = 0.0
        if odom is not None:
            self.data.qpos[:2] = odom[:2]
            rpy = np.array([rpy[0], rpy[1], odom[2]])
        self.data.qpos[2] = 1.0
        self.data.qpos[3:7] = quat_from_euler_xyz(rpy)
        self.data.qpos[7:] = q
        mujoco.mj_kinematics(self.model, self.data)
        self.data.qpos[2] -= lowest_collision_point(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)
        self.ghost_data.qpos[:7] = self.data.qpos[:7]
        self.ghost_data.qpos[7:] = q if q_cmd is None else q_cmd
        mujoco.mj_kinematics(self.model, self.ghost_data)
        return state

    def _draw_overlay(self, viewer, state: str, show_ghost: bool) -> None:
        scn = viewer.user_scn
        if show_ghost:
            mujoco.mjv_updateScene(
                self.model, self.ghost_data, self._ghost_option, None,
                viewer.cam, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scn)
            for i in range(scn.ngeom):
                scn.geoms[i].rgba[:] = self.ghost_rgba
        else:
            scn.ngeom = 0
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            pos = self.data.qpos[:3] + np.array([0.0, 0.0, 0.45])
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), pos,
                np.eye(3).flatten(), np.array([1, 1, 1, 1], np.float32))
            g.label = f"{state}"
            scn.ngeom += 1

    def run(self, viewer: bool = True, show_ghost: bool = True) -> None:
        executor = SingleThreadedExecutor()
        executor.add_node(self)

        def on_signal(signum, frame):  # noqa: ARG001
            self._stop.set()

        previous = {s: signal.signal(s, on_signal)
                    for s in (signal.SIGINT, signal.SIGTERM)}
        spin_thread = threading.Thread(
            target=self._spin, args=(executor,), name="rclpy", daemon=True)
        spin_thread.start()
        self._spin_thread = spin_thread
        try:
            if viewer:
                self._run_viewer(show_ghost)
            else:
                last = 0.0
                while not self._stop.is_set():
                    time.sleep(0.05)
                    if time.perf_counter() - last >= 0.5:
                        last = time.perf_counter()
                        print(self.status_line(), flush=True)
        finally:
            self._stop.set()
            spin_thread.join(timeout=2.0)
            executor.shutdown(timeout_sec=1.0)
            for s, handler in previous.items():
                signal.signal(s, handler)
            self._save_log()

    def _spin(self, executor: SingleThreadedExecutor) -> None:
        while not self._stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    def _run_viewer(self, show_ghost: bool) -> None:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            self.model, self.data,
            show_left_ui=False, show_right_ui=False,
            key_callback=self._on_key,
        ) as v:
            v.cam.elevation = -20
            v.cam.distance = 2.0
            last = 0.0
            shown_texts = None
            while v.is_running() and not self._stop.is_set():
                self._service_pending()
                state = self._update_pose()
                with v.lock():
                    self._draw_overlay(v, state, show_ghost)
                    if self._follow:
                        v.cam.lookat[:] = self.data.qpos[:3]
                texts = self._overlay_texts(state)
                if texts != shown_texts:
                    shown_texts = texts
                    if texts:
                        v.set_texts(texts)
                    else:
                        v.clear_texts()
                v.sync()
                if time.perf_counter() - last >= 0.5:
                    last = time.perf_counter()
                    print("\r" + self.status_line() + "   ", end="",
                          flush=True)
                time.sleep(1.0 / 60.0)
            print()
            # Quiesce the ROS thread and drop the overlays before the viewer
            # tears down its GL context; racing that crashed the window.
            self._stop.set()
            self._spin_thread.join(timeout=2.0)
            with v.lock():
                v.user_scn.ngeom = 0
            v.clear_texts()
            v.sync()

    def _save_log(self) -> None:
        if self._log_path is None or not self._log["time"]:
            return
        path = self._log_path
        if not path.endswith(".npz"):
            path += ".npz"
        with self._lock:
            arrays = {k: np.asarray(v) for k, v in self._log.items()}
        np.savez(path, **arrays)
        print(f"[monitor] saved {len(arrays['time'])} samples to {path}",
              flush=True)

"""Child-process side of the deployment FSM: runs the active state.

The portal (main process) owns the ROS 2 I/O and the Loco RPC and decides
which state is requested; this executor runs in the inference process at
``policy_dt`` and performs the state's control behaviour:

- ``WALK`` / ``TASK``: step a :class:`BoosterRobotController` (policy
  inference + ``/joint_ctrl`` publishing) in the firmware's Custom mode;
- ``ESTOP``: publish one damping command (Kp 0), then nothing;
- ``IDLE`` / ``STAND``: publish nothing; the firmware's own controller
  (Damping, Prepare or Walking mode) owns the joints.

Both policy controllers are constructed up-front so that switching states
never stalls on model loading.

The child publishes ``/joint_ctrl`` through the publisher it inherits from
the portal, as Booster's own inference process does.  With
``booster.executor_ros_context`` it creates its own ROS 2 context, node and
publisher instead: an inherited publisher keeps working for subscribers
matched before the fork, but the DDS discovery threads do not survive the
fork, so subscribers that appear later (a monitor, a restarted simulator)
would never receive anything.  Creating DDS entities in a forked process can
hang on some Fast DDS builds, which is why the inherited publisher is the
default on the robot.  Either way the child reports readiness through
``portal.fsm_executor_ready`` once both policy controllers are built.
"""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Optional

import rclpy
from booster_interface.msg import LowCmd
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from ..controllers.base_controller import BaseController
from ..controllers.booster_robot_controller import BoosterRobotController
from ..controllers.controller_cfg import ControllerCfg
from .booster_states import ESTOP, STAND, TASK, WALK, state_index, state_name

if TYPE_CHECKING:
    from ..controllers.booster_robot_controller import BoosterRobotPortal

logger = logging.getLogger("booster_deploy.fsm")


class FsmPolicyController(BoosterRobotController):
    """A policy controller whose stop/finish become FSM transitions."""

    def __init__(
        self,
        cfg: ControllerCfg,
        portal: "BoosterRobotPortal",
        executor: "FsmExecutor",
    ) -> None:
        super().__init__(cfg, portal)
        self._executor = executor
        # Velocity commands are live whenever the state is active.
        self._velocity_commands_enabled = True

    def tick(self) -> None:
        self.update_state()
        if self.vel_command is not None:
            self.update_vel_command()
        self.portal.metrics["policy_step"].mark()
        dof_targets = self.policy_step()
        self.ctrl_step(dof_targets)

    def stop(self) -> None:
        # Safety fallback or unexpected policy stop -> ESTOP.
        BaseController.stop(self)
        self._executor.request(ESTOP, "policy stop")

    def finish(self) -> None:
        # Normal completion (e.g. motion end) -> configured follow-up state.
        BaseController.stop(self)
        self._executor.request(self._executor.after_task, "policy finished")


class FsmExecutor:
    def __init__(
        self,
        portal: "BoosterRobotPortal",
        task_cfg: ControllerCfg,
        walk_cfg: Optional[ControllerCfg],
    ) -> None:
        self.portal = portal
        self.policy_dt = float(task_cfg.policy_dt)
        self.after_task = task_cfg.booster.after_task.strip().upper()
        if self.after_task not in (STAND, WALK):
            raise ValueError(
                f"booster.after_task must be 'stand' or 'walk', got "
                f"{task_cfg.booster.after_task!r}")
        if self.after_task == WALK and walk_cfg is None:
            self.after_task = STAND
        self.walk = (FsmPolicyController(walk_cfg, portal, self)
                     if walk_cfg is not None else None)
        self.task = FsmPolicyController(task_cfg, portal, self)
        self.active = state_name(portal.fsm_active.value)
        self._pending: Optional[str] = None

    # ------------------------------------------------------------ requests
    def request(self, target: str, reason: str) -> None:
        """Executor-initiated transition (policy stop/finish)."""
        logger.info("FSM executor requests %s (%s)", target, reason)
        self._pending = target

    # ------------------------------------------------------------ switching
    def _switch(self, target: str) -> None:
        if target == self.active:
            return
        logger.info("FSM executor: %s -> %s", self.active, target)
        if target == WALK:
            if self.walk is None:  # the portal refuses WALK in this case
                logger.error("no locomotion policy for WALK; not publishing")
            else:
                self.walk.update_state()
                self.walk.start()
        elif target == TASK:
            self.task.update_state()
            self.task.start()
        elif target == ESTOP:
            self._publish_damping()
        self.active = target
        self.portal.fsm_active.value = state_index(target)

    def _publish_damping(self) -> None:
        state = self.portal.synced_state.read()[0]
        prepare = self.portal.robot.cfg.prepare_state
        motor_cmd = self.portal.motor_cmd
        for i in range(self.portal.robot.num_joints):
            motor_cmd[i].q = float(state["joint_pos"][i])
            motor_cmd[i].dq = 0.0
            motor_cmd[i].tau = 0.0
            motor_cmd[i].kp = 0.0
            motor_cmd[i].kd = float(prepare.damping[i])
        self.portal.low_cmd_publisher.publish(self.portal.low_cmd)

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        portal = self.portal
        next_t = time.perf_counter()
        while not portal.exit_event.is_set():
            if time.perf_counter() < next_t:
                time.sleep(0.0002)
                continue
            next_t += self.policy_dt

            if self._pending is not None:
                target, self._pending = self._pending, None
                self._switch(target)
                # Update the shared request too, so the loop does not fall
                # back to the portal's stale request before it has synced.
                portal.fsm_requested.value = state_index(target)
                portal.fsm_executor_request.value = state_index(target)
            else:
                requested = state_name(portal.fsm_requested.value)
                if requested != self.active:
                    self._switch(requested)

            if self.active == WALK and self.walk is not None:
                self.walk.tick()
            elif self.active == TASK:
                self.task.tick()
            # IDLE / STAND / ESTOP: the firmware owns the joints
        logger.info("FSM executor stopped in %s", self.active)


def fsm_process_func(
    portal: "BoosterRobotPortal",
    task_cfg: ControllerCfg,
    walk_cfg: Optional[ControllerCfg],
) -> None:
    """Entry point of the inference process."""
    t0 = time.perf_counter()
    context = node = None
    if task_cfg.booster.executor_ros_context:
        # Replace the inherited publisher (see the module docstring); the
        # controllers publish through ``portal.low_cmd_publisher``.
        context = rclpy.Context()
        rclpy.init(context=context)
        node = rclpy.create_node(
            "booster_deploy_fsm_executor", context=context)
        portal.low_cmd_publisher = node.create_publisher(
            LowCmd, "joint_ctrl",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST),
        )
        logger.info("FSM executor: own ROS 2 node up after %.1fs",
                    time.perf_counter() - t0)
    try:
        executor = FsmExecutor(portal, task_cfg, walk_cfg)
        portal.fsm_executor_ready.value = 1
        logger.info("FSM executor ready after %.1fs (pid %d, walk %s)",
                    time.perf_counter() - t0, os.getpid(),
                    "yes" if executor.walk is not None else "no")
        executor.run()
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown(context=context)

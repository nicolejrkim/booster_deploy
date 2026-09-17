"""BM154 motion-tracking policy: BeyondMimic with a hardware-style observation.

BM154 is the actor observation of the ``Tracking-Flat-*-BM154-v0`` tasks in
``whole_body_tracking`` (the name comes from its 154-dim G1 layout).
Compared with the default BeyondMimic observation it drops the anchor
position and the base linear velocity, and it expresses orientation as
gravity directions in the root (IMU) frame, so every term is directly
measurable on the robot.  On the 22-DOF K1 the layout is 119 dims, in this
order:

    command                   2 * num_joints  reference joint pos, joint vel
    motion_projected_gravity  3               gravity in reference root frame
    projected_gravity         3               gravity in robot root frame
    base_ang_vel              3               gyro
    joint_pos                 num_joints      joint pos - default joint pos
    joint_vel                 num_joints
    actions                   num_joints      previous raw policy output

Joint terms use the training (simulation) joint order.  Motion loading, action
scaling, ghost rendering and the safety fallback are inherited from
:class:`BeyondMimicPolicy`.
"""
from __future__ import annotations

import torch

from booster_deploy.controllers.base_controller import BaseController
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass

from ..beyond_mimic import BeyondMimicPolicy, BeyondMimicPolicyCfg


class BM154Policy(BeyondMimicPolicy):
    def __init__(self, cfg: BM154PolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self._gravity_w = torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float32, device=self.cfg.device)

    @property
    def motion_root_body_name(self) -> str:
        return self.cfg.motion_root_body_name or self.cfg.anchor_body_name

    def _motion_track_body_names(self) -> list[str]:
        names = [self.cfg.anchor_body_name]
        if self.motion_root_body_name != self.cfg.anchor_body_name:
            names.append(self.motion_root_body_name)
        return names

    def reset(self) -> None:
        super().reset()
        self.root_index = self.motion.track_body_names.index(
            self.motion_root_body_name)
        self._gravity_w = self._gravity_w.to(self.cfg.device)

    def compute_observation(self) -> torch.Tensor:
        self._set_command()
        row_ids = min(self.current_frame, self.motion.time_step_total - 1)
        cmd_root_quat_w = self.motion.body_quat_w[row_ids, self.root_index]

        command = torch.cat([self.cmd_dof_pos, self.cmd_dof_vel], dim=0)
        # Gravity directions are yaw-invariant, so neither the motion's
        # first-frame alignment nor the IMU yaw drift affects these terms.
        motion_projected_gravity = lab_math.quat_apply_inverse(
            cmd_root_quat_w, self._gravity_w)
        projected_gravity = lab_math.quat_apply_inverse(
            self.robot.data.root_quat_w, self._gravity_w)

        real2sim_map = self.robot.data.real2sim_joint_indexes
        joint_pos = (
            self.robot.data.joint_pos[real2sim_map]
            - self.default_joint_pos[real2sim_map]
        )
        joint_vel = self.robot.data.joint_vel[real2sim_map]

        obs = torch.cat(
            (
                command,
                motion_projected_gravity,
                projected_gravity,
                self.robot.data.root_ang_vel_b,
                joint_pos,
                joint_vel,
                self.last_action,
            ),
            dim=-1,
        )
        return obs.reshape(1, -1)


@configclass
class BM154PolicyCfg(BeyondMimicPolicyCfg):
    constructor = BM154Policy
    # Motion body whose orientation gives the reference projected gravity: the
    # floating base of the training robot.  ``None`` uses ``anchor_body_name``
    # (K1: both are ``trunk``; G1 would use ``pelvis`` with a ``torso_link``
    # anchor).
    motion_root_body_name: str | None = None

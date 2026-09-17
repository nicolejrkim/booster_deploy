from __future__ import annotations
from dataclasses import MISSING
import os

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import PolicyCfg
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.motion_loader import MotionLoader
from booster_deploy.utils.policy_runner import create_policy_runner


class BeyondMimicPolicy(Policy):
    def __init__(self, cfg: BeyondMimicPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        checkpoint_path = self.cfg.checkpoint_path
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(self.task_path, checkpoint_path)
        self._model = create_policy_runner(
            checkpoint_path,
            torch.device(self.cfg.device),
        )

        self.robot = controller.robot

        if self.cfg.action_scale_factors is not None:
            action_scale_factors = torch.tensor(
                self.cfg.action_scale_factors,
                dtype=torch.float32,
                device=self.cfg.device,
            )
            self.action_scale = (
                action_scale_factors
                * self.robot.effort_limit
                / self.robot.joint_stiffness
            ).to(self.cfg.device)
        elif self.cfg.fixed_action_scale is None:
            self.action_scale = (
                0.25 * self.robot.effort_limit / self.robot.joint_stiffness
            ).to(self.cfg.device)
        else:
            self.action_scale = torch.tensor(
                self.cfg.fixed_action_scale,
                dtype=torch.float32,
                device=self.cfg.device,
            )

        self.robot.data.to(self.cfg.device)

        self.motion = MotionLoader(
            motion_file=f"{self.task_path}/{self.cfg.motion_path}",
            track_body_names=self._motion_track_body_names(),
            track_joint_names=self.robot.cfg.sim_joint_names,
            default_motion_body_names=self.robot.cfg.sim_body_names,
            default_motion_joint_names=self.robot.cfg.sim_joint_names,
            frame_range=self.cfg.motion_frame_range,
            align_to_first_frame=True,
            device=self.cfg.device
        )

        self.default_joint_pos = self.robot.default_joint_pos.to(
            self.cfg.device)

    def _motion_track_body_names(self) -> list[str]:
        """Motion bodies loaded from the motion file.

        The anchor body is always first; subclasses may append more bodies.
        """
        return [self.cfg.anchor_body_name]

    def reset(self) -> None:
        self.init_root_yaw_quat_w_inv = lab_math.quat_inv(
            lab_math.yaw_quat_zxy(self.robot.data.root_quat_w))
        self.anchor_index = self.motion.track_body_names.index(
            self.cfg.anchor_body_name)
        self.current_frame = 0
        self.last_action = torch.zeros(
            self.robot.num_joints,
            dtype=torch.float32, device=self.cfg.device)
        self.motion.to(self.cfg.device)

    def _set_command(self):
        row_ids = min(self.current_frame, self.motion.time_step_total - 1)

        self.cmd_dof_pos = self.motion.joint_pos[row_ids]
        self.cmd_dof_vel = self.motion.joint_vel[row_ids]

        self.cmd_root_pos_w = self.motion.body_pos_w[
            row_ids, self.anchor_index]
        self.cmd_root_quat_w = self.motion.body_quat_w[
            row_ids, self.anchor_index]

    def compute_observation(self) -> torch.Tensor:
        """Computes observations"""
        self._set_command()

        command = torch.cat([self.cmd_dof_pos, self.cmd_dof_vel], dim=0)
        cur_root_quat_w = lab_math.quat_mul(
            self.init_root_yaw_quat_w_inv, self.robot.data.root_quat_w)

        pos, ori = lab_math.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            cur_root_quat_w,
            self.cmd_root_pos_w,
            self.cmd_root_quat_w,
        )

        motion_anchor_pos_b = pos  # noqa: F841

        motion_anchor_ori_b = lab_math.matrix_from_quat(ori)[..., :2].flatten()

        real2sim_map = self.robot.data.real2sim_joint_indexes
        dof_pos = self.robot.data.joint_pos[real2sim_map]
        joint_pos = dof_pos - self.default_joint_pos[real2sim_map]
        joint_vel = self.robot.data.joint_vel[real2sim_map]

        obs = torch.cat(
            (
                command,
                # motion_anchor_pos_b,    # linear states
                motion_anchor_ori_b,
                # self.robot.data.root_lin_vel_b,           # linear states
                self.robot.data.root_ang_vel_b,
                joint_pos,
                joint_vel,
                self.last_action,
            ),
            dim=-1,
        )
        return obs.reshape(1, -1)

    def inference(self) -> torch.Tensor:
        """Called by the controller each step to obtain the action tensor.

        Reads `robot.data` and velocity commands from the controller,
        runs the underlying model's inference, and returns an action
        as a `torch.Tensor`.
        """

        with torch.no_grad():
            obs = self.compute_observation()
            action = self._model(obs).flatten()

        # for motion visualization in Mujoco controller
        if hasattr(self.controller, "set_reference_qpos"):
            joint_pos = self.cmd_dof_pos[self.robot.data.sim2real_joint_indexes]
            ref_qpos = torch.cat(
                [self.cmd_root_pos_w, self.cmd_root_quat_w, joint_pos],
                dim=0,
            )
            self.controller.set_reference_qpos(ref_qpos)    # type: ignore

        self.current_frame += 1
        self.last_action = action

        if (
            self.cfg.stop_at_motion_end
            and self.current_frame >= self.motion.time_step_total
        ):
            self.controller.finish()

        if action is None:
            raise RuntimeError("Underlying model returned None from inference")

        if self.cfg.enable_safety_fallback:
            # monitor policy state validity during execution.
            gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32,
                                     device=self.cfg.device)
            projected_gravity = lab_math.quat_apply_inverse(
                self.robot.data.root_quat_w, gravity_w)
            motion_projected_gravity = lab_math.quat_apply_inverse(
                self.cmd_root_quat_w, gravity_w)

            if torch.dot(projected_gravity, motion_projected_gravity) < 0.5:
                print("\nLarge root tracking error is detected, stopping policy"
                      " for safety. You can disable safety fallback by setting "
                      f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                      "to False.")
                self.controller.stop()

        sim2real_map = self.robot.data.sim2real_joint_indexes
        return (
            action[sim2real_map] * self.action_scale
            + self.default_joint_pos
        )


@configclass
class BeyondMimicPolicyCfg(PolicyCfg):
    constructor = BeyondMimicPolicy
    checkpoint_path: str = MISSING
    motion_path: str = MISSING
    motion_frame_range: tuple[int, int] | list[int] | None = None
    fixed_action_scale: list[float] | None = None
    action_scale_factors: list[float] | None = None
    stop_at_motion_end: bool = False

    anchor_body_name: str = "trunk"

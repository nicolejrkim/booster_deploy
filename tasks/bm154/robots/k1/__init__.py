"""K1 BM154 motion-tracking tasks.

The policies were trained with ``Tracking-Flat-K1-BM154-v0`` from
``whole_body_tracking`` (BeyondMimic).  The gains, effort limits, default pose
and action scale below mirror ``whole_body_tracking/robots/k1.py`` so that the
mapping from a policy output to a joint target is identical to training.
"""
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
)
from booster_deploy.robots.k1 import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from ...bm154 import BM154PolicyCfg

# Real joint order: head 2, left arm 4, right arm 4, left leg 6, right leg 6.
# Leg order: hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll.
TRAIN_JOINT_STIFFNESS = (
    [8.0] * 2
    + [15.0] * 8
    + [100.0, 100.0, 100.0, 100.0, 50.0, 50.0] * 2
)
TRAIN_JOINT_DAMPING = (
    [0.4] * 2
    + [0.5] * 8
    + [2.0, 2.0, 2.0, 2.0, 1.0, 1.0] * 2
)
# Official K1_22dof effort limits (also the MuJoCo ``forcerange``).
TRAIN_EFFORT_LIMIT = (
    [6.0] * 2
    + [14.0] * 8
    + [68.0, 43.0, 38.3, 112.0, 38.3, 38.3] * 2
)
# Training keyframe: slight crouch, arms down at the sides.
TRAIN_DEFAULT_JOINT_POS = [
    0.0, 0.0,
    0.2, -1.35, 0.0, -0.5,
    0.2, 1.35, 0.0, 0.5,
    -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
]
# Isaac Lab JointPositionAction scale: a unit action is a quarter of the
# torque limit divided by the stiffness.  Kept fixed so that re-tuning the
# real-robot PD gains does not silently change the policy's action mapping.
TRAIN_ACTION_SCALE = [
    0.25 * effort / stiffness
    for effort, stiffness in zip(TRAIN_EFFORT_LIMIT, TRAIN_JOINT_STIFFNESS)
]


@configclass
class K1BM154ControllerCfg(ControllerCfg):
    robot = K1_CFG.replace(  # type: ignore
        joint_stiffness=list(TRAIN_JOINT_STIFFNESS),
        joint_damping=list(TRAIN_JOINT_DAMPING),
        effort_limit=list(TRAIN_EFFORT_LIMIT),
        default_joint_pos=list(TRAIN_DEFAULT_JOINT_POS),
    )
    policy: BM154PolicyCfg = BM154PolicyCfg(
        anchor_body_name="trunk",
        fixed_action_scale=list(TRAIN_ACTION_SCALE),
    )
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.55],
        visualize_reference_ghost=True,
    )


@configclass
class K1BM154JamesBrownControllerCfg(K1BM154ControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = (
            "robots/k1/motions/k1_dance_jamesbrown_marg_stmr.npz"
        )
        self.policy.checkpoint_path = (
            "robots/k1/models/k1_dance_jamesbrown_marg_bm154.pt"
        )


@configclass
class K1BM154FlossControllerCfg(K1BM154ControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = (
            "robots/k1/motions/k1_dance_floss_marg_stmr.npz"
        )
        self.policy.checkpoint_path = (
            "robots/k1/models/k1_dance_floss_marg_bm154.pt"
        )


@configclass
class K1BM154BoogleControllerCfg(K1BM154ControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = (
            "robots/k1/motions/k1_dance_boogle_marg_stmr.npz"
        )
        self.policy.checkpoint_path = (
            "robots/k1/models/k1_dance_boogle_marg_bm154.pt"
        )


register_task("k1_bm154_jamesbrown", K1BM154JamesBrownControllerCfg())
register_task("k1_bm154_floss", K1BM154FlossControllerCfg())
register_task("k1_bm154_boogle", K1BM154BoogleControllerCfg())

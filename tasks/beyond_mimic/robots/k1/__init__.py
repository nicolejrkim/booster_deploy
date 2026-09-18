from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
)
from booster_deploy.robots.k1 import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from ...beyond_mimic import BeyondMimicPolicyCfg


@configclass
class K1BeyondMimicControllerCfg(ControllerCfg):
    robot = K1_CFG.replace(  # type: ignore
        joint_stiffness=[
            4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            80., 80.0, 80., 80., 30., 30.,
            80., 80.0, 80., 80., 30., 30.,
        ],
        joint_damping=[
            1., 1.,
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2.,
        ],
    )
    enable_velocity_commands = False
    policy: BeyondMimicPolicyCfg = BeyondMimicPolicyCfg(
        anchor_body_name="trunk",
    )
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.57],
        visualize_reference_ghost=True,
    )


@configclass
class K1MJ2ControllerCfg(K1BeyondMimicControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "robots/k1/motions/k1_mj2_seg1.npz"
        self.policy.checkpoint_path = (
            "robots/k1/models/k1_mj_dance_002_2025-12-03_00-10-28.pt"
        )
        self.robot.joint_stiffness = [
            10.0, 10.0,
            4., 4., 4., 4.,
            4., 4., 4., 4.,
            80., 80., 80., 80., 30., 30.,
            80., 80., 80., 80., 30., 30.,
        ]
        self.robot.joint_damping = [
            2., 2.,
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2.,
        ]
        self.robot.effort_limit = [
            6, 6,
            14, 14, 14, 14,
            14, 14, 14, 14,
            30, 35, 20, 40, 20, 20,
            30, 35, 20, 40, 20, 20,
        ]


@configclass
class K1FightControllerCfg(K1BeyondMimicControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "robots/k1/motions/k1_fight_final_deploy.npz"
        self.policy.checkpoint_path = "robots/k1/models/k1_fight_001.pt"
        self.robot.joint_stiffness = [
            10.0, 10.0,
            3.95, 3.95, 3.95, 3.95,
            3.95, 3.95, 3.95, 3.95,
            80., 80., 80., 80., 30., 30.,
            80., 80., 80., 80., 30., 30.,
        ]
        self.robot.joint_damping = [
            2., 2.,
            0.3, 0.3, 0.3, 0.3,
            0.3, 0.3, 0.3, 0.3,
            2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2.,
        ]
        self.robot.effort_limit = [
            4, 4,
            12, 12, 12, 12,
            12, 12, 12, 12,
            30, 35, 20, 40, 20, 20,
            30, 35, 20, 40, 20, 20,
        ]


register_task("k1_mj2", K1MJ2ControllerCfg())
register_task("k1_fight", K1FightControllerCfg())


# --- Policies trained with Booster's booster_train ---------------------------
# (see scripts/booster_train_pipeline.py).  The values below are what the
# booster_train K1 actuator model trains with (Kp = J (2 pi f)^2,
# Kd = 2 zeta J 2 pi f; f = 4 Hz for the legs, 10 Hz for arms and head), read
# from a run's saved env config.  The action scale is pinned so that tuning
# Kp/Kd on the robot never changes what the policy commands.
# Real joint order: head 2, left arm 4, right arm 4, left leg 6, right leg 6
# (leg: hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll).
BT_JOINT_STIFFNESS = (
    [3.9478] * 2 + [3.9478] * 8
    + [30.201, 21.448, 17.846, 60.402, 35.692, 35.692] * 2
)
BT_JOINT_DAMPING = (
    [0.2513] * 2 + [0.2513] * 8
    + [3.605, 2.5602, 2.1302, 4.8066, 4.2604, 4.2604] * 2
)
BT_EFFORT_LIMIT = (
    [6.0] * 2 + [14.0] * 8 + [68.0, 76.0, 38.3, 112.0, 38.3, 38.3] * 2
)
BT_ACTION_SCALE = [
    0.25 * effort / stiffness
    for effort, stiffness in zip(BT_EFFORT_LIMIT, BT_JOINT_STIFFNESS)
]


@configclass
class K1BoosterTrainControllerCfg(ControllerCfg):
    """K1 motion tracking policies trained with booster_train.

    Same anchor-orientation observation as ``k1_fight``; gains, effort limits
    and action scale mirror the booster_train actuator model.
    """

    robot = K1_CFG.replace(  # type: ignore
        joint_stiffness=list(BT_JOINT_STIFFNESS),
        joint_damping=list(BT_JOINT_DAMPING),
        effort_limit=list(BT_EFFORT_LIMIT),
    )
    policy: BeyondMimicPolicyCfg = BeyondMimicPolicyCfg(
        anchor_body_name="trunk",
        fixed_action_scale=list(BT_ACTION_SCALE),
    )
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.57],
        visualize_reference_ghost=True,
    )


def register_booster_train_dance(task_name: str, motion_name: str) -> None:
    """Register a booster_train policy exported by the pipeline script.

    Expects ``robots/k1/models/<motion_name>_bt.pt`` and
    ``robots/k1/motions/<motion_name>.npz``.
    """
    cfg = K1BoosterTrainControllerCfg()
    cfg.policy.motion_path = f"robots/k1/motions/{motion_name}.npz"
    cfg.policy.checkpoint_path = f"robots/k1/models/{motion_name}_bt.pt"
    register_task(task_name, cfg)

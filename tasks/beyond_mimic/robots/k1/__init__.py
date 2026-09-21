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


def register_booster_train_dance(
    task_name: str,
    motion_name: str,
    cfg_cls: type = K1BoosterTrainControllerCfg,
    suffix: str = "_bt",
) -> None:
    """Register a booster_train policy exported by the pipeline script.

    Expects ``robots/k1/models/<motion_name><suffix>.pt`` and
    ``robots/k1/motions/<motion_name>.npz``; ``cfg_cls`` picks the gain set
    the run was trained with.
    """
    cfg = cfg_cls()
    cfg.policy.motion_path = f"robots/k1/motions/{motion_name}.npz"
    cfg.policy.checkpoint_path = f"robots/k1/models/{motion_name}{suffix}.pt"
    register_task(task_name, cfg)


# Trained runs, task name (without the ``k1_bt_`` prefix) -> motion stem
# (``models/<stem>_bt.pt``, ``motions/<stem>.npz``).  ``dance_*`` are the
# three STMR dances; the others are the showcase (``k1s_*_ext``) and, with
# the ``sp_`` prefix, the seedpicks (``spk1_*_k1b15``) retargeting sets,
# ``aNNN`` being the source clip.  All trained on Euler, 10 000 iterations.
BOOSTER_TRAIN_TASKS = [
    ("dance_jamesbrown_stmr", "k1_dance_jamesbrown_marg_stmr"),
    ("dance_floss_stmr", "k1_dance_floss_marg_stmr"),
    ("dance_boogle_stmr", "k1_dance_boogle_marg_stmr"),
    ("basic_chaines_180_a306",
     "k1s_dance_basic_chaines_180_R_001__A306_ext_stmr"),
    ("cartwheel_a415", "k1s_cartwheel_R_001__A415_ext_stmr"),
    ("cartwheel_a416", "k1s_cartwheel_R_001__A416_ext_stmr"),
    ("flip_360_a415", "k1s_flip_360_001__A415_ext_stmr"),
    ("high_jump_a277", "k1s_high_jump_R_001__A277_ext_stmr"),
    ("high_jump_a340m", "k1s_high_jump_R_002__A340_M_ext_stmr"),
    ("hiphop_floss_a316",
     "k1s_dance_hiphop_floss_R_fast_003__A316_ext_stmr"),
    ("hiphop_james_brown_a320",
     "k1s_dance_hiphop_james_brown_R_fast_004__A320_ext_stmr"),
    ("hiphop_shuffle_square_a318",
     "k1s_dance_hiphop_shuffle_square_R_fast_002__A318_ext_stmr"),
    ("ib_dodge_270_a437", "k1s_ib_dodge_270_R_001__A437_ext_stmr"),
    ("ib_dodge_back_a437", "k1s_ib_dodge_back_L_002__A437_ext_stmr"),
    ("ib_dodge_up_a437", "k1s_ib_dodge_up_R_001__A437_ext_stmr"),
    ("jump_around_a493", "k1s_jump_around_001__A493_ext_stmr"),
    ("turn_jump_0045_a023", "k1s_turn_jump_0045_007__A023_ext_stmr"),
    ("turn_jump_135_a037", "k1s_turn_jump_135_002__A037_ext_stmr"),
    ("vouge_boogle_180_a317",
     "k1s_dance_vouge_dancehall_open_close_boogle_180_R_fast_002__A317"
     "_ext_stmr"),
    ("vouge_duck_walk_180_a319",
     "k1s_dance_vouge_the_duck_walk_180_R_fast_001__A319_ext_stmr"),
    ("sp_frog_jump_a360", "spk1_frog_jump_002__A360_k1b15_stmr"),
    ("sp_high_jump_a277", "spk1_high_jump_R_001__A277_k1b15_stmr"),
    ("sp_jump_sideway_090_a024",
     "spk1_jump_sideway_090_002__A024_k1b15_stmr"),
    ("sp_scissors_jump_a360", "spk1_scissors_jump_R_003__A360_k1b15_stmr"),
    ("sp_ib_dodge_270_a437", "spk1_ib_dodge_270_R_001__A437_k1b15_stmr"),
    ("sp_turn_jump_0045_a023", "spk1_turn_jump_0045_007__A023_k1b15_stmr"),
    ("sp_turn_jump_135_a037", "spk1_turn_jump_135_002__A037_k1b15_stmr"),
]

for _name, _stem in BOOSTER_TRAIN_TASKS:
    register_booster_train_dance(f"k1_bt_{_name}", _stem)

# --- booster_train retrains with the k1_mj2 deploy gains -------------------
# (``Booster-K1-*-Mj2-v0`` tasks on ``BOOSTER_K1_MJ2_CFG``, 2026-09-20.)
# Same booster_train actuator model (delay, torque-speed knee, sim torque
# limits) but kp/kd = the K1_CFG / k1_mj2 deploy gains (head 10/2, arms 4/1,
# hips + knee 80/2, ankles 30/2); the training action scale is
# 0.25 * sim effort / kp, pinned here so re-tuning the robot's PD never
# changes what the policy commands.  Models: ``models/<stem>_mj2_bt.pt``.
MJ2_JOINT_STIFFNESS = (
    [10.0] * 2 + [4.0] * 8 + [80.0, 80.0, 80.0, 80.0, 30.0, 30.0] * 2
)
MJ2_JOINT_DAMPING = [2.0] * 2 + [1.0] * 8 + [2.0] * 6 * 2
MJ2_ACTION_SCALE = [
    0.25 * effort / stiffness
    for effort, stiffness in zip(BT_EFFORT_LIMIT, MJ2_JOINT_STIFFNESS)
]


@configclass
class K1BoosterTrainMj2ControllerCfg(K1BoosterTrainControllerCfg):
    """booster_train policies trained on ``BOOSTER_K1_MJ2_CFG``."""

    def __post_init__(self):
        super().__post_init__()
        self.robot.joint_stiffness = list(MJ2_JOINT_STIFFNESS)
        self.robot.joint_damping = list(MJ2_JOINT_DAMPING)
        self.policy.fixed_action_scale = list(MJ2_ACTION_SCALE)


# The clips retrained so far (task name without the ``k1_bt2_`` prefix ->
# motion stem, same motions as above); extend as more ``-Mj2-v0`` runs land.
BOOSTER_TRAIN_MJ2_TASKS = [
    ("dance_jamesbrown_stmr", "k1_dance_jamesbrown_marg_stmr"),
    ("dance_floss_stmr", "k1_dance_floss_marg_stmr"),
    ("high_jump_a277", "k1s_high_jump_R_001__A277_ext_stmr"),
    ("ib_dodge_270_a437", "k1s_ib_dodge_270_R_001__A437_ext_stmr"),
    ("turn_jump_0045_a023", "k1s_turn_jump_0045_007__A023_ext_stmr"),
    ("sp_high_jump_a277", "spk1_high_jump_R_001__A277_k1b15_stmr"),
    ("sp_jump_sideway_090_a024",
     "spk1_jump_sideway_090_002__A024_k1b15_stmr"),
]

for _name, _stem in BOOSTER_TRAIN_MJ2_TASKS:
    register_booster_train_dance(
        f"k1_bt2_{_name}", _stem, K1BoosterTrainMj2ControllerCfg, "_mj2_bt")

from typing import Callable, List, Optional
from dataclasses import MISSING
import torch

from ..utils.isaaclab.configclass import configclass


@configclass
class PrepareStateCfg:
    stiffness: List[float] = MISSING
    damping: List[float] = MISSING
    joint_pos: List[float] = MISSING


@configclass
class MujocoControllerCfg:
    init_pos: List[float] = [0.0, 0.0, 0.6]
    init_quat: List[float] = [1.0, 0.0, 0.0, 0.0]
    decimation: int = 10
    # physics_dt will automatically be set by ControllerCfg
    physics_dt: float = None  # type: ignore
    log_states: Optional[str] = None
    log_joint_torque_csv: Optional[str] = None
    log_joint_velocity_csv: Optional[str] = None
    log_joint_position_csv: Optional[str] = None
    visualize_reference_ghost: bool = False
    ghost_rgba: List[float] = [0.2, 0.8, 0.2, 0.25]
    show_left_ui: bool = False
    show_right_ui: bool = False
    # Headless run that writes an mp4 instead of opening the viewer (deploy.py --record): offscreen EGL render of the
    # simulated robot + reference ghost, camera tracking the base.
    record: Optional[str] = None
    record_fps: int = 25
    record_size: List[int] = [854, 480]
    cam_distance: float = 2.4
    cam_azimuth: float = 135.0
    cam_elevation: float = -18.0
    record_max_steps: Optional[int] = None  # safety cap; default = motion length + 100 control steps


@configclass
class BoosterRobotControllerCfg:
    metrics_max_events: int = 2000
    # Firmware mode the robot is handed to when the deployment exits from a
    # Custom state (WALK, TASK): "walking" (Booster's controller, IDLE) or
    # "damping" (ESTOP).  In the other states the firmware already controls
    # the robot and is left as is.
    exit_mode: str = "walking"
    # State entered when the task policy finishes (e.g. motion end):
    # "stand" (Prepare mode) or "walk" (the locomotion policy).
    after_task: str = "stand"
    # Booster get-up used for IDLE -> STAND when the robot is not upright:
    # GetUpVersion (0 = V1 for K1/T1/T2, 1 = V2, K1 only) and how long to
    # wait for the robot to be up (the firmware gets up into Walking mode,
    # then Prepare is requested).
    getup_version: int = 0
    getup_timeout_s: float = 20.0
    # How often the supervisor reads the robot's mode (GetStatus RPC) and
    # follows it when the firmware changed mode on its own (fall protection,
    # a restart, the operator app or remote).
    mode_check_period_s: float = 1.0


@configclass
class RobotCfg:
    name: str = MISSING
    # Where A goes from STAND: "walking" inserts WALK (the robot's locomotion
    # policy, tasks/locomotion) before TASK, "standing" enters TASK straight
    # from Prepare mode.
    prepare_mode: str = "walking"

    joint_names: list[str] = MISSING
    body_names: list[str] = MISSING

    sim_joint_names: list[str] = MISSING
    sim_body_names: list[str] = MISSING

    joint_stiffness: List[float] = MISSING
    joint_damping: List[float] = MISSING

    default_joint_pos: List[float] = MISSING
    effort_limit: List[float] = MISSING

    mjcf_path: str = MISSING

    prepare_state: PrepareStateCfg = MISSING

    def __post_init__(self):
        assert (
            len(self.joint_names)
            == len(self.joint_stiffness)
            == len(self.joint_damping)
            == len(self.default_joint_pos)
            == len(self.effort_limit)
        )


@configclass
class VelocityCommandCfg:
    vx_max: float = 1.0
    # Direction-specific forward velocity limits.  Keeping these at the
    # default value preserves the historical symmetric +/-vx_max behavior.
    vx_forward_max: Optional[float] = None
    vx_backward_max: Optional[float] = None
    vy_max: float = 1.0
    vyaw_max: float = 1.0


@configclass
class PolicyCfg:
    constructor: Callable = MISSING
    checkpoint_path: str = MISSING
    enable_safety_fallback: bool = True
    device: str | torch.device = "cpu"


@configclass
class EvaluatorCfg:
    constructor: Callable = MISSING
    # Rendering
    render: bool = True


@configclass
class ControllerCfg:
    """Controller configuration class.
    """

    policy_dt: float = 0.02
    robot: RobotCfg = MISSING
    vel_command: Optional[VelocityCommandCfg] = None
    policy: PolicyCfg = MISSING

    mujoco: MujocoControllerCfg = MujocoControllerCfg()
    booster: BoosterRobotControllerCfg = BoosterRobotControllerCfg()
    evaluator: Optional[EvaluatorCfg] = None

    def __post_init__(self):
        self.mujoco.physics_dt = self.policy_dt / self.mujoco.decimation

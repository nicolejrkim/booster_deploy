"""Run a simulated Booster robot that speaks the real robot's ROS 2 interface.

Start this in one terminal, then run ``scripts/deploy.py --task <name>``
(without ``--mujoco``) in another: the deployment code path used on the real
robot (DDS state/command topics, Custom-mode RPC, prepare stage, remote
control) runs unchanged against MuJoCo.

    source /opt/ros/humble/setup.bash
    source <booster_ros2_ws>/install/setup.bash
    python scripts/sim_robot.py --robot k1 --viewer
"""
import argparse
import sys

sys.path.append(".")

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", choices=("k1", "t1", "t2"), default="k1")
parser.add_argument("--viewer", action="store_true", default=False,
                    help="open the MuJoCo viewer")
parser.add_argument("--physics-dt", type=float, default=None,
                    help="physics step in seconds (default: MJCF timestep)")
parser.add_argument("--state-rate", type=float, default=500.0,
                    help="/low_state publish rate in Hz")
parser.add_argument("--rtf", type=float, default=1.0,
                    help="real-time factor (1.0 = wall clock)")
parser.add_argument("--initial-mode", default="walking",
                    choices=("damping", "prepare", "walking", "custom"),
                    help="robot mode at start-up")
parser.add_argument("--mode-transition", type=float, default=1.0,
                    help="seconds to blend into the prepare pose when the "
                         "built-in stand controller takes over")
parser.add_argument("--log-states", type=str, default=None,
                    help="save time/qpos/qvel/ctrl/mode to this .npz")
parser.add_argument("--band", action="store_true", default=False,
                    help="start with the elastic band on: a slack rope that "
                         "catches the trunk when it drops below the anchor "
                         "height; toggle with the viewer key E or the "
                         "elastic_band service")
parser.add_argument("--band-height", type=float, default=None,
                    help="rope anchor height in m (default: spawn height, so "
                         "standing and walking are unaffected)")
parser.add_argument("--band-stiffness", type=float, default=2000.0)
parser.add_argument("--band-damping", type=float, default=100.0)
args = parser.parse_args()


def main():
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from booster_deploy.simulator import BoosterRobotSim

    if args.robot == "k1":
        from booster_deploy.robots import K1_CFG as robot_cfg
    elif args.robot == "t1":
        from booster_deploy.robots import T1_23DOF_CFG as robot_cfg
    else:
        from booster_deploy.robots import T2_31DOF_CFG as robot_cfg

    # The node handles SIGINT/SIGTERM itself (orderly thread shutdown).
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    sim = BoosterRobotSim(
        robot_cfg,
        physics_dt=args.physics_dt,
        state_rate_hz=args.state_rate,
        real_time_factor=args.rtf,
        initial_mode=args.initial_mode,
        mode_transition_s=args.mode_transition,
        log_states=args.log_states,
        elastic_band=args.band,
        band_height=args.band_height,
        band_stiffness=args.band_stiffness,
        band_damping=args.band_damping,
    )
    try:
        sim.run(viewer=args.viewer)
    finally:
        sim.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

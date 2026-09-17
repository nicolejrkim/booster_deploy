"""Watch a running deployment (real robot or simulated) in a MuJoCo viewer.

    source /opt/ros/humble/setup.bash
    source <booster_ros2_ws>/install/setup.bash
    python scripts/monitor.py --robot k1

Shows the robot posed from /low_state (encoders + IMU, feet on the floor)
and /odometer_state, a ghost at the /joint_ctrl targets and the deployment's
FSM state, and prints rates, tracking error, torque and tilt.
``--no-viewer`` prints only.
"""
import argparse
import os
import sys

sys.path.append(".")

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", choices=("k1", "t1", "t2"), default="k1")
parser.add_argument("--no-viewer", action="store_true", default=False,
                    help="terminal status line only (no display needed)")
parser.add_argument("--no-ghost", action="store_true", default=False,
                    help="do not draw the commanded-target ghost")
parser.add_argument("--log", type=str, default=None,
                    help="record the received stream to this .npz")
args = parser.parse_args()


def main():
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from booster_deploy.monitor import RobotMonitor

    if args.robot == "k1":
        from booster_deploy.robots import K1_CFG as robot_cfg
    elif args.robot == "t1":
        from booster_deploy.robots import T1_23DOF_CFG as robot_cfg
    else:
        from booster_deploy.robots import T2_31DOF_CFG as robot_cfg

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    monitor = RobotMonitor(robot_cfg, log_path=args.log)
    monitor.run(viewer=not args.no_viewer, show_ghost=not args.no_ghost)
    # run() has stopped its threads and saved the log.  ROS 2 / DDS teardown
    # occasionally hangs after a viewer session; there is nothing left to do.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

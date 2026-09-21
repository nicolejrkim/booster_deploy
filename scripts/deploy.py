import argparse
import os
import shlex
import signal
import subprocess
import sys

sys.path.append(".")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
# require either --task or --list (mutually exclusive)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--task", type=str, help="Name of the configuration file.")
group.add_argument("-l", "--list", action="store_true", dest="list_tasks",
                   default=False, help="list available tasks")

parser.add_argument("--mujoco", action="store_true", default=False,
                    help="deploy in mujoco simulation")
parser.add_argument("--sim", action="store_true", default=False,
                    help="start a simulated robot (scripts/sim_robot.py with "
                         "its viewer) and run the real-robot path against it")
parser.add_argument("--sim-args", type=str, default="",
                    help="extra arguments for scripts/sim_robot.py with "
                         "--sim; use the = form, e.g. "
                         "--sim-args=\"--rtf 0.5\"")
parser.add_argument("--monitor", action="store_true", default=False,
                    help="also start the live monitor (scripts/monitor.py) "
                         "for the task's robot; with --sim the simulator "
                         "then runs headless and the monitor is the window")
parser.add_argument("--monitor-args", type=str, default="",
                    help="extra arguments for scripts/monitor.py with "
                         "--monitor; use the = form, e.g. "
                         "--monitor-args=\"--log run1\"")
parser.add_argument(
    "--device", type=str, default="cpu",
    help="Device to run the evaluation on (e.g., 'cpu', 'cuda')")
parser.add_argument(
    "--record", type=str, default=None,
    help="with --mujoco: run headless (EGL) and write this mp4 of the "
         "simulated robot + reference ghost, start to end of the motion")
parser.add_argument("--record-fps", type=int, default=25)
parser.add_argument(
    "--no-fsm", action="store_true", default=False,
    help="run Booster's original flow instead of the state machine: X/x "
         "enters Custom mode (prepare per robot.prepare_mode), A/r starts "
         "the task policy, Ctrl+C hands back per exit_mode; no monitor "
         "topics or verdicts")
parser.add_argument(
    "--executor-ros-context", action="store_true", default=False,
    help="let the FSM executor child create its own ROS 2 node and "
         "/joint_ctrl publisher (implied by --sim and --monitor). Off by "
         "default: the child publishes through the inherited publisher, as "
         "Booster's own inference process does on the robot.")
parser.add_argument(
    "--exit-mode",
    choices=("walking", "damping"),
    default=None,
    help="Robot mode to enter after controller exit (default: task config, walking)",
)
args = parser.parse_args()


def robot_short_name(task_cfg):
    """The robot family (k1, t1, t2) the helper scripts take."""
    name = task_cfg.robot.name.lower()
    robot = next((r for r in ("k1", "t1", "t2") if r in name), None)
    if robot is None:
        raise RuntimeError(f"unknown robot family {task_cfg.robot.name!r}")
    return robot


def start_helper(label, script_name, script_args, log_name):
    """Run scripts/<script_name> as a subprocess in its own process group.

    The terminal's Ctrl+C then reaches only the deploy, which hands the
    robot back first and stops the helpers afterwards (stop_helper).
    """
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_name)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          script_name)
    process = subprocess.Popen(
        [sys.executable, script, *script_args],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        start_new_session=True)
    print(f"{label} started (pid {process.pid}, log {log_path})")
    return process


def stop_helper(label, process):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)
    print(f"{label} stopped")


def start_simulated_robot(task_cfg):
    """scripts/sim_robot.py for the task's robot.  Headless when the
    monitor provides the window (add --viewer to --sim-args for both)."""
    script_args = ["--robot", robot_short_name(task_cfg)]
    if not args.monitor:
        script_args.append("--viewer")
    script_args += shlex.split(args.sim_args)
    return start_helper("Simulated robot", "sim_robot.py", script_args,
                        "sim_robot.log")


def start_monitor(task_cfg):
    """scripts/monitor.py for the task's robot."""
    script_args = ["--robot", robot_short_name(task_cfg),
                   *shlex.split(args.monitor_args)]
    return start_helper("Monitor", "monitor.py", script_args, "monitor.log")


def missing_policy_files(task_cfg):
    """Policy and motion files a task needs that are not on disk.

    Paths are relative to the policy's task package, as the policies resolve
    them; the robot's locomotion policy (walking preparation / WALK) is
    checked too.
    """
    import sys as _sys
    cfgs = [task_cfg]
    robot_name = task_cfg.robot.name.lower()
    try:
        if "t2" in robot_name:
            from tasks.locomotion.robots.t2 import T2WalkTaskCfg as Walk
        elif "t1" in robot_name:
            from tasks.locomotion.robots.t1 import T1WalkControllerCfg1 as Walk
        elif "k1" in robot_name:
            from tasks.locomotion.robots.k1 import K1WalkTaskCfg as Walk
        else:
            Walk = None
        if Walk is not None:
            cfgs.append(Walk())
    except Exception:
        pass
    missing = []
    for cfg in cfgs:
        policy_cls = cfg.policy.constructor
        task_path = os.path.dirname(
            _sys.modules[policy_cls.__module__].__file__)
        for attr in ("checkpoint_path", "motion_path"):
            path = getattr(cfg.policy, attr, None)
            if not isinstance(path, str):
                continue
            full = (path if os.path.isabs(path)
                    else os.path.join(task_path, path))
            if not os.path.isfile(full):
                missing.append(os.path.relpath(full, REPO_ROOT))
    return missing


def main():
    # load task registry and dispatch
    import pkgutil
    import tasks as tasks_pkg

    # auto-import all submodules under tasks (recursive) so they can register themselves
    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        full_name = mod_info.name
        try:
            __import__(full_name)
        except Exception as e:
            raise e
    from booster_deploy.utils.registry import get_task, list_tasks

    if args.list_tasks:
        print("Available tasks:")
        for task_name, cfg in list_tasks().items():
            cls = type(cfg)
            full_cls = f"{cls.__module__}.{cls.__qualname__}"
            missing = missing_policy_files(cfg)
            note = f"\t(files missing: {len(missing)})" if missing else ""
            print(f"  {task_name}\t:\t{full_cls}{note}")
        sys.exit(0)

    try:
        task_cfg = get_task(args.task)
    except KeyError:
        print(f"Unknown task '{args.task}'. Available tasks: {list(list_tasks().keys())}")
        sys.exit(1)

    # Set device for policy
    task_cfg.policy.device = args.device
    if args.exit_mode is not None:
        task_cfg.booster.exit_mode = args.exit_mode
    missing = missing_policy_files(task_cfg)
    if missing:
        parser.error(
            f"task '{args.task}' cannot run, these files are not on disk "
            "(not committed, or not copied to this machine):\n  "
            + "\n  ".join(missing))
    if args.sim or args.monitor or args.executor_ros_context:
        # Workstation cases: the simulator and a monitor started after the
        # fork need the child's own publisher.  Left off on the robot.
        task_cfg.booster.executor_ros_context = True

    if args.sim and args.mujoco:
        parser.error("--sim and --mujoco are mutually exclusive")
    if args.monitor and args.mujoco:
        parser.error("--monitor needs the ROS 2 path (not --mujoco)")

    # decide how to run based on flags
    if args.mujoco:
        if args.record:
            os.environ.setdefault("MUJOCO_GL", "egl")
            task_cfg.mujoco.record = args.record
            task_cfg.mujoco.record_fps = args.record_fps
            if hasattr(task_cfg.policy, "stop_at_motion_end"):
                task_cfg.policy.stop_at_motion_end = True
        # run mujoco controller
        from booster_deploy.controllers.mujoco_controller import MujocoController

        MujocoController(task_cfg).run()
    else:
        if args.no_fsm:
            from booster_deploy.controllers.legacy_portal import (
                BoosterRobotPortal)
        else:
            from booster_deploy.controllers.booster_robot_controller import (
                BoosterRobotPortal)
        sim_process = start_simulated_robot(task_cfg) if args.sim else None
        monitor_process = start_monitor(task_cfg) if args.monitor else None
        try:
            with BoosterRobotPortal(task_cfg) as portal:
                portal.run()
        finally:
            stop_helper("Monitor", monitor_process)
            stop_helper("Simulated robot", sim_process)
        # The portal has handed the robot back and joined its threads and
        # processes.  Interpreter teardown of a forked ROS 2 / DDS process can
        # occasionally hang; do not let that leave a zombie deployment behind.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()

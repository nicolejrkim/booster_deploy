import argparse
import os
import shlex
import signal
import subprocess
import sys

sys.path.append(".")

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
                         "--sim-args=\"--band --rtf 0.5\"")
parser.add_argument(
    "--device", type=str, default="cpu",
    help="Device to run the evaluation on (e.g., 'cpu', 'cuda')")
parser.add_argument(
    "--exit-mode",
    choices=("walking", "damping"),
    default=None,
    help="Robot mode to enter after controller exit (default: task config, walking)",
)
args = parser.parse_args()


def start_simulated_robot(task_cfg):
    """Launch scripts/sim_robot.py for the task's robot as a subprocess."""
    name = task_cfg.robot.name.lower()
    robot = next((r for r in ("k1", "t1", "t2") if r in name), None)
    if robot is None:
        raise RuntimeError(f"no simulated robot for {task_cfg.robot.name!r}")
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "sim_robot.log")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "sim_robot.py")
    cmd = [sys.executable, script, "--robot", robot, "--viewer",
           *shlex.split(args.sim_args)]
    # Own process group: the terminal's Ctrl+C reaches only the deploy,
    # which hands the robot back first and then stops the simulator.
    process = subprocess.Popen(
        cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        start_new_session=True)
    print(f"Simulated robot started (pid {process.pid}, log {log_path})")
    return process


def stop_simulated_robot(process):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)
    print("Simulated robot stopped")


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
            print(f"  {task_name}\t:\t{full_cls}")
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

    if args.sim and args.mujoco:
        parser.error("--sim and --mujoco are mutually exclusive")

    # decide how to run based on flags
    if args.mujoco:
        # run mujoco controller
        from booster_deploy.controllers.mujoco_controller import MujocoController

        MujocoController(task_cfg).run()
    else:
        from booster_deploy.controllers.booster_robot_controller import BoosterRobotPortal
        sim_process = start_simulated_robot(task_cfg) if args.sim else None
        try:
            with BoosterRobotPortal(task_cfg) as portal:
                portal.run()
        finally:
            stop_simulated_robot(sim_process)
        # The portal has handed the robot back and joined its threads and
        # processes.  Interpreter teardown of a forked ROS 2 / DDS process can
        # occasionally hang; do not let that leave a zombie deployment behind.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()

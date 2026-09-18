#!/usr/bin/env python3
"""Hand-off between this repo and Booster's ``booster_train`` (Isaac Lab).

Two directions:

``motion``: a retargeted K1 motion CSV (Booster format: root xyz, quat xyzw,
22 joints in serial order, 50 Hz) becomes a training task in booster_train.
The CSV is copied to ``<booster_assets>/motions/K1``, converted to the
BeyondMimic ``.npz`` with booster_train's own ``scripts/csv_to_npz.py`` (needs
Isaac Sim), and a task package ``robots/k1/<name>`` is generated from the
``fight_001`` template, registered as ``Booster-K1-<Name>-v0``.

``deploy``: a trained ``model_<iter>.pt`` becomes a deployment task here.  The
checkpoint is exported with ``scripts/export_rsl_rl_policy.py`` into
``tasks/beyond_mimic/robots/k1/models``, the motion ``.npz`` is copied next to
it, and the ``register_booster_train_dance(...)`` line to add to
``tasks/beyond_mimic/robots/k1/__init__.py`` is printed.

Examples:

    python scripts/booster_train_pipeline.py --python <isaaclab python> \\
        motion --csv k1_dance_floss_marg_stmr.csv --name dance_floss_stmr
    python scripts/booster_train_pipeline.py deploy \\
        --run <booster_train>/logs/rsl_rl/k1_dance_floss_stmr/<run> \\
        --name dance_floss_stmr
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BOOSTER_TRAIN = os.path.join(os.path.dirname(REPO), "booster_train")
DEFAULT_BOOSTER_ASSETS = os.path.join(os.path.dirname(REPO), "booster_assets")
TEMPLATE_TASK = "fight_001"
K1_TASKS = ("source/booster_train/booster_train/tasks/manager_based/"
            "beyond_mimic/robots/k1")


def task_id(name: str) -> str:
    """``dance_floss_stmr`` -> ``Booster-K1-Dance_Floss_Stmr-v0``."""
    camel = "_".join(part.capitalize() for part in name.split("_"))
    return f"Booster-K1-{camel}-v0"


def cmd_motion(args: argparse.Namespace) -> int:
    name = args.name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise SystemExit("--name must be a lowercase python identifier")
    motions_dir = os.path.join(args.booster_assets, "motions", "K1")
    os.makedirs(motions_dir, exist_ok=True)
    csv_dst = os.path.join(motions_dir, f"k1_{name}.csv")
    npz_dst = os.path.join(motions_dir, f"k1_{name}.npz")
    if os.path.abspath(args.csv) != os.path.abspath(csv_dst):
        shutil.copy2(args.csv, csv_dst)
    print(f"[motion] csv -> {csv_dst}")

    convert = [
        args.python,
        os.path.join(args.booster_train, "scripts", "csv_to_npz.py"),
        "--input_file", csv_dst, "--input_fps", str(args.fps),
        "--output_fps", "50", "--output_name", npz_dst, "--headless",
    ]
    print("[motion] converting with booster_train (Isaac Sim)...")
    result = subprocess.run(convert, cwd=args.booster_train)
    if result.returncode != 0 or not os.path.isfile(npz_dst):
        raise SystemExit("csv_to_npz failed")
    print(f"[motion] npz -> {npz_dst}")

    tasks_dir = os.path.join(args.booster_train, K1_TASKS)
    src = os.path.join(tasks_dir, TEMPLATE_TASK)
    dst = os.path.join(tasks_dir, name)
    if os.path.isdir(dst):
        if not args.force:
            raise SystemExit(f"task package exists: {dst} (use --force)")
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    template_motion = re.compile(r"motions/K1/k1_[A-Za-z0-9_]+\.npz")
    for fname, edits in (
        ("env_cfg.py", [(template_motion, f"motions/K1/k1_{name}.npz")]),
        ("ppo_cfg.py", [(re.compile(r'experiment_name = "[^"]+"'),
                         f'experiment_name = "k1_{name}"')]),
        ("__init__.py", [(re.compile(r"Booster-K1-[A-Za-z0-9_]+-v0"),
                          task_id(name))]),
    ):
        path = os.path.join(dst, fname)
        text = open(path).read()
        for pattern, replacement in edits:
            text = pattern.sub(replacement, text)
        open(path, "w").write(text)
    print(f"[motion] task package -> {dst}")
    print("\nTrain with:\n"
          f"  cd {args.booster_train} && {args.python} scripts/rsl_rl/train.py "
          f"--task {task_id(name)} --headless --max_iterations 10000 "
          f"--run_name k1_{name}")
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    name = args.name
    run_dir = args.run
    if args.checkpoint is None:
        ckpts = sorted(
            glob.glob(os.path.join(run_dir, "model_*.pt")),
            key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
        if not ckpts:
            raise SystemExit(f"no model_*.pt in {run_dir}")
        checkpoint = ckpts[-1]
    else:
        checkpoint = args.checkpoint
    npz_src = os.path.join(args.booster_assets, "motions", "K1",
                           f"k1_{name}.npz")
    if not os.path.isfile(npz_src):
        raise SystemExit(f"motion not found: {npz_src}")

    task_dir = os.path.join(REPO, "tasks", "beyond_mimic", "robots", "k1")
    models_dir = os.path.join(task_dir, "models")
    motions_dir = os.path.join(task_dir, "motions")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(motions_dir, exist_ok=True)
    out_prefix = os.path.join(models_dir, f"k1_{name}_bt")
    export = [sys.executable,
              os.path.join(REPO, "scripts", "export_rsl_rl_policy.py"),
              "--checkpoint", checkpoint, "--output", out_prefix]
    print(f"[deploy] exporting {checkpoint}")
    if subprocess.run(export).returncode != 0:
        raise SystemExit("export failed")
    shutil.copy2(npz_src, os.path.join(motions_dir, f"k1_{name}.npz"))
    print(f"[deploy] motion -> {motions_dir}/k1_{name}.npz")
    print("\nAdd to tasks/beyond_mimic/robots/k1/__init__.py:\n"
          f'  register_booster_train_dance("k1_bt_{name}", "k1_{name}")')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--booster-train", default=DEFAULT_BOOSTER_TRAIN,
                        help="path to the booster_train checkout")
    parser.add_argument("--booster-assets", default=DEFAULT_BOOSTER_ASSETS,
                        help="path to the booster_assets checkout")
    parser.add_argument("--python", default=sys.executable,
                        help="python with Isaac Lab for booster_train scripts")
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("motion", help="CSV -> booster_train motion + task")
    m.add_argument("--csv", required=True, help="Booster-format K1 motion CSV")
    m.add_argument("--name", required=True,
                   help="task name, e.g. dance_floss_stmr")
    m.add_argument("--fps", type=int, default=50, help="CSV frame rate")
    m.add_argument("--force", action="store_true",
                   help="overwrite an existing task package")
    m.set_defaults(func=cmd_motion)
    d = sub.add_parser("deploy", help="booster_train run -> deploy task")
    d.add_argument("--run", required=True,
                   help="booster_train run directory (logs/rsl_rl/<exp>/<run>)")
    d.add_argument("--name", required=True, help="task name used for 'motion'")
    d.add_argument("--checkpoint", default=None,
                   help="model_<iter>.pt (default: highest iteration)")
    d.set_defaults(func=cmd_deploy)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

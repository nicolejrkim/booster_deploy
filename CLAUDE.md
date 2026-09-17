# Booster Deploy

Lightweight deployment harness for running RL control policies on Booster
humanoids (K1, T1, T2) either on the real robot (ROS 2 / DDS) or in MuJoCo
(sim2sim). Policies are plain TorchScript/ONNX files; all robot-specific
bookkeeping (joint order, gains, action scaling) lives in config objects.

## Layout

- `booster_deploy/controllers/` — `BaseController` (policy loop contract),
  `MujocoController` (sim2sim, PD torque control with a "ghost" reference
  robot), `BoosterRobotController` (real robot: `/low_state` subscriber,
  `/joint_ctrl` publisher, RPC mode switches, remote-control handling).
- `booster_deploy/robots/` — `RobotCfg` per robot (`K1_CFG`, `T1_23DOF_CFG`,
  `T2_31DOF_CFG`): joint/body names, default pose, PD gains, effort limits,
  MJCF path, prepare pose.
- `booster_deploy/utils/` — `policy_runner.py` (TorchScript vs ONNX picked by
  file suffix), `motion_loader.py` (BeyondMimic `.npz` motions), `registry.py`
  (task name -> `ControllerCfg`), vendored Isaac Lab math/configclass helpers.
- `tasks/<task>/` — one package per policy family: a `Policy`/`PolicyCfg`
  module plus `robots/<robot>/__init__.py` that builds `ControllerCfg`s and
  calls `register_task`. Checkpoints go in `robots/<robot>/models/`, motions in
  `robots/<robot>/motions/`; config paths are relative to the task package.
  - `locomotion/` — velocity-command walking (K1/T1/T2).
  - `beyond_mimic/` — BeyondMimic motion tracking (anchor-orientation obs).
  - `bm154/` — BeyondMimic with the BM154 hardware-style obs (K1 dances).
- `booster_deploy/fsm/` — deployment state machine (IDLE, STAND, WALK, TASK,
  ESTOP): `StateMachine` + transition table, and the child-process
  `FsmExecutor` that runs the active state's behaviour at 50 Hz.
- `booster_deploy/simulator/` — `BoosterRobotSim`, a MuJoCo ROS 2 node that
  emulates the robot firmware interface (`/low_state`, `/joint_ctrl`,
  `booster_rpc_service`) so the real-robot deploy path runs on a workstation.
- `scripts/deploy.py` — entry point; `scripts/sim_robot.py` — simulated robot
  for software-in-the-loop; `scripts/export_rsl_rl_policy.py` — RSL-RL
  checkpoint -> TorchScript + ONNX (folds in the obs normalizer).

## Commands

```bash
source .venv/bin/activate                 # see README for venv setup
python scripts/deploy.py --list           # registered tasks
python scripts/deploy.py --task <name> --mujoco   # sim2sim (needs booster_assets)
python scripts/deploy.py --task <name>            # real robot (needs ROS 2)
python scripts/sim_robot.py --robot k1 --viewer   # simulated robot for the line above
python scripts/export_rsl_rl_policy.py --checkpoint <model_N.pt> --output <prefix>
flake8                                    # max-line-length 80, see .flake8
```

`scripts/deploy.py` imports every module under `tasks/` so registration is a
side effect of import; a new task only needs its package and a
`register_task(...)` call.

## Conventions that matter

- **Two joint orders.** `RobotCfg.joint_names` is the real-robot/MuJoCo serial
  order (head, L arm, R arm, L leg, R leg). `RobotCfg.sim_joint_names` is the
  Isaac Lab training order (breadth-first; head/shoulder-pitch joints carry the
  `aa` prefix so they sort first). `robot.data.real2sim_joint_indexes` and
  `sim2real_joint_indexes` convert between them. Policy observations and raw
  actions are in sim order; joint targets sent to the robot are in real order.
- **Action mapping** is `target = default_joint_pos + action * scale` with
  `scale = 0.25 * effort_limit / stiffness` per joint (Isaac Lab
  `JointPositionAction` convention). Set `fixed_action_scale` explicitly when
  the deployed PD gains may differ from training.
- **Gains on the real robot**: `joint_stiffness`/`joint_damping` are sent
  directly as motor Kp/Kd. `effort_limit` is only used for MuJoCo torque
  clipping and the default action scale.
- **Motion files** are BeyondMimic `.npz` (`fps, joint_pos, joint_vel,
  body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w`) in sim joint/body
  order; `MotionLoader` re-indexes by name when the file carries names.
- **Assets are gitignored by default** (`tasks/**/models/*`,
  `tasks/**/motions/*`); release assets are whitelisted explicitly in
  `.gitignore`.
- **Policy rate** is `ControllerCfg.policy_dt` (0.02 s = 50 Hz by default);
  MuJoCo runs `policy_dt / mujoco.decimation` physics steps per policy step.
- **Software-in-the-loop** needs ROS 2 Humble plus `booster_interface` from
  Booster's `booster_robotics_sdk_ros2` built with colcon (its `Subtitle.msg`
  is malformed; drop it from the CMakeLists). On this workstation the
  workspace is `~/stmr/booster_ros2_ws`; `.venv` (conda-based Python 3.10)
  can import Humble's `rclpy` once both setup scripts are sourced.
- Real-robot flow is the FSM: X/`x` IDLE->STAND (posture check, primed hold,
  Custom RPC), A/`r` forward (STAND->WALK->TASK; `prepare_mode="standing"`
  skips WALK), Y/`n` back, B/`b` ESTOP (Damping RPC). The portal (main
  process) validates transitions and does RPCs; the executor child
  acknowledges via `fsm_active` and may request ESTOP (policy `stop()`) or
  `after_task` (policy `finish()`). Ctrl+C hands back per `exit_mode`.

## BM154 (tasks/bm154)

Observation (119 dims on K1, sim joint order): reference joint pos + vel,
reference-root projected gravity, robot projected gravity, gyro, joint pos -
default, joint vel, last action. Training source:
`whole_body_tracking` task `Tracking-Flat-K1-BM154-v0` (config in
`.../tasks/tracking/config/{g1,k1}/flat_env_cfg.py`). Checkpoints are RSL-RL
`model_N.pt` with `empirical_normalization=True`, so export them with
`scripts/export_rsl_rl_policy.py`, which bakes `(obs - mean) / (std + 1e-2)`
into the graph. `tasks/bm154/robots/k1/__init__.py` pins the training gains,
effort limits, default pose and action scale.

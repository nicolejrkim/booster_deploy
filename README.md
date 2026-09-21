# Booster Deploy

Booster Deploy is a lightweight deployment framework that supports running control policies on Booster robots (sim2real) and MuJoCo (sim2sim). The system adopts many well-established designs from IsaacLab to provide modular abstractions, allowing unified policy execution across simulated and real platforms.


## What's new in this branch

> **Added in this branch.** Everything in this section comes from the
> features added on the `feat/monitor-verdicts` branch on top of `main`.
> The detailed sections below that describe them carry the same notice.

- **`STAND` is the firmware's Prepare mode.** `IDLE` and `ESTOP` are
  Damping, `STAND` is Prepare (the firmware's standing controller) and
  `WALK` and `TASK` are Custom with our policies, so X and Y follow the same
  Damping -> Prepare sequence as Booster's remote and Custom mode is only
  entered for a policy. Before, `STAND` was a Python PD hold in Custom mode
  and `IDLE` was Walking mode. The deploy starts in the state matching the
  robot's mode and follows the firmware when it changes mode on its own.
  See [Deployment state machine](#deployment-state-machine).
- **Transition verdicts.** Every state-machine request, whether it comes
  from the keyboard, a gamepad or the `booster_deploy/fsm_request` topic,
  is answered by the deployment on `booster_deploy/fsm_result` with `OK`,
  `REJECTED`, `IGNORED` or `FAILED`, the target state and a reason. See
  [Deployment state machine](#deployment-state-machine).
- **Verdicts in the monitor.** The monitor's state-machine panel shows the
  deployment's answer to the request it just sent for a few seconds and
  logs it, and reports `NO RESPONSE` when nothing answers within 2 s.
  See [Live monitor](#live-monitor).
- **`deploy.py --monitor`.** The deployment can start the live monitor
  next to itself and stop it after the robot has been handed back
  (`--monitor-args=` passes options through, output in
  `logs/monitor.log`). Combined with `--sim`, the simulator runs headless
  and the monitor window is the only window, so one command brings up the
  simulator, the controller and the monitor. See
  [Software-in-the-loop](#software-in-the-loop-the-real-robot-path-against-a-simulated-robot).
- **Running from the workstation over an Ethernet cable.** `source
  scripts/robot_link.sh <adapter> [domain]` pins Fast DDS to the USB
  Ethernet adapter and `booster_link_check` confirms the robot's bridge is
  reachable; the deploy and the monitor can then run on the workstation
  the way crl-humanoid-ros drives a G1, or the monitor alone while the
  deploy runs on the robot. See
  [Running from the workstation over an Ethernet cable](#running-from-the-workstation-over-an-ethernet-cable).
- **Removed.** The simulator's elastic band (`--band`, the `elastic_band`
  service, key `E` in the simulator window and `B` in the monitor) and
  the monitor's translucent ghost at the `/joint_ctrl` targets
  (`--no-ghost`) are gone. The monitor shows the measured pose and the
  FSM label only.


## Prerequisites

| Environment | Notes |
|-------------|-------|
| Booster firmware >= v1.7.2 | Required for real robot deployments. |
| Python 3.10+ | Already installed on the robot |
| ROS 2 with `booster_interface` | Required for the DDS-backed `/low_state`, `/joint_ctrl`, and RPC interfaces. Already installed on the robot. |
| MuJoCo | Optional; install if you plan to run simulation locally. |


## Running Deployments

### Python environment

Create and activate a local virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Debian/Ubuntu, install `python3-venv` if needed. On the robot, activate
`.venv` before loading ROS 2; `booster_interface` is provided by the robot:

```bash
source .venv/bin/activate
source /opt/booster/BoosterRos2Interface/install/setup.bash
```

### Add and list tasks:
   1. Create a subfolder under `tasks/` for your task.
   2. Implement a `Policy`/`PolicyCfg` and provide a `ControllerCfg` referencing the policy.
   3. Place policy checkpoints under `models/` and reference the path in the config.
   4. Register your `ControllerCfg` config in the task registry (see existing tasks for the registration pattern).
   5. Check all available tasks:
      ```bash
      python scripts/deploy.py --list
      ```

### Policy inference backends

The checkpoint suffix selects the inference backend automatically:

- `.pt`, `.jit`, `.torchscript`: TorchScript
- `.onnx`: ONNX Runtime with the CPU execution provider
make sure `onnxruntime` is installed in the deployment environment.

### K1 deployment policies

Every registered K1 task is a `ControllerCfg` that replaces the `K1_CFG`
gains with the ones its policy was trained or tuned with, so `Kp`/`Kd` are a
property of the task, not of the robot. In Custom mode `robot.joint_stiffness`
and `robot.joint_damping` are sent to the motors unchanged as `kp`/`kd`
(see [PD damping on the real robot](#pd-damping-kd-on-the-real-robot) before
retuning `Kd`). All K1 tasks run at 50 Hz with `prepare_mode="walking"`, so
they are reached through `IDLE -> STAND -> WALK -> TASK`.

| Task | Policy / observation | Checkpoint (`tasks/<pkg>/robots/k1/models/`) | Motion (`.../motions/`) |
|------|----------------------|-----------------------------------------------|-------------------------|
| `k1_walk` | `locomotion`: velocity-command walking, 20 policy joints (the head holds its default pose); also the `WALK` state of every K1 task | `k1_walk.pt` | – |
| `k1_mj2` | `beyond_mimic`: anchor-orientation motion tracking (Booster's own K1 dance) | `k1_mj_dance_002_2025-12-03_00-10-28.pt` | `k1_mj2_seg1.npz` |
| `k1_fight` | `beyond_mimic`, as above | `k1_fight_001.pt` | `k1_fight_final_deploy.npz` |
| `k1_bm154_jamesbrown`, `k1_bm154_floss`, `k1_bm154_boogle` | `bm154`: BeyondMimic with the 119-dim hardware-style observation (see [below](#bm154-motion-tracking-k1)) | `k1_dance_<name>_marg_bm154.pt` (`.onnx` next to it) | `k1_dance_<name>_marg_stmr.npz` |
| `k1_bt_<name>` (27 tasks: `k1_bt_dance_{jamesbrown,floss,boogle}_stmr`, 17 showcase clips such as `k1_bt_hiphop_floss_a316`, 7 seedpicks clips `k1_bt_sp_*`; `BOOSTER_TRAIN_TASKS` in `tasks/beyond_mimic/robots/k1/__init__.py` has the list) | `beyond_mimic` observation with the booster_train actuator-model gains (see [below](#training-with-boosters-booster_train-and-deploying-here)) | `<motion>_bt.pt` | `<motion>.npz` |
| `k1_bt2_<name>` (7 tasks, `BOOSTER_TRAIN_MJ2_TASKS`: the two STMR dances `jamesbrown`/`floss`, `high_jump_a277`, `ib_dodge_270_a437`, `turn_jump_0045_a023`, `sp_high_jump_a277`, `sp_jump_sideway_090_a024`) | the same, retrained in booster_train with the `k1_mj2` gains (`BOOSTER_K1_MJ2_CFG`) | `<motion>_mj2_bt.pt` | `<motion>.npz` |

`Kp / Kd` sent per joint, left and right identical (`K1_CFG` is what a new
task inherits; "Prepare hold" is `K1_CFG.prepare_state`, published once when
Custom mode is entered until the policy's first action, and its `Kd` is the
damping the executor publishes on `ESTOP` before the Damping RPC):

| Joint | `K1_CFG` | Prepare hold | `k1_walk` | `k1_mj2` | `k1_fight` | `k1_bm154_*` | `k1_bt_*` | `k1_bt2_*` |
|-------|----------|--------------|-----------|----------|------------|--------------|-----------|--------------|
| head yaw, head pitch | 4 / 1 | 40 / 1.5 | 4 / 1 | 10 / 2 | 10 / 2 | 8 / 0.4 | 3.95 / 0.25 | 10 / 2 |
| shoulder pitch | 4 / 1 | 40 / 0.5 | 20 / 2 | 4 / 1 | 3.95 / 0.3 | 15 / 0.5 | 3.95 / 0.25 | 4 / 1 |
| shoulder roll | 4 / 1 | 50 / 1.5 | 20 / 2 | 4 / 1 | 3.95 / 0.3 | 15 / 0.5 | 3.95 / 0.25 | 4 / 1 |
| elbow pitch | 4 / 1 | 20 / 0.2 | 20 / 2 | 4 / 1 | 3.95 / 0.3 | 15 / 0.5 | 3.95 / 0.25 | 4 / 1 |
| elbow yaw | 4 / 1 | 20 / 0.2 | 20 / 2 | 4 / 1 | 3.95 / 0.3 | 15 / 0.5 | 3.95 / 0.25 | 4 / 1 |
| hip pitch | 80 / 2 | 350 / 7.5 | 100 / 2 | 80 / 2 | 80 / 2 | 100 / 2 | 30.2 / 3.61 | 80 / 2 |
| hip roll | 80 / 2 | 350 / 7.5 | 100 / 2 | 80 / 2 | 80 / 2 | 100 / 2 | 21.4 / 2.56 | 80 / 2 |
| hip yaw | 80 / 2 | 180 / 3 | 100 / 2 | 80 / 2 | 80 / 2 | 100 / 2 | 17.8 / 2.13 | 80 / 2 |
| knee | 80 / 2 | 350 / 5.5 | 100 / 2 | 80 / 2 | 80 / 2 | 100 / 2 | 60.4 / 4.81 | 80 / 2 |
| ankle pitch | 30 / 2 | 250 / 5 | 65 / 1 | 30 / 2 | 30 / 2 | 50 / 1 | 35.7 / 4.26 | 30 / 2 |
| ankle roll | 30 / 2 | 250 / 5 | 65 / 1 | 30 / 2 | 30 / 2 | 50 / 1 | 35.7 / 4.26 | 30 / 2 |

The joint target is `default_joint_pos + action * scale`. `effort_limit`
only clips the MuJoCo torque and sets the default `scale =
0.25 * effort_limit / Kp`, but that default ties the action mapping to `Kp`:

- `k1_walk` uses a flat `scale = 0.25` and a crouched default pose
  (hip pitch -0.15, knee 0.3, ankle pitch -0.15, shoulder pitch 0.2, elbow
  yaw ±0.5). Effort limits
  6 (head), 14 (arms), 30 / 20 / 15 / 35 / 24 / 15 (hip pitch, hip roll,
  hip yaw, knee, ankle pitch, ankle roll).
- `k1_mj2` and `k1_fight` derive `scale` from the gains above at start-up
  (effort limits 6 / 14 / 30, 35, 20, 40, 20, 20 for `k1_mj2`; 4 / 12 / same
  legs for `k1_fight`), so changing their `Kp` also changes what the policy
  commands. Set `policy.fixed_action_scale` to the current values first if
  you only want to retune the motors.
- `k1_bm154_*` pins the training default pose (hip pitch -0.2, knee 0.4,
  ankle pitch -0.25, shoulder pitch 0.2, elbow yaw ±0.5), the official
  effort limits (6 / 14 / 68, 43, 38.3, 112, 38.3, 38.3) and
  `fixed_action_scale`, so the mapping stays the training one whatever
  `Kp`/`Kd` are set to.
- `k1_bt_*` pins `fixed_action_scale` the same way; its gains are the
  booster_train actuator model (`Kp = J (2 pi f)^2`, `Kd = 2 zeta J 2 pi f`,
  4 Hz legs, 10 Hz arms and head) with effort limits
  6 / 14 / 68, 76, 38.3, 112, 38.3, 38.3.
- `k1_bt2_*` is the same actuator model retrained with the `k1_mj2` gains
  (`BOOSTER_K1_MJ2_CFG` in our booster_train branch), so the motors get the
  same Kp/Kd as Booster's own dances while the policy saw booster_train's
  delays and torque limits. Same effort limits as `k1_bt_*`, hence a
  different pinned action scale than `k1_mj2` (for example 0.35 on the knee
  instead of 0.125).

The values come from `booster_deploy/robots/k1.py`,
`tasks/locomotion/robots/k1/__init__.py`,
`tasks/beyond_mimic/robots/k1/__init__.py` and
`tasks/bm154/robots/k1/__init__.py`; those files are authoritative if this
table drifts.

### BM154 motion tracking (K1)

`tasks/bm154/` deploys BeyondMimic policies trained with the BM154
observation (`Tracking-Flat-K1-BM154-v0` in `whole_body_tracking`): reference
joint positions/velocities, the reference root's projected gravity, the robot's
projected gravity, the gyro, joint positions relative to the default pose,
joint velocities and the previous action (119 dims on the 22-DOF K1). All terms
are directly measurable on the robot, so no anchor position or base velocity
estimate is needed.

Registered K1 tasks: `k1_bm154_jamesbrown`, `k1_bm154_floss`,
`k1_bm154_boogle`. The controller configuration pins the training PD gains,
effort limits, default pose and per-joint action scale so the action mapping
matches training; tune the real-robot `Kd` per the note above if needed.

```bash
python scripts/deploy.py --task k1_bm154_jamesbrown --mujoco
```

#### Exporting an RSL-RL checkpoint

BM154 runs are trained with `empirical_normalization=True`, so the raw
`model_<iter>.pt` cannot be loaded directly. `scripts/export_rsl_rl_policy.py`
rebuilds the actor MLP, folds the observation normalizer in front of it and
writes both a TorchScript (`.pt`) and an ONNX (`.onnx`) file:

```bash
python scripts/export_rsl_rl_policy.py \
    --checkpoint <whole_body_tracking>/logs/rsl_rl/k1_flat/<run>/model_9999.pt \
    --output tasks/bm154/robots/k1/models/<name>
```

The MLP layout is inferred from the checkpoint; the activation and the
normalization flag are read from `params/agent.yaml` next to it when present.
Copy the matching motion `.npz` into `tasks/bm154/robots/k1/motions/` and add a
`ControllerCfg` in `tasks/bm154/robots/k1/__init__.py` that points to both.

### Training with Booster's booster_train and deploying here

[booster_train](https://github.com/BoosterRobotics/booster_train) is Booster's
Isaac Lab port of BeyondMimic, the code their own K1 dances (`k1_fight`,
`k1_mj2`) were trained with: delayed PD actuators with torque-speed curves,
extra foot/hand/trunk pose rewards, the estimator-free anchor-orientation
observation. `scripts/booster_train_pipeline.py` moves motions in and
policies out:

```bash
# 1. retargeted K1 CSV (Booster format, 50 Hz) -> booster_train motion + task
python scripts/booster_train_pipeline.py --python <isaaclab python> motion \
    --csv k1_dance_floss_marg_stmr.csv --name dance_floss_stmr
# 2. train in booster_train (command printed by step 1)
# 3. trained run -> exported policy + motion in tasks/beyond_mimic/robots/k1
python scripts/booster_train_pipeline.py deploy \
    --run <booster_train>/logs/rsl_rl/k1_dance_floss_stmr/<run> --name dance_floss_stmr
```

Step 3 exports the run's latest checkpoint as `models/<motion>_bt.pt` and
copies `motions/<motion>.npz`, where `<motion>` is the motion file recorded in
the run's `params/env.yaml` (override with `--motion`). It prints the
`register_booster_train_dance("k1_bt_<name>", "<motion>")` line; add it, or
an entry, to `BOOSTER_TRAIN_TASKS` in `tasks/beyond_mimic/robots/k1/__init__.py`.
Runs trained on the `-Mj2-v0` task variants (the `k1_mj2` gains) are exported
with `--gains mj2`, which writes `models/<motion>_mj2_bt.pt` and prints the
`K1BoosterTrainMj2ControllerCfg` registration for `BOOSTER_TRAIN_MJ2_TASKS`.
The task then uses the `BeyondMimicPolicy` observation with the booster_train
actuator-model gains (see [K1 deployment policies](#k1-deployment-policies)). Our booster_train checkout carries a branch that renames its K1 config to
the official `booster_assets` names and adds an Isaac Lab 2.1 shim.

### Run Sim2Sim (MuJoCo)

- Download and install BoosterAssets:
   - Clone the [booster_assets](https://github.com/BoosterRobotics/booster_assets) which contains Booster robot models and resources.
   - Install booster_assets python helper following the instructions in the repository.

- Install Python dependencies in the activated virtual environment:
   ```
   python -m pip install -r requirements.txt
   ```

- Launch the task in mujoco:
   ```bash
   python scripts/deploy.py --task <TASK_NAME> --mujoco
   ```

### Software-in-the-loop: the real-robot path against a simulated robot

`scripts/sim_robot.py` runs a MuJoCo robot that speaks the robot firmware's
ROS 2 interface: it publishes `/low_state`, consumes `/joint_ctrl` and serves
`booster_rpc_service` for the `ChangeMode`/`GetStatus` calls. With it, the
exact code path used on the real robot (DDS topics, Custom-mode switch,
prepare stage, remote-control handling, inference process, exit mode) runs
unchanged on a workstation.

Prerequisites: ROS 2 Humble and the `booster_interface` message package from
[booster_robotics_sdk_ros2](https://github.com/BoosterRobotics/booster_robotics_sdk_ros2)
built in a colcon workspace (drop `msg/Subtitle.msg` from its `CMakeLists.txt`;
it is not a valid ROS message). Both `rclpy` and `booster_interface` must be
importable from the Python environment that runs the scripts.

```bash
# terminal 1: simulated robot
source /opt/ros/humble/setup.bash
source <booster_ros2_ws>/install/setup.bash
python scripts/sim_robot.py --robot k1 --viewer

# terminal 2: the real-robot entry point, no --mujoco
source /opt/ros/humble/setup.bash
source <booster_ros2_ws>/install/setup.bash
python scripts/deploy.py --task k1_bm154_jamesbrown
```

Or in one terminal, letting the deploy start and stop the simulated robot
itself (`--sim-args=` passes options through; the `=` form is needed because
the values start with dashes):

```bash
python scripts/deploy.py --task k1_bm154_jamesbrown --sim --sim-args="--rtf 0.5"
```

> **Added in this branch.** `--monitor` and `--monitor-args=` are among the
> features added on this branch (see
> [What's new in this branch](#whats-new-in-this-branch)).

Add `--monitor` to also start the live monitor (below) from the same command,
so one terminal brings up the simulator, the controller and the monitor. The
simulator then runs headless and the monitor window is the view; append
`--viewer` to `--sim-args` to keep the simulator's own window as well.
`--monitor-args=` passes options to the monitor:

```bash
python scripts/deploy.py --task k1_bm154_jamesbrown --sim --monitor
```

Then drive the state machine from terminal 2 exactly as on the robot (`x`,
`r`, `n`, `b`), or from the monitor window (below). The simulated robot starts
in Walking mode holding the prepare pose with the prepare gains (the deploy
starts in `IDLE`); `--initial-mode damping` starts it limp on the floor like
a robot that has just booted, so `x` goes through the get-up. Prepare and
Walking mode are the same standing hold: Booster's built-in locomotion
controller is not emulated. Damping mode only damps the joints; Custom mode
applies the latest
`/joint_ctrl` command, and commands received before the switch are retained
as on the robot. Useful options: `--state-rate` (default
500 Hz), `--physics-dt`, `--rtf` to slow the simulation down, and
`--log-states <file>` to record `time/qpos/qvel/ctrl/mode` for offline checks.
Get-up is not simulated: the get-up RPC puts the robot back upright in the
prepare pose. The simulator also publishes ground-truth odometry on
`/odometer_state` and the normal contact force under each foot on
`booster_sim/contact_forces`, and warns (once per second) when a commanded
PD torque exceeds a joint's limit before clamping it, as crl-humanoid-ros'
simulator does. The deploy prints the same warning on the real robot from
the torque the firmware will compute for each command.

### Live monitor

> **Added in this branch.** Starting the monitor from `deploy.py --monitor`
> and the transition verdicts shown in the state-machine panel are among
> the features added on this branch (see
> [What's new in this branch](#whats-new-in-this-branch)). The elastic-band
> key `B` and the commanded-target ghost (`--no-ghost`) were removed.

`scripts/monitor.py` watches a running deployment, real or simulated, from
any machine on the same ROS 2 domain:

```bash
python scripts/monitor.py --robot k1            # viewer + status line
python scripts/monitor.py --robot k1 --no-viewer  # status line only (SSH)
python scripts/monitor.py --robot k1 --log run1   # also record the stream
```

`deploy.py --monitor` starts it next to the deployment on the same machine
(its output goes to `logs/monitor.log`; `--monitor-args="--log run1"` passes
options through) and stops it after the robot has been handed back.

It poses the model from `/low_state` (encoders and IMU, feet kept on the
floor) and places it with `/odometer_state` (`booster_interface/Odometer`,
published by the robot firmware and by the simulator; without it the robot
stays at the origin and steps in place), draws a floating label with the
FSM state, and prints the
topic rates, the largest joint tracking error, the largest torque relative to
the effort limit (flagged `TORQUE LIMIT` at 95 %), joints outside their angle
range, the trunk tilt and, against the simulator, the foot contact forces.
Keys in the monitor window:

| Key | Action |
|-----|--------|
| `M` | show/hide the state-machine panel |
| `Up`/`Down`, `Enter` | select a state in the panel and request the transition; the deployment's verdict appears under the list |
| `N` | show/hide the status text |
| `V` | camera follows the robot on/off |

Every request is answered by the deployment on `booster_deploy/fsm_result`
(`OK STAND`, `REJECTED TASK: not allowed from IDLE`, `IGNORED STAND: already
the current state`, `FAILED STAND: unsafe posture, switched to Damping`). The
panel shows the verdict for a few seconds and the monitor logs it; a request
that nothing answers within 2 s is reported as `NO RESPONSE`, which means no
`deploy.py` is listening.

### Run Sim2Real (Real Robots)

**IMPORTANT**: Make sure to install [Booster Firmware](https://booster.feishu.cn/wiki/E3q5wF5SnitXZgkY18Uc8odBnXb) >= v1.4 on the robot before proceeding.

- After you finish testing your task with Sim2Sim locally, copy the project to the robot.

- Install Python dependencies in the activated virtual environment on the robot:
   ```
   python -m pip install -r requirements.txt
   ```

- SSH into the robot and start the ROS 2 environment by sourcing the provided setup script:
   ```bash
   source /opt/booster/BoosterRos2Interface/install/setup.bash
   ```

- Launch the task on the robot and follow the prompts shown in the command line..
   ```bash
   python scripts/deploy.py --task <TASK_NAME>
   ```

#### Running without the state machine (`--no-fsm`)

> **Added in this branch.** The state machine is the branch's own; this flag
> restores what `main` does.

`python scripts/deploy.py --task <TASK_NAME> --no-fsm` runs Booster's
original flow, kept verbatim from upstream in
`booster_deploy/controllers/legacy_portal.py`: `X` (keyboard `x`) enters
Custom mode with the prepare stage selected by `robot.prepare_mode`
(`"walking"`: the locomotion policy holds the robot with zero velocity
until `A`; `"standing"`: a one-second move to `prepare_state.joint_pos`),
`A` (keyboard `r`) starts the task policy, and `Ctrl+C` hands the robot
back per `booster.exit_mode`. There is no `STAND`/Prepare state, no get-up,
no `booster_deploy/fsm_*` topics and no verdicts, so the monitor shows the
pose but no state. It works with `--sim` too.

#### Running from the workstation over an Ethernet cable

> **Added in this branch.** `scripts/robot_link.sh`, `booster_link_check`
> and this way of running are among the features added on this branch (see
> [What's new in this branch](#whats-new-in-this-branch)).

The deploy does not have to run on the robot. With the workstation plugged
into the robot through a USB Ethernet adapter, the real-robot code path can
run here and reach the robot's ROS 2 bridge over the cable, the way
crl-humanoid-ros (the CRL lab's GitLab, `crl/crl-humanoid-ros`) drives
a Unitree G1 from a laptop: its hardware node, state machine and monitor all
run on the laptop, bound to the adapter, and only the low-level loop crosses
the cable. Both ways of running below use the same link setup; pick per
session.

| | Everything on the workstation ("crl way") | Deploy on the robot, monitor here (the sequence above) |
|---|---|---|
| `deploy.py` runs on | the workstation | the robot (SSH) |
| `monitor.py` runs on | the workstation, talks to the deploy on this host | the workstation, talks to the deploy over the cable |
| Crosses the cable | `/low_state` 500 Hz, `/joint_ctrl` 50 Hz, the mode-switch RPC, odometry | `/low_state`, `/joint_ctrl` and odometry for display, `booster_deploy/fsm_*` requests and verdicts |
| A cable fault costs | the control loop (see [Failure behaviour](#failure-behaviour-and-safety)) | the picture and the monitor's buttons; the controller is unaffected |
| Files on the robot | none; policies and motions stay here | the repo with the task's checkpoint and motion |
| Use it for | iterating on policies, workstation inference, no copying | anything where the loop must not depend on the cable |

##### Prerequisites

- ROS 2 Humble and the `booster_interface` workspace on the workstation, as
  for [software-in-the-loop](#software-in-the-loop-the-real-robot-path-against-a-simulated-robot);
  `scripts/robot_link.sh` sources `~/stmr/booster_ros2_ws` by default
  (override with `BOOSTER_ROS2_WS`).
- Fast DDS on both sides. It is the Humble default and what Booster's ROS 2
  image ships; leave `RMW_IMPLEMENTATION` alone on the robot. Two different
  middlewares never discover each other.
- The robot's ROS domain. The robot's bridge publishes on whatever
  `ROS_DOMAIN_ID` its setup script sets, 0 if none; check with
  `env | grep ROS` in an SSH session on the robot. The link script takes it
  as its second argument.
- A USB Ethernet adapter with a static IPv4 address on the robot's subnet.
  The adapter's interface is named after its MAC (`enx...`); `ip -brief
  addr` lists it once it is plugged in. For example, with the robot at
  `192.168.10.102`:

  ```bash
  sudo ip addr add 192.168.10.5/24 dev enx00e04c680123
  sudo ip link set enx00e04c680123 up
  ping -c 2 192.168.10.102
  ```

  (or the same through NetworkManager: `nmcli con add type ethernet ifname
  enx00e04c680123 ip4 192.168.10.5/24`).

##### Step 1: source the link

```bash
cd ~/stmr/booster_deploy
source scripts/robot_link.sh enx00e04c680123 0      # interface name or its IPv4, then the robot's domain
```

Sourcing (not running) the script in the shell you will work from does, in
order:

1. resolves the interface name to its IPv4 address and refuses if it has
   none, which usually means the adapter is unplugged or unconfigured;
2. sources `/opt/ros/humble/setup.bash`, the `booster_interface` workspace
   and `.venv/bin/activate`;
3. writes `logs/fastdds_link.xml`, a Fast DDS profile with a single UDPv4
   transport whose interface whitelist is that one address and with the
   built-in transports disabled, and exports
   `FASTRTPS_DEFAULT_PROFILES_FILE` so every ROS 2 node started from this
   shell uses it;
4. exports `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and `ROS_DOMAIN_ID`, and
   unsets `ROS_LOCALHOST_ONLY`;
5. defines the `booster_link_check` shell function.

The pinning matters because a workstation typically has several interfaces
(WiFi, a VPN, Docker and libvirt bridges). Fast DDS advertises on all of
them by default, and discovery or traffic can end up on the wrong one, which
shows as a link that works intermittently. With the profile, nothing leaves
the adapter. The generated file looks like this:

```xml
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>link_udp</transport_id>
            <type>UDPv4</type>
            <interfaceWhiteList>
                <address>192.168.10.5</address>
            </interfaceWhiteList>
        </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="link_participant" is_default_profile="true">
        <rtps>
            <userTransports><transport_id>link_udp</transport_id></userTransports>
            <useBuiltinTransports>false</useBuiltinTransports>
        </rtps>
    </participant>
</profiles>
```

Note the spelling `interfaceWhiteList`, capital L. Fast DDS 2.6 rejects
`interfaceWhitelist` with `XMLPARSER Error` lines at node start-up and then
runs unpinned, so a wrong profile fails open rather than closed. Everything
the script sets is per shell: open a new terminal to get the normal
environment back.

##### Step 2: check the link

```bash
booster_link_check
```

listens for five seconds and expects two things: `/low_state` at about
500 Hz and the `booster_rpc_service` service. Success looks like

```text
domain 0, interface 192.168.10.5; listening 5 s for /low_state ...
  /low_state: average rate: 499.995
  booster_rpc_service: available
link ok
```

`nothing received` means DDS discovery did not happen: the cable, the
adapter's address, the subnet, the domain or the middleware. Fix that before
starting anything; nothing below can work without it. Topics visible but no
RPC service means the robot's bridge is only partly up.

##### Step 3a: everything on the workstation

```bash
python scripts/deploy.py --task k1_bt_dance_floss_stmr --monitor
```

This is the same command as against the simulated robot minus `--sim`: the
deploy subscribes to the robot's `/low_state`, publishes `/joint_ctrl`, does
the mode switches through the RPC service, and starts `scripts/monitor.py`
next to itself (output in `logs/monitor.log`; `--monitor-args=` passes
options through, `--exit-mode damping` overrides the task's exit mode). The
state machine prints its state and the keys that act on it; drive it from
this terminal (`x`, `r`, `n`, `b`), from the monitor's FSM panel or from the
gamepad, exactly as described under
[Deployment state machine](#deployment-state-machine). Ctrl+C hands the
robot back per the exit mode and stops the monitor.

Never run this while a deploy is also running on the robot: two publishers
on `/joint_ctrl` fight over the motors.

##### Step 3b: deploy on the robot, monitor here

Start the deploy on the robot over SSH as in the sequence above, then in the
sourced shell here:

```bash
python scripts/monitor.py --robot k1
```

The monitor shows the measured pose and the deployment's state, and its FSM
panel requests transitions on `booster_deploy/fsm_request`; the deploy on
the robot validates each request like a key press, performs the RPCs and
answers on `booster_deploy/fsm_result`, which the panel shows for a few
seconds (`NO RESPONSE` after two seconds means the request never reached a
deploy). The control loop never touches the cable.

##### Failure behaviour and safety

With everything on the workstation, the 50 Hz loop depends on the cable and
on DDS. If `/low_state` stops arriving, the deploy has no staleness guard: the
executor keeps stepping the policy on the last state it received and keeps
publishing `/joint_ctrl`, and the mode watchdog, whose `GetStatus` call then
fails, simply skips that check. What the firmware does when commands stop
reaching it in Custom mode is Booster's behaviour, not ours. So for this way:

- keep the physical remote in hand; its B button is the emergency stop that
  does not depend on the cable. The monitor's ESTOP is a network message;
- use a short, directly attached cable and the pinned profile, not a switch
  shared with other traffic;
- prefer the deploy-on-robot way for anything that matters more than
  convenience. That is why both are documented.

In the deploy-on-robot way a cable fault costs only the monitor: its status
line stops updating and its buttons get `NO RESPONSE`, while the deploy on
the robot continues with its keyboard and gamepad.

##### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `booster_link_check`: `nothing received` | Discovery failed. `ping` the robot; check `ip -4 addr show dev <adapter>` has an address on the robot's subnet; match `ROS_DOMAIN_ID` to the robot's; make sure `RMW_IMPLEMENTATION` is unset or Fast DDS on the robot; check the robot's setup does not export `ROS_LOCALHOST_ONLY=1`. |
| `XMLPARSER Error` lines when a node starts | The profile did not parse; the node runs unpinned. Regenerate it by sourcing the script again, and keep the `interfaceWhiteList` spelling if you edit it by hand. |
| Topics listed but `booster_rpc_service` missing | The bridge on the robot is not fully up. Restart it on the robot, then re-check. |
| Link works, then flaps | Another interface took part in discovery. Confirm the profile is in effect: `echo $FASTRTPS_DEFAULT_PROFILES_FILE`, and start every node from the sourced shell. |
| Monitor shows `NO RESPONSE` | No deploy answered on this domain: it is not running, runs on another domain, or (deploy-on-robot way) the cable is down. |
| Deploy prints `Waiting for first '/low_state' message` forever | Same as the first row; the deploy waits and never starts publishing without state. |
| Monitor window empty, only a status line | `--no-viewer`, or no display. The status line carries the same information. |

##### How this was verified

`logs/sil_check/run_link_sil.sh` (gitignored, on the workstation) runs the
whole crl way against the simulated robot with the link pinned to a real
interface of this machine: sources the script, starts `sim_robot.py`, runs
`booster_link_check`, then `deploy.py --task k1_bt_dance_floss_stmr
--monitor` and drives `IDLE -> STAND -> WALK -> TASK` through the remote
control path. With the profile pinned to an address the machine does not
have, the same check finds nothing, which shows the pin is in effect and not
just declared. It has not yet been run against the real robot; the robot's
domain and subnet are the two things to confirm on the day.

#### PD damping (`Kd`) on the real robot

For parallel-actuated joints, `robot.joint_damping` is sent directly to the
motors, so do not reuse the training-simulator `Kd`. Compute the motor-side
value as:

```text
Kd = 2 * zeta * J_eq * (2 * pi * f_n)
```

where `J_eq` is the armature of the parallel-actuated joint, `f_n` is the
natural frequency, and `zeta` is the damping ratio.


#### Controller exit mode

`booster.exit_mode` controls the robot mode entered after the custom
controller exits. It applies to robots whose firmware supports the
corresponding DDS RPC mode-switch API:

- `"damping"`: switch to damping mode
- `"walking"`: switch to walking mode (default)

The value can be set in the task controller configuration, for example:

```python
booster = BoosterRobotControllerCfg(exit_mode="damping")
```

You can override the task configuration at startup when damping is preferred:

```bash
python3 scripts/deploy.py --task <TASK_NAME> --exit-mode damping
```

The value `"walk"` is also accepted as an alias for `"walking"` in Python configuration.


#### Deployment state machine

On the real robot, `deploy.py` runs a finite state machine
(`booster_deploy/fsm`). The states outside Custom mode are the firmware's
own modes, so X and Y follow the same Damping -> Prepare sequence as
Booster's remote; Custom mode is entered only for our policies. Only the
listed transitions are accepted; anything else is refused and logged. The
current state and the buttons that act on it are printed at every change.

| State | Robot mode | What runs |
|-------|------------|-----------|
| `IDLE` | Damping (or Walking after a get-up / `exit_mode="walking"`) | the firmware; nothing published; start state |
| `STAND` | Prepare | the firmware's standing controller |
| `WALK` | Custom | the robot's locomotion policy (`tasks/locomotion`) with stick commands |
| `TASK` | Custom | the policy selected with `--task` |
| `ESTOP` | Damping | robot goes limp after an abort; nothing published |

```text
IDLE --X--> STAND --A--> WALK --A--> TASK        Y steps back one state
any --B--> ESTOP --Y--> IDLE                      Ctrl+C: exit_mode, then quit
```

- The deploy starts in the state matching the robot's mode (Damping or
  Walking -> `IDLE`, Prepare -> `STAND`; Custom under another controller ->
  `IDLE`, publishing nothing).
- `robot.prepare_mode` decides where `A` goes from `STAND`: `"walking"`
  (default) inserts the `WALK` state (`STAND -> WALK -> TASK`), `"standing"`
  goes straight to `TASK`. `WALK` exists only when the robot has a
  locomotion task (`k1_walk`, `t1_walk`, `t2_walk`) and the task launched is
  not that policy itself.
- `IDLE -> STAND` switches the firmware to Prepare mode. When the robot is
  not upright it first runs Booster's built-in get-up (`GetUpWithMode`,
  `booster.getup_version` 0 = V1, 1 = V2 on K1), which ends in Walking mode,
  and requests Prepare after. `STAND -> IDLE` switches to Damping: the robot
  goes limp, as at the end of Booster's own sequence, so hold it.
- Entering Custom mode (`STAND -> WALK` or `STAND -> TASK`) checks the
  posture, primes a hold of the current pose with the `prepare_state` gains
  and switches the firmware to Custom mode; the policy then publishes
  `/joint_ctrl`. An unsafe posture switches to Damping instead (`ESTOP`).
  `WALK <-> TASK` only switches the policy. Leaving Custom mode stops the
  policy first and then switches the firmware to the target's mode; if the
  firmware refuses, the next mode is tried (Prepare <-> Walking, then
  Damping) and the verdict names the state reached.
- The velocity command is zeroed when `WALK` is entered, so a keyboard
  velocity left over from before does not make the robot walk off.
- The executor child loads both policies at start-up and then logs
  `FSM executor ready after N s`. Until then every request is answered
  `REJECTED <state>: executor not ready` and the firmware is not touched;
  if it is not ready after 5 s the deploy warns with the child's PID
  (`py-spy dump --pid <PID>` shows where it is). On the robot the child
  publishes `/joint_ctrl` through the publisher inherited from the main
  process, as Booster's own inference process does. `--sim` and
  `--monitor` make it create its own ROS 2 node and publisher instead
  (`--executor-ros-context` forces it), which is what a simulator or a
  monitor started after the deploy needs to receive the commands, and
  which some Fast DDS builds cannot do safely in a forked process.
- A policy's safety fallback moves to `ESTOP`; a policy that finishes (for
  example a motion with `stop_at_motion_end`) moves to `booster.after_task`
  (`"stand"` by default, or `"walk"`).
- `Ctrl+C` hands the robot back from `WALK` or `TASK` (`booster.exit_mode`:
  `"walking"` -> Walking mode, `"damping"` -> Damping mode) and exits; in
  `STAND` the firmware already controls the robot and is left as it is.
- A mode watchdog polls `GetStatus` every `booster.mode_check_period_s`
  (1 s). If the firmware changed mode on its own (fall protection, a
  restart, the operator app or the Booster remote; or a restarted
  simulator), the deploy follows it: Prepare -> `STAND`, Walking -> `IDLE`,
  Damping -> `ESTOP`, Custom under another controller -> `IDLE`; the policy
  stops publishing if it was.
- Transitions can also be requested by name on the `booster_deploy/fsm_request`
  topic (`std_msgs/String`, used by the monitor). The current state is
  published latched on `booster_deploy/fsm_state`, and every request (topic,
  keyboard or gamepad) gets a verdict on `booster_deploy/fsm_result`
  (`OK`, `REJECTED`, `IGNORED` or `FAILED`, followed by the target and a
  reason):

  ```bash
  ros2 topic echo /booster_deploy/fsm_result &
  ros2 topic pub --once /booster_deploy/fsm_request std_msgs/msg/String "{data: STAND}"
  ```

> **Added in this branch.** The verdicts on `booster_deploy/fsm_result` are
> among the features added on this branch (see
> [What's new in this branch](#whats-new-in-this-branch)); `main` only
> accepted requests on `booster_deploy/fsm_request` and published the
> state. The verdict texts are:
>
> | Verdict | When |
> |---------|------|
> | `OK <state>` | the transition happened |
> | `REJECTED <state>: not allowed from <current>` | not in the transition table |
> | `REJECTED <name>: unknown state` | topic request with a name that is not a state |
> | `IGNORED <state>: already the current state` | topic request for the current state |
> | `REJECTED WALK: no locomotion policy for this robot` | the robot has no locomotion task, or the task launched is that policy |
> | `FAILED <WALK or TASK>: unsafe posture, switched to Damping` | posture check failed on entering Custom mode; the robot is in `ESTOP` |
> | `FAILED <state>: <current> -> <state> failed[, robot is in <other>]` | the mode RPC, get-up or executor hand-over failed; a fallback mode may have been reached |

The same flow can be exercised without hardware with the simulated robot
(see below).

### Remote Controller

The deployment supports both remote controllers and keyboard input:

- GameSir: detected automatically.
- Booster remote: used through `/remote_controller_state`.
- Keyboard: available.

<table>
  <tr>
    <th align="left">GameSir</th>
    <th align="left">Booster remote</th>
  </tr>
  <tr>
    <td valign="top"><img src="docs/images/gamesir.jpg" alt="GameSir remote" width="320"></td>
    <td valign="top"><img src="docs/images/booster_remote.jpg" alt="Booster remote" width="320"></td>
  </tr>
</table>

On either remote controller, use the left stick for forward/lateral motion, the
right stick for rotation, and the face buttons to drive the state machine.

| Control | Action |
|---------|--------|
| Left stick forward/back | Increase/decrease forward velocity (`vx`) |
| Left stick left/right | Increase/decrease lateral velocity (`vy`) |
| Right stick left/right | Rotate left/right (`vyaw`) |
| Joystick `X` | `IDLE -> STAND` |
| Joystick `A` | forward: `STAND -> WALK -> TASK` |
| Joystick `Y` | back one state (`ESTOP -> IDLE`) |
| Joystick `B` | `ESTOP` |

Keyboard:

| Key | Action |
|-----|--------|
| `w` / `s` | Increase/decrease `vx` by `0.1` |
| `a` / `d` | Increase/decrease `vy` by `0.1` |
| `q` / `e` | Increase/decrease `vyaw` by `0.1` |
| `x` | `IDLE -> STAND` |
| `r` | forward: `STAND -> WALK -> TASK` |
| `n` | back one state (`ESTOP -> IDLE`) |
| `b` | `ESTOP` |
| `Space` | Set all velocity commands to zero |

Stop the deployment with `Ctrl+C`; the robot is handed back to the mode
selected by `booster.exit_mode`.


## Repository Layout

```
booster_deploy/
├─ booster_deploy/           # Controllers, policies, utilities
│  └─ robots/                # Robot model configurations
│     ├─ k1.py               # K1 configuration
│     ├─ t1.py               # T1 23-DOF configuration
│     ├─ t2.py               # T2 31-DOF configuration
│     ├─ __init__.py         # Public configuration exports
│     └─ booster.py          # Backward-compatible import path
├─ scripts/                  # Entry-point scripts (deploy.py)
├─ tasks/                    # Task registry and configs
└─ requirements.txt          # Python dependencies
```

Key modules:
- `booster_deploy/`: Core module providing a unified abstraction for MuJoCo and physical robots. Real-robot communication uses ROS 2 DDS (a `/low_state` subscriber, `/joint_ctrl` publisher, and RPC client).
- `booster_deploy/robots/`: Robot configuration modules. Each robot has a dedicated module that defines a `RobotCfg` describing:
    - `k1.py`: `K1_CFG`
    - `t1.py`: `T1_23DOF_CFG`
    - `t2.py`: `T2_31DOF_CFG`
    - joint names and body names
    - default joint positions
    - default joint stiffness (`joint_stiffness`) and damping (`joint_damping`)
    - effort limits
    - `mjcf_path` for MuJoCo model loading
    - `prepare_state` (prepare pose, stiffness and damping used when entering custom mode)

  Import configurations from the package or from the robot-specific module:

  ```python
  from booster_deploy.robots import K1_CFG
  # Equivalent:
  from booster_deploy.robots.k1 import K1_CFG
  ```

  `booster_deploy.robots.booster` remains available as a backward-compatible import path for existing deployments.

 - `tasks/`: User task definitions and implementations. Each task module contains:
    - `Policy`/`PolicyCfg` class implementing the inference logic;
    - a `ControllerCfg` class describing the task configuration including the policy;
    - registering a task with a `ControllerCfg` instance.

   Typical task layout (example):

   ```text
   tasks/my_task/
   ├─ __init__.py        # registers the task via utils.register.register_task
   ├─ task.py            # Policy and ControllerCfg implementation
   ├─ models/            # optional policy checkpoints
   └─ motions/           # optional motion primitives or recordings
   ```

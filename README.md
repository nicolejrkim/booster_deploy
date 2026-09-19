# Booster Deploy

Booster Deploy is a lightweight deployment framework that supports running control policies on Booster robots (sim2real) and MuJoCo (sim2sim). The system adopts many well-established designs from IsaacLab to provide modular abstractions, allowing unified policy execution across simulated and real platforms.


## What's new in this branch

> **Added in this branch.** Everything in this section comes from the
> features added on the `feat/monitor-verdicts` branch on top of `main`.
> The detailed sections below that describe them carry the same notice.

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

Step 3 prints the `register_booster_train_dance(...)` line to add to
`tasks/beyond_mimic/robots/k1/__init__.py`; the deploy task then uses the
`BeyondMimicPolicy` observation with the gains Booster deploys its K1 dances
with. Our booster_train checkout carries a branch that renames its K1 config to
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
in Walking mode holding the prepare pose with the prepare gains;
Damping mode only damps the joints; Custom mode applies the latest
`/joint_ctrl` command, and commands received before the switch are retained
as on the robot. Booster's built-in locomotion controller is not emulated, so
Walking mode is a standing hold. Useful options: `--state-rate` (default
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
(`booster_deploy/fsm`). Only the listed transitions are accepted; anything
else is refused and logged. The current state and the buttons that act on it
are printed at every change.

| State | Robot mode | What runs |
|-------|------------|-----------|
| `IDLE` | Walking (Booster's own controller) | nothing published; start state |
| `STAND` | Custom | PD hold of `robot.prepare_state` (1 s interpolation on entry) |
| `WALK` | Custom | the robot's locomotion policy with stick commands |
| `TASK` | Custom | the policy selected with `--task` |
| `ESTOP` | Damping | robot goes limp; nothing published |

```text
IDLE --X--> STAND --A--> WALK --A--> TASK        Y steps back one state
any --B--> ESTOP --Y--> IDLE                      Ctrl+C: exit_mode, then quit
```

- `robot.prepare_mode` decides where `A` goes from `STAND`: `"walking"`
  (default) inserts the `WALK` state (`STAND -> WALK -> TASK`), `"standing"`
  goes straight to `TASK`.
- Entering `STAND` from `IDLE` checks the posture, primes a hold of the
  current pose with the `prepare_state` gains and switches the firmware to
  Custom mode. An unsafe posture switches to Damping instead (`ESTOP`).
- A policy's safety fallback moves to `ESTOP`; a policy that finishes (for
  example a motion with `stop_at_motion_end`) moves to `booster.after_task`
  (`"stand"` by default, or `"walk"`).
- `Ctrl+C` hands the robot back (`booster.exit_mode`: `"walking"` -> `IDLE`,
  `"damping"` -> `ESTOP`) and exits.
- A mode watchdog polls `GetStatus` every `booster.mode_check_period_s`
  (1 s) while a Custom state is active. If the firmware left Custom mode on
  its own (fall protection, restart, the operator app; or a restarted
  simulator), the deploy stops publishing and follows the robot to `IDLE` or
  `ESTOP` instead of driving a ghost.
- `ESTOP -> IDLE` on a robot that is not upright uses Booster's built-in
  get-up (`GetUpWithMode`, `booster.getup_version` 0 = V1, 1 = V2 on K1) and
  waits until the robot reports Walking mode.
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
> | `FAILED STAND: unsafe posture, switched to Damping` | posture check failed on `IDLE -> STAND`; the robot is in `ESTOP` |
> | `FAILED <state>: <current> -> <state> failed` | the mode RPC, get-up or executor hand-over failed |

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

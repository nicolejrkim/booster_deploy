"""States and transitions of the Booster real-robot deployment.

The states outside Custom mode are the robot firmware's own modes (the Loco
RPC ``RobotMode`` enum), so X and Y follow the same Damping -> Prepare
sequence as Booster's remote; WALK and TASK run our policies in Custom mode.

::

    IDLE ──X──▶ STAND ──A──▶ WALK ──A──▶ TASK
      ▲          │  ▲          │  ▲         │
      └────Y─────┘  └────Y─────┘  └────Y────┘        (A from STAND goes
                                                       straight to TASK when
    any ──B──▶ ESTOP ──Y──▶ IDLE                       the robot prepares
                                                       standing)

- ``IDLE``: Damping mode, the robot as it boots (motors limp), or Walking
  mode (Booster's own controller, after a get-up or ``exit_mode="walking"``).
  Start state; the deployment publishes nothing.
- ``STAND``: Prepare mode; the firmware's standing controller holds a
  two-foot standing posture.  Entered from ``IDLE`` with the firmware's
  get-up first when the robot is not upright.
- ``WALK``: Custom mode; the robot's locomotion policy (``tasks/locomotion``)
  with stick commands.
- ``TASK``: Custom mode; the policy selected with ``--task``.
- ``ESTOP``: Damping mode after an abort; the robot goes limp.  Entered on
  B, on a policy safety fallback, when the firmware damps on its own (fall
  protection) and on Ctrl+C with ``exit_mode="damping"``.

Entering Custom mode (STAND -> WALK / TASK) checks the posture and primes a
hold of the current pose with the prepare gains; WALK <-> TASK is a policy
switch only.  A policy that finishes (``controller.finish()``, e.g. at the
end of a motion) returns to ``booster.after_task`` (STAND by default).  The
deployment starts in the state matching the robot's current mode and follows
the firmware whenever it changes mode on its own
(``booster.mode_check_period_s``).
"""
from __future__ import annotations

IDLE = "IDLE"
STAND = "STAND"
WALK = "WALK"
TASK = "TASK"
ESTOP = "ESTOP"

STATES = (IDLE, STAND, WALK, TASK, ESTOP)
# States in which the deployment publishes /joint_ctrl (firmware Custom mode).
CUSTOM_STATES = (WALK, TASK)
# Firmware mode of each state (names of BoosterRobotPortal._change_robot_mode).
# IDLE also covers Walking mode (Booster's own controller).
ROBOT_MODE = {
    IDLE: "damping",
    STAND: "prepare",
    WALK: "custom",
    TASK: "custom",
    ESTOP: "damping",
}

TRANSITIONS = {
    (IDLE, STAND),
    (STAND, WALK), (STAND, TASK),
    (WALK, TASK), (WALK, STAND),
    (TASK, WALK), (TASK, STAND),
    (STAND, IDLE), (WALK, IDLE), (TASK, IDLE),
    (IDLE, ESTOP), (STAND, ESTOP), (WALK, ESTOP), (TASK, ESTOP),
    (ESTOP, IDLE),
}

_INDEX = {name: i for i, name in enumerate(STATES)}


def state_index(name: str) -> int:
    return _INDEX[name]


def state_name(index: int) -> str:
    return STATES[index]

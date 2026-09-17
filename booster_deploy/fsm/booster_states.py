"""States and transitions of the Booster real-robot deployment.

::

    IDLE ──X──▶ STAND ──A──▶ WALK ──A──▶ TASK
      ▲          │  ▲          │  ▲         │
      └────Y─────┘  └────Y─────┘  └────Y────┘        (A from STAND goes
                                                       straight to TASK when
    any ──B──▶ ESTOP ──Y──▶ IDLE                       the robot prepares
                                                       standing)

- ``IDLE``: the robot runs Booster's built-in controller (Walking mode).
  Start state; Ctrl+C returns here (``booster.exit_mode="walking"``).
- ``STAND``: Custom mode; a PD hold of ``robot.prepare_state`` (interpolated
  from the current pose over ``booster.stand_transition_s``).
- ``WALK``: Custom mode; the robot's locomotion policy with stick commands.
- ``TASK``: Custom mode; the policy selected with ``--task``.
- ``ESTOP``: Booster Damping mode; the robot goes limp.  Entered on B, on
  a policy safety fallback, and on Ctrl+C with ``exit_mode="damping"``.

A policy that finishes (``controller.finish()``, e.g. at the end of a motion)
returns to ``booster.after_task`` (STAND by default).
"""
from __future__ import annotations

IDLE = "IDLE"
STAND = "STAND"
WALK = "WALK"
TASK = "TASK"
ESTOP = "ESTOP"

STATES = (IDLE, STAND, WALK, TASK, ESTOP)
CUSTOM_STATES = (STAND, WALK, TASK)

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

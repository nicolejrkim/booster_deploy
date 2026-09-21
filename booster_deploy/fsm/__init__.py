"""Finite state machine for real-robot deployments."""

from .state_machine import StateMachine
from .booster_states import (
    CUSTOM_STATES,
    ESTOP,
    IDLE,
    ROBOT_MODE,
    STAND,
    STATES,
    TASK,
    TRANSITIONS,
    WALK,
    state_index,
    state_name,
)

__all__ = [
    "StateMachine",
    "CUSTOM_STATES", "ESTOP", "IDLE", "ROBOT_MODE", "STAND", "STATES",
    "TASK",
    "TRANSITIONS", "WALK", "state_index", "state_name",
]

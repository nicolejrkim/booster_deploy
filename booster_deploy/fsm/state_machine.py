"""A minimal finite state machine with an explicit transition table."""
from __future__ import annotations

from typing import Iterable


class StateMachine:
    """States are strings; only listed ``(from, to)`` transitions are legal.

    The machine only tracks and validates the current state.  Performing a
    transition (mode switches, controller hand-over) is the caller's job:
    call :meth:`can` first, do the work, then :meth:`switch`.
    """

    def __init__(
        self,
        states: Iterable[str],
        transitions: Iterable[tuple[str, str]],
        initial: str,
    ) -> None:
        self.states = list(states)
        self.transitions = set(transitions)
        if initial not in self.states:
            raise ValueError(f"unknown initial state {initial!r}")
        for src, dst in self.transitions:
            if src not in self.states or dst not in self.states:
                raise ValueError(f"transition {src!r} -> {dst!r} uses an "
                                 "unknown state")
        self.current = initial
        self.history: list[tuple[str, str]] = []

    def can(self, to: str) -> bool:
        return (self.current, to) in self.transitions

    def targets(self) -> list[str]:
        """States reachable from the current one, in declaration order."""
        return [s for s in self.states if (self.current, s) in self.transitions]

    def switch(self, to: str) -> None:
        """Record a completed transition (no validation; see :meth:`can`)."""
        if to not in self.states:
            raise ValueError(f"unknown state {to!r}")
        self.history.append((self.current, to))
        self.current = to

"""Game-agnostic contracts shared by every game plugin.

A game plugin provides:

* a **rule engine**: the single source of truth for what is legal (no strategy),
* zero or more **policies**: rank the rule engine's legal actions (heuristics today;
  search, Monte Carlo or an LLM agent later),
* a **GameDefinition** registered in :mod:`card_game_engine.core.registry`.

Planners, agents and adapters should depend only on these protocols, so they
work with any game that implements them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

S = TypeVar("S")  # game state
A = TypeVar("A")  # game action


@runtime_checkable
class GameAction(Protocol):
    """A decision a player can take. Must be hashable/immutable and self-describing."""

    kind: str

    def describe(self) -> str: ...


@dataclass
class Transition(Generic[S]):
    """Result of applying an action: the new state plus human-readable notes."""

    state: S
    notes: list[str] = field(default_factory=list)


class IllegalActionError(ValueError):
    """Raised by ``RuleEngine.apply`` when an action violates the rules."""

    def __init__(self, action: Any, reasons: list[str]):
        self.action = action
        self.reasons = reasons
        label = action.describe() if hasattr(action, "describe") else repr(action)
        super().__init__(f"Illegal action '{label}': " + "; ".join(reasons))


@runtime_checkable
class RuleEngine(Protocol[S, A]):
    """Deterministic rules for one game. Pure: ``apply`` never mutates its input."""

    def legal_actions(self, state: S) -> list[A]: ...

    def check(self, state: S, action: A) -> list[str]:
        """Reasons the action is illegal; empty list means legal."""
        ...

    def apply(self, state: S, action: A) -> Transition[S]: ...

    def blocking_issues(self, state: S) -> list[str]:
        """Hidden information or manual steps that must be resolved before acting."""
        ...


@runtime_checkable
class Policy(Protocol[S]):
    """Ranks legal actions for a state. Must only choose among ``engine.legal_actions``."""

    def recommend(self, state: S, top_n: int | None = None) -> Any: ...


@dataclass(frozen=True)
class GameDefinition:
    """Everything the framework needs to know about one game plugin."""

    id: str                      # CLI/registry key, e.g. "dominion"
    name: str                    # display name
    ruleset: str                 # rules edition / subset supported
    create_engine: Callable[[], RuleEngine]
    create_policy: Callable[[RuleEngine], Policy]
    parse_state: Callable[[dict], Any]      # snapshot dict -> typed state
    register_cli: Callable[[argparse.ArgumentParser], None]
    description: str = ""
    initialize_session: Callable[[dict], Any] | None = None  # config dict -> tracking session (optional)

"""Player decisions the rule engine can validate and apply."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class PlayAction:
    """Play an Action card from hand during the Action phase.

    ``throne_target``: for Throne Room, the Action card to play twice (None = play nothing).
    ``trash_copper``: for Moneylender, whether to trash a Copper for +$3.
    """

    card: str
    throne_target: str | None = None
    trash_copper: bool | None = None
    kind: str = "play_action"

    def describe(self) -> str:
        if self.throne_target:
            return f"Play {self.card} on {self.throne_target}"
        if self.card == "Throne Room":
            return "Play Throne Room (no target)"
        if self.trash_copper is True:
            return f"Play {self.card} (trash a Copper for +$3)"
        if self.trash_copper is False:
            return f"Play {self.card} (don't trash)"
        return f"Play {self.card}"


@dataclass(frozen=True)
class EndActionPhase:
    kind: str = "end_action_phase"

    def describe(self) -> str:
        return "End Action phase (go to Buy phase)"


@dataclass(frozen=True)
class PlayTreasure:
    card: str
    kind: str = "play_treasure"

    def describe(self) -> str:
        return f"Play {self.card}"


@dataclass(frozen=True)
class PlayAllTreasures:
    kind: str = "play_all_treasures"

    def describe(self) -> str:
        return "Play all Treasures"


@dataclass(frozen=True)
class Buy:
    card: str
    kind: str = "buy"

    def describe(self) -> str:
        return f"Buy {self.card}"


@dataclass(frozen=True)
class EndTurn:
    kind: str = "end_turn"

    def describe(self) -> str:
        return "End turn (Clean-up)"


Action = Union[PlayAction, EndActionPhase, PlayTreasure, PlayAllTreasures, Buy, EndTurn]

_KINDS = {
    "play_action": PlayAction,
    "end_action_phase": EndActionPhase,
    "play_treasure": PlayTreasure,
    "play_all_treasures": PlayAllTreasures,
    "buy": Buy,
    "end_turn": EndTurn,
}


def action_to_dict(action: Action) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": action.kind, "label": action.describe()}
    for attr in ("card", "throne_target", "trash_copper"):
        if hasattr(action, attr) and getattr(action, attr) is not None:
            out[attr] = getattr(action, attr)
    return out


def action_from_dict(data: dict[str, Any]) -> Action:
    kind = data.get("kind")
    cls = _KINDS.get(kind)
    if cls is None:
        raise ValueError(f"Unknown action kind {kind!r}")
    kwargs = {k: v for k, v in data.items() if k in ("card", "throne_target", "trash_copper")}
    return cls(**kwargs)  # type: ignore[arg-type]

"""Typed, JSON-serialisable observation events for a Dominion tracking session.

Events record what the user saw or did in the real game. They hold card names as
entered; names are resolved (and validated) when the session replays them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Union


class SessionError(ValueError):
    """A recorded event could not be applied. ``event_index`` is its position in the log."""

    def __init__(self, event_index: int | None, kind: str | None, problem: str):
        self.event_index = event_index
        self.kind = kind
        self.problem = problem
        where = f"Event #{event_index}" if event_index is not None else "Event"
        super().__init__(f"{where} ({kind}): {problem}" if kind else f"{where}: {problem}")


class InconsistentObservation(SessionError):
    """The event contradicts the tracked history or the rules (impossible cards, counts or resources)."""


class IllegalEvent(SessionError):
    """The event is malformed (unknown card, bad arguments) or cannot be tracked from what is known."""


def _cards(value: Iterable[str] | Mapping[str, int] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise TypeError("cards must be a list of card names or a {name: count} mapping, not a string")
    if isinstance(value, Mapping):
        return tuple(name for name, n in value.items() for _ in range(int(n)))
    return tuple(value)


@dataclass(frozen=True)
class ObserveHand:
    """My current hand is exactly ``cards``; resolves pending draws (e.g. the start-of-turn 5)."""

    cards: tuple[str, ...]
    kind: str = "observe_hand"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cards", _cards(self.cards))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "cards": list(self.cards)}


@dataclass(frozen=True)
class RecordDraw:
    """Cards drawn mid-turn: the observed ``cards``, or ``count`` unobserved cards."""

    cards: tuple[str, ...] | None = None
    count: int | None = None
    kind: str = "record_draw"

    def __post_init__(self) -> None:
        if self.cards is not None:
            object.__setattr__(self, "cards", _cards(self.cards))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "cards": None if self.cards is None else list(self.cards), "count": self.count}


@dataclass(frozen=True)
class RecordPlay:
    """A card played from hand (Action or Treasure), applied through the rule engine."""

    card: str
    throne_target: str | None = None
    trash_copper: bool | None = None
    kind: str = "record_play"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "card": self.card, "throne_target": self.throne_target,
                "trash_copper": self.trash_copper}


@dataclass(frozen=True)
class RecordPurchase:
    """A card I bought (engine Buy; Treasures in hand are played first if needed)."""

    card: str
    kind: str = "record_purchase"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "card": self.card}


@dataclass(frozen=True)
class RecordGain:
    """A card I gained without buying it; ``destination`` is "discard", "hand" or "deck" (on top)."""

    card: str
    destination: str = "discard"
    kind: str = "record_gain"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "card": self.card, "destination": self.destination}


@dataclass(frozen=True)
class RecordTrash:
    """A card of mine trashed from ``zone``: "hand", "in_play" or "discard"."""

    card: str
    zone: str = "hand"
    kind: str = "record_trash"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "card": self.card, "zone": self.zone}


@dataclass(frozen=True)
class RecordDiscard:
    """Cards discarded from my hand (e.g. Cellar, or an opponent's Militia)."""

    cards: tuple[str, ...]
    kind: str = "record_discard"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cards", _cards(self.cards))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "cards": list(self.cards)}


@dataclass(frozen=True)
class RecordSupply:
    """An observed Supply pile count. Never used to infer my own gains."""

    pile: str
    count: int
    kind: str = "record_supply"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "pile": self.pile, "count": self.count}


@dataclass(frozen=True)
class EndTurnObservation:
    """My turn ended: Clean-up moves hand and play to the discard pile; the next turn draws 5."""

    kind: str = "end_turn"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


Event = Union[ObserveHand, RecordDraw, RecordPlay, RecordPurchase, RecordGain, RecordTrash, RecordDiscard,
              RecordSupply, EndTurnObservation]

EVENT_TYPES: dict[str, type] = {
    cls.kind: cls  # type: ignore[attr-defined]
    for cls in (ObserveHand, RecordDraw, RecordPlay, RecordPurchase, RecordGain, RecordTrash, RecordDiscard,
                RecordSupply, EndTurnObservation)
}


def event_from_dict(data: Mapping[str, Any]) -> Event:
    """Inverse of ``event.to_dict()``."""
    if not isinstance(data, Mapping):
        raise IllegalEvent(None, None, f"an event must be a JSON object, not {type(data).__name__}")
    kind = data.get("kind")
    cls = EVENT_TYPES.get(kind)  # type: ignore[arg-type]
    if cls is None:
        raise IllegalEvent(None, None, f"unknown event kind {kind!r} (known: {', '.join(EVENT_TYPES)})")
    try:
        return cls(**{k: v for k, v in data.items() if k != "kind"})
    except TypeError as e:
        raise IllegalEvent(None, kind, f"bad fields: {e}") from None

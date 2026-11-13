"""Zone knowledge (draw pile / discard pile) and the deterministic zone rules.

What is known about a hidden zone is modelled explicitly: a known ordered top
prefix, a known multiset in random order, a number of cards of unknown identity
("?"), or nothing at all (size unknown). Knowing a composition never implies
knowing an order, and nothing here assigns probabilities - see
``games.dominion.planning.draws`` for that.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:  # avoid an import cycle: state.py imports this module
    from .cards import CardRegistry
    from .state import GameState

UNKNOWN_CARD = "?"

CardCounts = tuple[tuple[str, int], ...]


class InformationLevel(str, Enum):
    FULLY_KNOWN_ORDER = "fully_known_order"  # whole draw pile order known
    KNOWN_COMPOSITION = "known_composition"  # multiset known, order unknown (a known top prefix is allowed)
    PARTIALLY_KNOWN = "partially_known"      # some cards known, unknown > 0
    INSUFFICIENT = "insufficient"            # draw pile size unknown


def _normalise_counts(counts: Mapping[str, int] | Iterable[tuple[str, int]]) -> CardCounts:
    items = dict(counts.items() if isinstance(counts, Mapping) else counts)
    if any(n < 0 for n in items.values()):
        raise ValueError(f"Card counts must be non-negative: {items}")
    return tuple(sorted((c, n) for c, n in items.items() if n > 0))


@dataclass(frozen=True)
class DrawPileKnowledge:
    """What is known about the draw pile. ``rest`` also accepts a ``{card: count}`` mapping."""

    top: tuple[str, ...] = ()  # known ordered prefix, top card first
    rest: CardCounts = ()      # known multiset below ``top``, random order
    unknown: int = 0           # cards of unknown identity, shuffled in with ``rest``
    size_known: bool = True    # False => INSUFFICIENT; top/rest/unknown are then ignored
    source: str = "given"      # "given" | "derived:<rule>"

    def __post_init__(self) -> None:
        if self.unknown < 0:
            raise ValueError("unknown must be >= 0")
        object.__setattr__(self, "top", tuple(self.top))
        object.__setattr__(self, "rest", _normalise_counts(self.rest))

    @property
    def rest_counter(self) -> Counter:
        return Counter(dict(self.rest))

    @property
    def size(self) -> int:
        return len(self.top) + sum(n for _, n in self.rest) + self.unknown

    @property
    def level(self) -> InformationLevel:
        if not self.size_known:
            return InformationLevel.INSUFFICIENT
        if self.unknown:
            return InformationLevel.PARTIALLY_KNOWN
        if self.rest:
            return InformationLevel.KNOWN_COMPOSITION
        return InformationLevel.FULLY_KNOWN_ORDER

    def to_snapshot(self) -> dict[str, Any]:
        """Snapshot form accepted by ``state_from_dict`` (``player.draw_pile``)."""
        if not self.rest and not self.unknown:
            return {"order": list(self.top)}
        out: dict[str, Any] = {}
        if self.top:
            out["top"] = list(self.top)
        if self.rest:
            out["composition"] = dict(self.rest)
        if self.unknown:
            out["unknown_count"] = self.unknown
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "top": list(self.top), "composition": dict(self.rest), "unknown_count": self.unknown,
            "size": self.size if self.size_known else None, "size_known": self.size_known,
            "level": self.level.value, "source": self.source,
        }


@dataclass(frozen=True)
class DiscardKnowledge:
    """What is known about the discard pile. ``cards`` also accepts a ``{card: count}`` mapping."""

    cards: CardCounts = ()
    unknown: int = 0
    size_known: bool = True  # False => contents and size unknown; cards/unknown are then ignored
    source: str = "given"

    def __post_init__(self) -> None:
        if self.unknown < 0:
            raise ValueError("unknown must be >= 0")
        object.__setattr__(self, "cards", _normalise_counts(self.cards))

    @property
    def counter(self) -> Counter:
        return Counter(dict(self.cards))

    @property
    def size(self) -> int:
        return sum(n for _, n in self.cards) + self.unknown

    def to_snapshot(self) -> dict[str, Any]:
        """Snapshot form accepted by ``state_from_dict`` (``player.discard_pile``)."""
        out: dict[str, Any] = {"composition": dict(self.cards)}
        if self.unknown:
            out["unknown_count"] = self.unknown
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "composition": dict(self.cards), "unknown_count": self.unknown,
            "size": self.size if self.size_known else None, "size_known": self.size_known,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Deterministic mechanics
# ---------------------------------------------------------------------------

def reshuffle(discard: DiscardKnowledge) -> tuple[DrawPileKnowledge, DiscardKnowledge]:
    """Shuffle the discard pile to form a new deck (order unknown); the discard pile becomes empty."""
    empty = DiscardKnowledge(source="derived:reshuffle")
    if not discard.size_known:
        return DrawPileKnowledge(size_known=False, source="derived:reshuffle"), empty
    return DrawPileKnowledge(rest=discard.cards, unknown=discard.unknown, source="derived:reshuffle"), empty


def put_on_top(draw: DrawPileKnowledge, cards: Iterable[str]) -> DrawPileKnowledge:
    """Put cards onto the deck (last card ends on top). No-op if the pile size is unknown."""
    if not draw.size_known:
        return draw
    return DrawPileKnowledge(tuple(reversed(list(cards))) + draw.top, draw.rest, draw.unknown, True, draw.source)


def add_to_discard(discard: DiscardKnowledge, cards: Iterable[str], unknown: int = 0) -> DiscardKnowledge:
    """Put known cards (and ``unknown`` unidentified cards) into the discard pile. No-op if its size is unknown."""
    if not discard.size_known:
        return discard
    return DiscardKnowledge(discard.counter + Counter(cards), discard.unknown + unknown, True, discard.source)


def cleanup_zones(
    discard: DiscardKnowledge, hand: Iterable[str], in_play: Iterable[str], unknown_in_hand: int = 0
) -> DiscardKnowledge:
    """Clean-up: hand and in-play cards go to the discard pile (drawing the next hand is a separate draw)."""
    return add_to_discard(discard, [*hand, *in_play], unknown_in_hand)


# ---------------------------------------------------------------------------
# Derivation from a snapshot
# ---------------------------------------------------------------------------

def _subtract(owned: Counter, *parts: Counter) -> tuple[Counter, list[str]]:
    """``owned - parts``; second item lists cards that would go negative (empty if consistent)."""
    remaining = Counter(owned)
    for part in parts:
        remaining.subtract(part)
    return +remaining, sorted(c for c, n in remaining.items() if n < 0)


def _derive_draw(state: GameState, visible: Counter, notes: list[str]) -> DrawPileKnowledge:
    owned, discard = state.owned_cards, state.discard_pile
    if state.draw_pile is not None:
        notes.append("Draw pile: given in the snapshot.")
        return state.draw_pile
    if owned is not None and discard is not None and discard.size_known and not discard.unknown:
        draw = _derived_draw(state, owned, visible, discard.counter, "owned-minus-visible", notes)
        if draw is not None:
            return draw
    if owned is not None and state.discard_count == 0:
        draw = _derived_draw(state, owned, visible, Counter(), "empty-discard", notes)
        if draw is not None:
            return draw
    if state.deck_count is not None:
        notes.append(f"Draw pile: {state.deck_count} card(s), contents unknown (deck_count).")
        return DrawPileKnowledge(unknown=state.deck_count, source="derived:deck-count")
    notes.append("Draw pile: size and contents unknown (give deck_count, draw_pile or owned_cards + discard).")
    return DrawPileKnowledge(size_known=False, source="derived:none")


def _derived_draw(
    state: GameState, owned: Mapping[str, int], visible: Counter, discard: Counter, rule: str, notes: list[str]
) -> DrawPileKnowledge | None:
    rest, negative = _subtract(Counter(owned), visible, discard)
    if negative:
        notes.append(f"Draw pile not derived ({rule}): owned_cards has too few {', '.join(negative)}.")
        return None
    size = sum(rest.values())
    if state.deck_count is not None and state.deck_count != size:
        notes.append(f"Draw pile not derived ({rule}): it would hold {size} cards but deck_count is {state.deck_count}.")
        return None
    notes.append(f"Draw pile: composition derived as owned cards minus hand, play and discard ({rule}); order unknown.")
    return DrawPileKnowledge(rest=rest, source=f"derived:{rule}")


def _derive_discard(
    state: GameState, visible: Counter, draw: DrawPileKnowledge, notes: list[str]
) -> DiscardKnowledge:
    owned = state.owned_cards
    if state.discard_pile is not None:
        notes.append("Discard pile: given in the snapshot.")
        return state.discard_pile
    if owned is not None and draw.size_known and not draw.unknown:
        cards, negative = _subtract(Counter(owned), visible, draw.rest_counter, Counter(draw.top))
        size = sum(cards.values())
        if negative:
            notes.append(f"Discard pile not derived: owned_cards has too few {', '.join(negative)}.")
        elif state.discard_count is not None and state.discard_count != size:
            notes.append(f"Discard pile not derived: it would hold {size} cards but discard_count is {state.discard_count}.")
        else:
            notes.append("Discard pile: derived as owned cards minus hand, play and draw pile.")
            return DiscardKnowledge(cards=cards, source="derived:owned-minus-visible")
    if state.discard_count is not None:
        notes.append(f"Discard pile: {state.discard_count} card(s), contents unknown (discard_count).")
        return DiscardKnowledge(unknown=state.discard_count, source="derived:discard-count")
    notes.append("Discard pile: size and contents unknown.")
    return DiscardKnowledge(size_known=False, source="derived:none")


def derive_zone_knowledge(
    state: GameState, registry: CardRegistry | None = None
) -> tuple[DrawPileKnowledge, DiscardKnowledge, list[str]]:
    """Derive draw/discard knowledge from a snapshot using exact rules only (never guesses).

    Card names in ``state`` are already canonical, so ``registry`` is accepted for API
    stability but not needed. Returns ``(draw, discard, notes)``.
    """
    notes: list[str] = []
    visible = Counter(state.turn.hand) + Counter(state.turn.in_play)
    draw = _derive_draw(state, visible, notes)
    discard = _derive_discard(state, visible, draw, notes)
    return draw, discard, notes

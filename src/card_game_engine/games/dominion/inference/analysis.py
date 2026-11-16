"""Deck analysis and draw-probability queries over a session's belief (never mutates the session).

* :func:`analyze_deck` - what is in each zone, per card and per type: deterministic deductions
  (``certain``) are kept apart from probability distributions over hidden states and from bounds
  caused by unknown ("?") cards.
* :func:`analyze_draw` - the distribution of the number of target cards among the next n draws.
* :func:`get_guaranteed_cards` - min/max of every card and type in the next n draws.

Draw queries take ``at``: ``now`` (the current piles; pending draws are the first draws),
``next_turn`` (after this turn's Clean-up, with the same assumptions as ``predict_next_hand``) or
``auto`` (``predict_next_hand``'s choice). The n draws are continuous - no Clean-up between them.
The probability work is done by :mod:`..planning.analysis` on the belief's weighted worlds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from ..planning.analysis import (
    TARGET_TYPES,
    CountDistribution,
    Target,
    amount,
    card_target,
    draw_count_distribution,
    guaranteed_range,
    number,
    parse_target,
    type_target,
)
from ..rules.zones import UNKNOWN_CARD
from .belief import ENUMERATION_LIMIT, ZONES, World, collapse
from .replay import shared_engine
from .session import MC_SAMPLES, Session, draw_source, infer_belief_state

CONTINUOUS_NOTE = ("The draws are continuous: no Clean-up happens between them (for multi-turn projections use "
                   "the planner's --horizon hands:N).")
BOUNDS_NOTE = ("Some cards are unknown ('?'): values are bounds - lower counts no '?' as a match, upper counts "
               "every drawn '?' as a match.")
NOT_DETERMINABLE_NOTE = ("Not determinable: the draws reach a pile whose size is unknown, so no probabilities are "
                         "given (min/max are still valid bounds). Record the hand or start from a snapshot with "
                         "deck_count / owned_cards.")
ANALYSIS_ZONES = ("draw_pile", "discard_pile", "hand", "in_play", "owned", "trash")


def _target(target: Target | str) -> Target:
    return target if isinstance(target, Target) else parse_target(target, shared_engine().registry)


def _sources(worlds: tuple[World, ...]) -> list[tuple[Fraction, Any, Any]]:
    return [(w.weight, w.draw, w.discard) for w in worlds]


def _at(at: str) -> str:
    return "next_turn" if at == "next-turn" else at


def _draw_notes(session: Session, at: str, n: int, notes: list[str], counts: CountDistribution) -> tuple[str, ...]:
    pending = infer_belief_state(session).pending_draws
    out = list(notes)
    if at == "now" and pending:
        out.append(f"The first {min(pending, n)} of these draw(s) are the {pending} pending draw(s).")
    out.append(CONTINUOUS_NOTE)
    if not counts.determinable:
        out.append(NOT_DETERMINABLE_NOTE)
    elif counts.bounded:
        out.append(BOUNDS_NOTE)
    if counts.method == "monte_carlo":
        out.append(f"Too many outcomes for exact enumeration: Monte Carlo estimate ({counts.samples} samples, "
                   f"seed {counts.seed}).")
    return tuple(out)


# ---------------------------------------------------------------------------
# analyze_draw
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DrawAnalysis:
    """How many ``target`` cards are among the next ``draw_count`` draws (from ``at``)."""

    target: Target
    draw_count: int
    at: str                      # resolved: "now" | "next_turn"
    turn: int | None             # the turn whose piles are drawn from
    counts: CountDistribution    # P(X = k), bounds, P(reshuffle), P(short draw), min/max
    worlds: int
    notes: tuple[str, ...] = ()

    @property
    def determinable(self) -> bool:
        return self.counts.determinable

    @property
    def deterministic(self) -> bool:
        return self.counts.deterministic

    @property
    def bounded(self) -> bool:
        return self.counts.bounded

    @property
    def method(self) -> str:
        return self.counts.method

    @property
    def min(self) -> int:
        return self.counts.min

    @property
    def max(self) -> int:
        return self.counts.max

    def p_exactly(self, k: int):
        return self.counts.p_exactly(k)

    def p_at_least(self, k: int):
        return self.counts.p_at_least(k)

    def to_dict(self) -> dict[str, Any]:
        c = self.counts.to_dict()
        return {"query": "draws", "at": self.at, "turn": self.turn, "worlds": self.worlds, **c,
                "notes": list(self.notes)}


def analyze_draw(session: Session, draw_count: int, target: Target | str, at: str = "auto") -> DrawAnalysis:
    """Distribution of the number of ``target`` cards (a card or a type) among the next ``draw_count`` draws."""
    if not isinstance(draw_count, int) or isinstance(draw_count, bool) or draw_count < 1:
        raise ValueError("draw count must be a positive integer")
    tgt = _target(target)
    resolved, turn, worlds, notes = draw_source(session, _at(at))
    counts = draw_count_distribution(_sources(worlds), draw_count, tgt, ENUMERATION_LIMIT, MC_SAMPLES, seed=0)
    return DrawAnalysis(tgt, draw_count, resolved, turn, counts, len(worlds),
                        _draw_notes(session, resolved, draw_count, notes, counts))


# ---------------------------------------------------------------------------
# get_guaranteed_cards
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CountBound:
    """Min/max of one card or type in the next draws (max counts drawn "?" as possible matches)."""

    target: Target
    min: int
    max: int

    def describe(self) -> str:
        what = self.target.name
        return f"exactly {self.min} {what}" if self.min == self.max else f"at least {self.min} {what}"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.target.name, "kind": self.target.kind, "min": self.min, "max": self.max}


@dataclass(frozen=True)
class GuaranteedCards:
    """Deterministic guarantees (min > 0) and impossibilities (max == 0) for the next ``draw_count`` draws."""

    draw_count: int
    at: str
    turn: int | None
    cards: tuple[CountBound, ...]    # every card that can be drawn (max > 0)
    types: tuple[CountBound, ...]    # every type that can be drawn
    impossible: tuple[str, ...]      # owned cards that cannot be among these draws
    determinable: bool
    bounded: bool
    p_reshuffle: Fraction | float | None
    p_short: Fraction | float | None
    worlds: int
    method: str
    notes: tuple[str, ...] = ()

    @property
    def guarantees(self) -> tuple[CountBound, ...]:
        return tuple(b for b in self.cards + self.types if b.min > 0)

    def bound(self, name: str) -> CountBound | None:
        return next((b for b in self.cards + self.types if b.target.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": "guaranteed", "draw_count": self.draw_count, "at": self.at, "turn": self.turn,
            "determinable": self.determinable, "bounded": self.bounded,
            "guarantees": [{**b.to_dict(), "text": b.describe()} for b in self.guarantees],
            "cards": [b.to_dict() for b in self.cards], "types": [b.to_dict() for b in self.types],
            "impossible": list(self.impossible), "p_reshuffle": number(self.p_reshuffle),
            "p_short_draw": number(self.p_short), "worlds": self.worlds, "method": self.method,
            "notes": list(self.notes),
        }


def _drawable_names(worlds: tuple[World, ...]) -> set[str]:
    names: set[str] = set()
    for w in worlds:
        for zone in ("draw_pile", "discard_pile"):
            names |= set(w.zone_counter(zone) or ())
    return names - {UNKNOWN_CARD}


def get_guaranteed_cards(session: Session, draw_count: int | None = None, at: str = "auto") -> GuaranteedCards:
    """Min/max of every card and type among the next ``draw_count`` draws (default: the hand size)."""
    eng = shared_engine()
    n = eng.ruleset.hand_size if draw_count is None else draw_count
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("draw count must be a positive integer")
    b = infer_belief_state(session)
    resolved, turn, worlds, notes = draw_source(session, _at(at))
    sources = _sources(worlds)
    if b.owned is not None:
        owned = set(b.owned)
    else:  # owned cards not tracked: every card known to be somewhere
        owned = ({c for w in b.worlds for z in ZONES for c in (w.zone_counter(z) or ())} | set(b.in_play)) \
            - {UNKNOWN_CARD}
    names = sorted(_drawable_names(worlds) | owned)
    ranges = {name: guaranteed_range(sources, n, card_target(name)) for name in names}
    cards = tuple(CountBound(card_target(c), *r) for c, r in ranges.items() if r[1] > 0)
    types = tuple(CountBound(t, *r) for t in (type_target(x, eng.registry) for x in TARGET_TYPES)
                  for r in [guaranteed_range(sources, n, t)] if r[1] > 0)
    summary = draw_count_distribution(sources, n, type_target("Treasure", eng.registry), ENUMERATION_LIMIT,
                                      MC_SAMPLES, seed=0)
    det = summary.determinable
    return GuaranteedCards(
        draw_count=n, at=resolved, turn=turn, cards=cards, types=types,
        impossible=tuple(c for c in sorted(owned) if ranges[c][1] == 0), determinable=det,
        bounded=summary.bounded, p_reshuffle=summary.p_reshuffle if det else None,
        p_short=summary.p_short if det else None, worlds=len(worlds), method=summary.method,
        notes=_draw_notes(session, resolved, n, notes, summary),
    )


# ---------------------------------------------------------------------------
# analyze_deck
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CountSummary:
    """One card or type in one zone.

    ``certain`` is set only for a deterministic count (the same in every hidden state, no "?" that could
    be it). Otherwise ``min``/``max`` bound it (max counts "?" cards as possible copies), ``expected`` is the
    exact mixture expectation of the known copies, and ``distribution`` gives P(count = c) when the count
    is ``probabilistic`` (varies between hidden states, no "?" involved).
    """

    name: str
    kind: str                    # "card" | "type"
    status: str                  # "observed" | "deduced" | "probabilistic" | "unknown"
    certain: int | None
    min: int
    max: int
    expected: Fraction
    distribution: tuple[tuple[int, Fraction], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "kind": self.kind, "status": self.status, "certain": self.certain,
                               "min": self.min, "max": self.max, "expected": amount(self.expected)}
        if self.distribution:
            out["distribution"] = [{"count": c, **number(p)} for c, p in self.distribution]
        return out


@dataclass(frozen=True)
class ZoneAnalysis:
    zone: str
    status: str                  # zone-level label (as in ``BeliefState.zone_status``)
    size: int | None             # None: unknown (or not the same in every hidden state)
    unknown: int | None          # most "?" cards in any hidden state; None if the size is unknown
    top: tuple[str, ...]         # draw pile: known top cards, top first
    cards: tuple[CountSummary, ...]
    types: tuple[CountSummary, ...]

    def card(self, name: str) -> CountSummary | None:
        return next((c for c in self.cards + self.types if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {"zone": self.zone, "status": self.status, "size": self.size, "unknown_count": self.unknown,
                "top": list(self.top), "cards": [c.to_dict() for c in self.cards],
                "types": [c.to_dict() for c in self.types]}


@dataclass(frozen=True)
class DeckAnalysis:
    turn: int | None
    phase: str
    pending_draws: int
    worlds: int
    information_level: str
    zones: tuple[ZoneAnalysis, ...]
    flags: tuple[str, ...] = ()

    def zone(self, name: str) -> ZoneAnalysis:
        return next(z for z in self.zones if z.zone == name)

    def to_dict(self) -> dict[str, Any]:
        return {"query": "deck", "turn": self.turn, "phase": self.phase, "pending_draws": self.pending_draws,
                "worlds": self.worlds, "information_level": self.information_level,
                "zones": {z.zone: z.to_dict() for z in self.zones}, "flags": list(self.flags)}


def _summaries(weighted: list[tuple[Fraction, Counter]], sure: str, targets: list[Target]) -> tuple[CountSummary, ...]:
    out = []
    for t in targets:
        known = [(w, sum(n for c, n in cnt.items() if t.matches(c)), cnt[UNKNOWN_CARD]) for w, cnt in weighted]
        lo, hi = min(k for _, k, _ in known), max(k + u for _, k, u in known)
        if hi == 0:
            continue
        dist: dict[int, Fraction] = {}
        for w, k, _ in known:
            dist[k] = dist.get(k, Fraction(0)) + w
        if lo == hi:
            status = sure
        else:
            status = "unknown" if any(u for _, _, u in known) else "probabilistic"
        out.append(CountSummary(t.name, t.kind, status, lo if lo == hi else None, lo, hi,
                                sum((w * k for w, k, _ in known), Fraction(0)),
                                tuple(sorted(dist.items())) if status == "probabilistic" else ()))
    return tuple(out)


def _zone(zone: str, status: str, weighted: list[tuple[Fraction, Counter]] | None, sure: str,
          top: tuple[str, ...] = ()) -> ZoneAnalysis:
    if weighted is None:
        return ZoneAnalysis(zone, "unknown", None, None, (), (), ())
    reg = shared_engine().registry
    names = sorted({c for _, cnt in weighted for c in cnt} - {UNKNOWN_CARD})
    sizes = {sum(cnt.values()) for _, cnt in weighted}
    return ZoneAnalysis(zone, status, sizes.pop() if len(sizes) == 1 else None,
                        max(cnt[UNKNOWN_CARD] for _, cnt in weighted), top,
                        _summaries(weighted, sure, [card_target(c) for c in names]),
                        _summaries(weighted, sure, [type_target(t, reg) for t in TARGET_TYPES]))


def analyze_deck(session: Session) -> DeckAnalysis:
    """Per zone (draw pile, discard, hand, in play, owned, trash): every card and type, deduced vs probabilistic."""
    b = infer_belief_state(session)
    zones = []
    for zone in ZONES:
        values = [(w.weight, w.zone_counter(zone)) for w in b.worlds]
        known = None if any(c is None for _, c in values) else [(w, c) for w, c in values]
        sure = "observed" if zone == "hand" and b.hand_observed else "deduced"
        top = ()
        if zone == "draw_pile" and known is not None:
            top = collapse(b.worlds).draw.top
        zones.append(_zone(zone, b.zone_status(zone), known, sure, top))  # type: ignore[arg-type]
    zones.insert(2, zones.pop(0))  # order: draw_pile, discard_pile, hand
    one = [(Fraction(1), Counter(b.in_play))]
    zones.append(_zone("in_play", "observed", one, "observed"))
    owned = b.owned
    zones.append(_zone("owned", "deduced", None if owned is None else [(Fraction(1), Counter(owned))], "deduced"))
    zones.append(_zone("trash", "observed", [(Fraction(1), Counter(b.trash))], "observed"))
    return DeckAnalysis(b.turn, b.phase, b.pending_draws, len(b.worlds), b.information_level, tuple(zones), b.flags)

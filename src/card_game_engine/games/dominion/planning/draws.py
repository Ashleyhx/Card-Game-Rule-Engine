"""Exact draw-outcome enumeration (multivariate hypergeometric) and seeded sampling.

Drawing ``n`` cards: the known ``top`` prefix comes first (deterministic), then cards
come uniformly at random from the rest of the pile (known cards plus "?" unknowns).
If the pile runs out, the discard pile is reshuffled (``zones.reshuffle``) and drawing
continues. Outcomes are grouped by multiset - permutations are never enumerated.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Any, Iterator

from ..rules.zones import UNKNOWN_CARD, DiscardKnowledge, DrawPileKnowledge, expand_counts, reshuffle

BOUNDARY_UNKNOWN_DRAW_PILE = "unknown_draw_pile"


@dataclass(frozen=True)
class DrawOutcome:
    cards: tuple[str, ...]          # sorted; "?" for a card of unknown identity
    draw_after: DrawPileKnowledge
    discard_after: DiscardKnowledge
    p_exact: Fraction | None        # None for a sampled outcome
    reshuffled: bool = False
    exhausted: bool = False         # fewer cards drawn because deck and discard ran out
    boundary: str | None = None     # set when drawing could not continue (cards still owed)
    owed: int = 0                   # draws not performed because of ``boundary``

    @property
    def probability(self) -> float | None:
        return None if self.p_exact is None else float(self.p_exact)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability, "cards": list(self.cards), "reshuffled": self.reshuffled,
            "exhausted": self.exhausted, "boundary": self.boundary, "owed": self.owed,
            "draw_after": self.draw_after.to_dict(), "discard_after": self.discard_after.to_dict(),
        }


@dataclass(frozen=True)
class _Plan:
    """The deterministic part of a draw: fixed cards, then ``m`` random cards from ``pile``."""

    fixed: tuple[str, ...]
    pile: DrawPileKnowledge
    discard: DiscardKnowledge
    m: int
    reshuffled: bool = False
    exhausted: bool = False
    boundary: str | None = None
    owed: int = 0


def _pool(pile: DrawPileKnowledge) -> list[tuple[str, int]]:
    """Cards drawn at random (everything below ``top``), with unknowns as "?"."""
    return list(pile.rest) + ([(UNKNOWN_CARD, pile.unknown)] if pile.unknown else [])


def _plan(draw: DrawPileKnowledge, discard: DiscardKnowledge, n: int) -> _Plan:
    if not draw.size_known:
        return _Plan((), draw, discard, 0, boundary=BOUNDARY_UNKNOWN_DRAW_PILE if n else None, owed=n)
    k = min(n, len(draw.top))
    fixed, n = draw.top[:k], n - k
    pool_size = draw.size - len(draw.top)
    if n <= pool_size:
        pile = DrawPileKnowledge(draw.top[k:], draw.rest, draw.unknown, True, draw.source)
        return _Plan(fixed, pile, discard, n)
    fixed += tuple(expand_counts(_pool(draw)))
    n -= pool_size
    reshuffled = not discard.size_known or discard.size > 0
    pile, discard = reshuffle(discard)
    if not pile.size_known:
        return _Plan(fixed, pile, discard, 0, reshuffled, boundary=BOUNDARY_UNKNOWN_DRAW_PILE, owed=n)
    return _Plan(fixed, pile, discard, min(n, pile.size), reshuffled, exhausted=n > pile.size)


def _remove(pile: DrawPileKnowledge, drawn: Counter) -> DrawPileKnowledge:
    if not drawn:
        return pile
    rest = pile.rest_counter
    rest.subtract({c: k for c, k in drawn.items() if c != UNKNOWN_CARD})
    return DrawPileKnowledge((), rest, pile.unknown - drawn[UNKNOWN_CARD], True, pile.source)


def _outcome(plan: _Plan, drawn: Counter, p: Fraction | None) -> DrawOutcome:
    return DrawOutcome(
        cards=tuple(sorted(plan.fixed + tuple(drawn.elements()))),
        draw_after=_remove(plan.pile, drawn),
        discard_after=plan.discard,
        p_exact=p,
        reshuffled=plan.reshuffled,
        exhausted=plan.exhausted,
        boundary=plan.boundary,
        owed=plan.owed,
    )


def grouped_draws(counts: tuple[int, ...], m: int) -> Iterator[tuple[tuple[int, ...], int]]:
    """Every multiset of ``m`` cards from a pile holding ``counts[i]`` cards of kind ``i``: ``(k_0, k_1, ...)``
    with its weight prod C(n_i, k_i) (the number of m-card subsets that give it)."""
    if not counts:
        if m == 0:
            yield (), 1
        return
    n, tail = counts[0], counts[1:]
    tail_size = sum(tail)
    for k in range(max(0, m - tail_size), min(n, m) + 1):
        ways = comb(n, k)
        for rest, w in grouped_draws(tail, m - k):
            yield (k,) + rest, w * ways


def _multisets(items: list[tuple[str, int]], m: int) -> Iterator[tuple[Counter, int]]:
    """Every multiset of size ``m`` drawable from ``items``, with its weight prod C(n_i, k_i)."""
    for ks, w in grouped_draws(tuple(n for _, n in items), m):
        yield Counter({card: k for (card, _), k in zip(items, ks) if k}), w


def enumerate_draws(draw: DrawPileKnowledge, discard: DiscardKnowledge, n: int) -> list[DrawOutcome]:
    """All distinct outcomes of drawing ``n`` cards, most likely first; probabilities sum to exactly 1."""
    plan = _plan(draw, discard, n)
    items = _pool(plan.pile)
    total = comb(sum(c for _, c in items), plan.m)
    outcomes = [_outcome(plan, drawn, Fraction(w, total)) for drawn, w in _multisets(items, plan.m)]
    return sorted(outcomes, key=lambda o: (-o.p_exact, o.cards))


def count_draw_outcomes(draw: DrawPileKnowledge, discard: DiscardKnowledge, n: int) -> int:
    """Number of outcomes ``enumerate_draws`` would return, computed without building them."""
    plan = _plan(draw, discard, n)
    ways = [1] + [0] * plan.m  # ways[j] = number of multisets of size j so far
    for _, cap in _pool(plan.pile):
        ways = [sum(ways[j - k] for k in range(min(cap, j) + 1)) for j in range(plan.m + 1)]
    return ways[plan.m]


def draw_unidentified(draw: DrawPileKnowledge, discard: DiscardKnowledge, n: int, seen: Counter) -> DrawOutcome:
    """Drawing ``n`` cards when "?" cards may be among them, so which cards left the pile is not known.

    ``seen`` holds the identities the drawn cards may have. The fixed part (``top`` prefix, the whole pile
    before a reshuffle) is drawn exactly as in ``enumerate_draws``; of the random part's pool only
    ``max(0, known_c - seen_c)`` of each card ``c`` is certainly left, the rest becomes "?" (same size).
    ``cards`` is the fixed part plus "?" per random card; ``p_exact`` is None (no likelihood is implied).
    """
    plan = _plan(draw, discard, n)
    pile = plan.pile
    if plan.m:
        left = pile.rest_counter - (seen - Counter(plan.fixed))
        pile = DrawPileKnowledge((), left, pile.size - plan.m - sum(left.values()), True, pile.source)
    return DrawOutcome(tuple(sorted(plan.fixed + (UNKNOWN_CARD,) * plan.m)), pile, plan.discard, None,
                       plan.reshuffled, plan.exhausted, plan.boundary, plan.owed)


def sample_draw(draw: DrawPileKnowledge, discard: DiscardKnowledge, n: int, rng: random.Random) -> DrawOutcome:
    """One outcome sampled with ``rng`` (reproducible for a given seed); ``probability`` is None."""
    plan = _plan(draw, discard, n)
    drawn = Counter(rng.sample(expand_counts(_pool(plan.pile)), plan.m))
    return _outcome(plan, drawn, None)

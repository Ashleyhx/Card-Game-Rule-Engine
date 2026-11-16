"""Belief over my hidden zones: weighted worlds, exact Bayesian conditioning, predictive draws.

A :class:`World` is one hypothesis about my draw pile, discard pile and hand; inside a
world the draw pile is still "known multiset in uniformly random order" (the
uniform-shuffle model), so a world stands for many orderings at once. Weights are
exact ``Fraction`` posteriors summing to 1. Draw mechanics (including reshuffles)
come only from ``planning.draws`` / ``rules.zones``.

"?" cards (identity unknown) are consistent with any observed card, but carry no prior,
so there is no likelihood for an observation that a "?" may have matched. Such a world is
kept (it is dropped only if no identities of its "?" cards allow the observation) with
only what is certain: of the pile drawn from, ``max(0, known - observed)`` of each card
stays known and the rest becomes "?"; and as the relative weights of worlds are then
unknown, several surviving worlds are collapsed into one (:func:`condition`). Worlds
without "?" cards involved are conditioned exactly. If ``max_worlds`` would be exceeded,
the worlds are collapsed into one world that keeps only what is certain and marks the rest "?".
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Callable, Iterable

from ..planning.analysis import EXACT_LIMIT as ENUMERATION_LIMIT  # total draw outcomes enumerated in one step
from ..planning.draws import count_draw_outcomes, draw_unidentified, enumerate_draws, sample_draw
from ..rules.state import GameState
from ..rules.zones import (
    UNKNOWN_CARD,
    CardCounts,
    DiscardKnowledge,
    DrawPileKnowledge,
    cleanup_zones,
    expand_counts,
)

ZONES = ("hand", "draw_pile", "discard_pile")


class Contradiction(ValueError):
    """No hidden state is consistent with an observation (the message names the problem)."""


def counts(cards: Iterable[str] | Counter) -> CardCounts:
    """Canonical sorted ``((card, n), ...)`` with positive counts only."""
    c = cards if isinstance(cards, Counter) else Counter(cards)
    return tuple(sorted((k, n) for k, n in c.items() if n > 0))


def fraction_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def describe_counts(c: Counter | CardCounts) -> str:
    items = sorted(c.items()) if isinstance(c, Counter) else c
    return ", ".join(f"{n} {k}" for k, n in items if n > 0) or "nothing"


@dataclass(frozen=True)
class World:
    """One fully specified hypothesis about my hidden zones."""

    weight: Fraction
    draw: DrawPileKnowledge
    discard: DiscardKnowledge
    hand: CardCounts = ()  # may contain "?"

    @property
    def hand_counter(self) -> Counter:
        return Counter(dict(self.hand))

    def key(self) -> tuple:
        d, x = self.draw, self.discard
        dk = (True, d.top, d.rest, d.unknown) if d.size_known else (False,)
        xk = (True, x.cards, x.unknown) if x.size_known else (False,)
        return dk, xk, self.hand

    def zone_counter(self, zone: str) -> Counter | None:
        """Cards in ``zone`` ("?" for unknown identities); None if the zone's size is unknown."""
        if zone == "hand":
            return self.hand_counter
        if zone == "draw_pile":
            if not self.draw.size_known:
                return None
            return self.draw.rest_counter + Counter(self.draw.top) + Counter({UNKNOWN_CARD: self.draw.unknown})
        if not self.discard.size_known:
            return None
        return self.discard.counter + Counter({UNKNOWN_CARD: self.discard.unknown})

    def to_dict(self) -> dict[str, Any]:
        return {"weight": fraction_str(self.weight), "p": float(self.weight), "hand": dict(self.hand),
                "draw_pile": self.draw.to_dict(), "discard_pile": self.discard.to_dict()}


# ---------------------------------------------------------------------------
# World-set operations
# ---------------------------------------------------------------------------

def normalise(worlds: Iterable[World]) -> tuple[World, ...]:
    """Merge identical worlds (summing weights), renormalise exactly, sort deterministically."""
    merged: dict[tuple, World] = {}
    for w in worlds:
        if w.weight == 0:
            continue
        k = w.key()
        merged[k] = replace(merged[k], weight=merged[k].weight + w.weight) if k in merged else w
    total = sum((w.weight for w in merged.values()), Fraction(0))
    if total == 0:
        raise Contradiction("no hidden state is consistent with this observation")
    return tuple(sorted((replace(w, weight=w.weight / total) for w in merged.values()),
                        key=lambda w: (-w.weight, w.key())))


def _certain(counters: list[Counter]) -> Counter:
    names = set().union(*counters) - {UNKNOWN_CARD}
    return +Counter({c: min(cn[c] for cn in counters) for c in names})


def collapse(worlds: Iterable[World]) -> World:
    """One world keeping only what is certain in every world; everything else becomes "?"."""
    ws = list(worlds)
    hands = [w.hand_counter for w in ws]
    known = _certain(hands)
    hand = known + Counter({UNKNOWN_CARD: max(sum(h.values()) for h in hands) - sum(known.values())})
    draws, discards = [w.draw for w in ws], [w.discard for w in ws]
    if all(d.size_known for d in draws):
        top = draws[0].top if len({d.top for d in draws}) == 1 else ()
        rest = _certain([d.rest_counter + (Counter() if top else Counter(d.top)) for d in draws])
        size = max(d.size for d in draws)
        draw = DrawPileKnowledge(top, rest, size - len(top) - sum(rest.values()), source="derived:collapsed")
    else:
        draw = DrawPileKnowledge(size_known=False, source="derived:collapsed")
    if all(x.size_known for x in discards):
        cards = _certain([x.counter for x in discards])
        discard = DiscardKnowledge(cards, max(x.size for x in discards) - sum(cards.values()), source="derived:collapsed")
    else:
        discard = DiscardKnowledge(size_known=False, source="derived:collapsed")
    return World(Fraction(1), draw, discard, counts(hand))


def cap(worlds: tuple[World, ...], max_worlds: int) -> tuple[tuple[World, ...], bool]:
    """Collapse to one world if there are more than ``max_worlds``; second item says whether it happened."""
    if len(worlds) <= max_worlds:
        return worlds, False
    return (collapse(worlds),), True


def condition(kept: Iterable[World], wildcard: bool) -> tuple[tuple[World, ...], bool]:
    """The worlds left by an observation; ``wildcard``: some were kept only because a "?" may match.

    Such a world's likelihood is unknown ("?" has no prior), so the relative weights of several
    worlds are too: they are collapsed into one keeping what is certain. Second item: whether that happened.
    """
    worlds = normalise(kept)
    if wildcard and len(worlds) > 1:
        return (collapse(worlds),), True
    return worlds, False


def map_worlds(worlds: Iterable[World], fn: Callable[[World], World]) -> tuple[World, ...]:
    return normalise(fn(w) for w in worlds)


def cleanup_world(w: World, in_play: list[str]) -> World:
    """Clean-up inside one world: the hand (known cards and '?') and the cards in play go to the discard pile."""
    hand = w.hand_counter
    unknown = hand.pop(UNKNOWN_CARD, 0)
    return World(w.weight, w.draw, cleanup_zones(w.discard, expand_counts(hand), in_play, unknown), ())


def determined_hand(worlds: Iterable[World]) -> list[str] | None:
    """The hand if it is the same in every world and contains no "?", else None."""
    hands = {w.hand for w in worlds}
    if len(hands) != 1:
        return None
    hand = hands.pop()
    return None if any(c == UNKNOWN_CARD for c, _ in hand) else expand_counts(hand)


# ---------------------------------------------------------------------------
# Drawing and conditioning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Outcome:
    p: Fraction
    known: CardCounts
    unknown: int
    draw: DrawPileKnowledge
    discard: DiscardKnowledge
    reshuffled: bool


def _outcomes(world: World, n: int) -> list[_Outcome]:
    if n == 0:
        return [_Outcome(Fraction(1), (), 0, world.draw, world.discard, False)]
    out = []
    for o in enumerate_draws(world.draw, world.discard, n):
        c = Counter(o.cards)
        unknown = c.pop(UNKNOWN_CARD, 0) + o.owed  # owed: drawn from a pile of unknown size
        out.append(_Outcome(o.p_exact, counts(c), unknown, o.draw_after, o.discard_after, o.reshuffled))
    return out


def draw_unobserved(worlds: tuple[World, ...], n: int, max_worlds: int) -> tuple[tuple[World, ...], bool]:
    """Draw ``n`` unseen cards into the hand in every world (branching exactly)."""
    collapsed = False
    if sum(count_draw_outcomes(w.draw, w.discard, n) for w in worlds) > ENUMERATION_LIMIT:
        worlds, collapsed = (collapse(worlds),), True
    new = []
    for w in worlds:
        for o in _outcomes(w, n):
            hand = w.hand_counter + Counter(dict(o.known)) + Counter({UNKNOWN_CARD: o.unknown})
            new.append(World(w.weight * o.p, o.draw, o.discard, counts(hand)))
    worlds, capped = cap(normalise(new), max_worlds)
    return worlds, collapsed or capped


def observe_draw(worlds: tuple[World, ...], n: int, observed: Counter, whole_hand: bool,
                 max_worlds: int) -> tuple[tuple[World, ...], bool, bool]:
    """Condition on seeing ``observed``: the ``n`` drawn cards, or (``whole_hand``) the whole hand after drawing.

    Posterior weight = prior weight x P(this multiset | world); worlds that cannot produce it are dropped.
    A world where "?" cards may be among the observed ones (in the hand, the pile drawn from or the pile
    drawn before a reshuffle) has no such likelihood: it keeps only what is certain
    (``draw_unidentified``) and the worlds go through :func:`condition`.
    Returns (worlds, collapsed for ``max_worlds``, "?" cards may have been observed).
    Raises :class:`Contradiction` if no world can produce the observation (whatever its "?" cards are).
    """
    total = sum(observed.values())
    new, wildcard = [], False
    for w in worlds:
        base = w.hand_counter if whole_hand else Counter()
        base_unknown = base.pop(UNKNOWN_CARD, 0)
        hand = counts(observed if whole_hand else w.hand_counter + observed)
        fits = []
        for o in _outcomes(w, n):
            known = base + Counter(dict(o.known))
            if sum(known.values()) + base_unknown + o.unknown == total and not known - observed:
                fits.append(o)
        if not fits:
            continue
        if base_unknown or any(o.unknown for o in fits):
            wildcard = True
            o = draw_unidentified(w.draw, w.discard, n, observed - base)
            new.append(World(w.weight, o.draw_after, o.discard_after, hand))
        else:
            new.extend(World(w.weight * o.p, o.draw, o.discard, hand) for o in fits)
    if not new:
        raise Contradiction(_explain(worlds, n, observed, whole_hand))
    worlds, _ = condition(new, wildcard)
    worlds, capped = cap(worlds, max_worlds)
    return worlds, capped, wildcard


def _explain(worlds: tuple[World, ...], n: int, observed: Counter, whole_hand: bool) -> str:
    sizes: set[int] = set()
    most: Counter = Counter()
    for w in worlds:
        base = w.hand_counter if whole_hand else Counter()
        base_unknown = base.pop(UNKNOWN_CARD, 0)
        for o in _outcomes(w, n):
            known = base + Counter(dict(o.known))
            unknown = base_unknown + o.unknown
            sizes.add(sum(known.values()) + unknown)
            for c in observed:
                most[c] = max(most[c], known[c] + unknown)
    what = "hand" if whole_hand else "draw"
    total = sum(observed.values())
    if total not in sizes:
        return f"observed {total} card(s) but the {what} must hold {' or '.join(map(str, sorted(sizes)))}"
    over = [f"{observed[c]} {c} (at most {most[c]} possible)" for c in sorted(observed) if observed[c] > most[c]]
    if over:
        return f"impossible card counts in the observed {what}: " + ", ".join(over)
    piles = (f"the deduced draw pile is {describe_counts(worlds[0].zone_counter('draw_pile') or Counter())}"
             if len(worlds) == 1 else f"none of {len(worlds)} possible hidden states allows it")
    return f"no possible {what} gives {describe_counts(observed)}; {piles}"


def take_from_hand(worlds: tuple[World, ...], cards: Counter) -> tuple[tuple[World, ...], bool]:
    """Remove ``cards`` from the hand, dropping worlds whose hand cannot contain them ("?" may match:
    see :func:`condition`, whose result this is)."""
    new, wildcard = [], False
    for w in worlds:
        h = w.hand_counter
        deficit = sum(max(0, n - h[c]) for c, n in cards.items())
        if deficit > h[UNKNOWN_CARD]:
            continue
        wildcard = wildcard or deficit > 0
        h.subtract(Counter({c: min(n, h[c]) for c, n in cards.items()}))
        h[UNKNOWN_CARD] -= deficit
        new.append(replace(w, hand=counts(h)))
    if not new:
        raise Contradiction(f"the hand cannot contain {describe_counts(cards)}")
    return condition(new, wildcard)


def take_from_discard(worlds: tuple[World, ...], card: str) -> tuple[tuple[World, ...], bool]:
    """Remove one ``card`` from the discard pile, dropping worlds where it cannot be there (see :func:`condition`)."""
    new, wildcard = [], False
    for w in worlds:
        x = w.discard
        if x.size_known and x.counter[card] > 0:
            new.append(replace(w, discard=DiscardKnowledge(x.counter - Counter({card: 1}), x.unknown, True, x.source)))
            continue
        if not x.size_known:
            new.append(w)
        elif x.unknown > 0:
            new.append(replace(w, discard=DiscardKnowledge(x.cards, x.unknown - 1, True, x.source)))
        else:
            continue
        wildcard = True
    if not new:
        raise Contradiction(f"there is no {card} in the discard pile")
    return condition(new, wildcard)


# ---------------------------------------------------------------------------
# Predictive distributions
# ---------------------------------------------------------------------------

def outcome_count(worlds: Iterable[World], n: int) -> int:
    return sum(count_draw_outcomes(w.draw, w.discard, n) for w in worlds)


def predictive(worlds: Iterable[World], n: int) -> tuple[dict[tuple[str, ...], Fraction], Fraction]:
    """Exact distribution of the next ``n`` drawn cards (mixture over worlds) and P(reshuffle)."""
    dist: dict[tuple[str, ...], Fraction] = {}
    p_reshuffle = Fraction(0)
    for w in worlds:
        for o in enumerate_draws(w.draw, w.discard, n):
            cards = tuple(sorted(o.cards + (UNKNOWN_CARD,) * o.owed))
            p = w.weight * o.p_exact
            dist[cards] = dist.get(cards, Fraction(0)) + p
            if o.reshuffled:
                p_reshuffle += p
    return dist, p_reshuffle


def sample_predictive(worlds: tuple[World, ...], n: int, samples: int, seed: int
                      ) -> tuple[dict[tuple[str, ...], float], float]:
    """Monte Carlo fallback of :func:`predictive` (reproducible for a seed; probabilities are estimates)."""
    rng = random.Random(seed)
    weights = [float(w.weight) for w in worlds]
    tally: Counter = Counter()
    reshuffles = 0
    for _ in range(samples):
        w = rng.choices(worlds, weights)[0]
        o = sample_draw(w.draw, w.discard, n, rng)
        tally[tuple(sorted(o.cards + (UNKNOWN_CARD,) * o.owed))] += 1
        reshuffles += o.reshuffled
    return {k: v / samples for k, v in tally.items()}, reshuffles / samples


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BeliefState:
    """Observed game state plus weighted worlds for my hidden zones.

    ``game`` holds everything observed or rule-derived (hand only when determined, in play,
    resources, Supply, owned cards, trash, turn, phase); its draw/discard fields are unset -
    those zones live in ``worlds``. ``flags`` list reasons the history is incomplete
    (as opposed to inconsistent, which raises).
    """

    game: GameState
    worlds: tuple[World, ...]
    flags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    hand_observed: bool = False
    unverified_spend: int = 0  # coins spent this turn on purchases made while the hand was undetermined

    @property
    def turn(self) -> int | None:
        return self.game.turn_number

    @property
    def phase(self) -> str:
        return self.game.turn.phase.value

    @property
    def pending_draws(self) -> int:
        return self.game.turn.pending_draws

    @property
    def in_play(self) -> tuple[str, ...]:
        return tuple(self.game.turn.in_play)

    @property
    def trash(self) -> tuple[str, ...]:
        return tuple(self.game.trash)

    @property
    def owned(self) -> dict[str, int] | None:
        return None if self.game.owned_cards is None else dict(self.game.owned_cards)

    def determined_hand(self) -> list[str] | None:
        return determined_hand(self.worlds)

    def is_deterministic(self, zone: str) -> bool:
        """True if ``zone``'s composition is the same in every world and has no "?" cards."""
        if zone not in ZONES:
            raise ValueError(f"zone must be one of {', '.join(ZONES)}")
        values = [w.zone_counter(zone) for w in self.worlds]
        first = values[0]
        return first is not None and not first[UNKNOWN_CARD] and all(v == first for v in values)

    def zone_status(self, zone: str) -> str:
        """"observed" | "deduced" (certain) | "probabilistic" | "unknown" (size or some identities unknown)."""
        values = [w.zone_counter(zone) for w in self.worlds]
        if any(v is None or v[UNKNOWN_CARD] for v in values):
            return "unknown"
        if not self.is_deterministic(zone):
            return "probabilistic"
        return "observed" if zone == "hand" and self.hand_observed else "deduced"

    def zone_summary(self, zone: str) -> dict[str, Any]:
        sure = collapse(self.worlds)
        c = sure.zone_counter(zone)
        out: dict[str, Any] = {"status": self.zone_status(zone), "size": None if c is None else sum(c.values())}
        if c is not None:
            out["certain"] = {k: n for k, n in sorted(c.items()) if k != UNKNOWN_CARD and n}
            out["uncertain_count"] = c[UNKNOWN_CARD]  # unknown identity, or differs between worlds
        if zone == "draw_pile" and sure.draw.size_known:
            out["top"] = list(sure.draw.top)
            out["information_level"] = sure.draw.level.value
        if out["status"] == "probabilistic":
            expected: dict[str, Fraction] = {}
            for w in self.worlds:
                for k, n in (w.zone_counter(zone) or Counter()).items():
                    expected[k] = expected.get(k, Fraction(0)) + w.weight * n
            out["expected"] = {k: fraction_str(v) for k, v in sorted(expected.items()) if v}
        return out

    @property
    def information_level(self) -> str:
        """Draw-pile information level of the (collapsed) belief, as the planner would see it."""
        return (self.worlds[0] if len(self.worlds) == 1 else collapse(self.worlds)).draw.level.value

    def to_dict(self) -> dict[str, Any]:
        t, g = self.game.turn, self.game
        return {
            "turn": self.turn, "phase": self.phase, "pending_draws": self.pending_draws,
            "observed": {
                "hand": sorted(t.hand) if self.hand_observed and self.determined_hand() is not None else None,
                "in_play": list(t.in_play), "actions": t.actions, "buys": t.buys, "coins": t.coins,
                "bought_this_turn": list(t.bought), "unresolved": list(t.unresolved),
                "supply": dict(g.supply), "trash": sorted(g.trash),
                "owned_cards": None if g.owned_cards is None else dict(sorted(g.owned_cards.items())),
            },
            "inferred": {
                **{zone: self.zone_summary(zone) for zone in ZONES},
                "worlds": len(self.worlds), "information_level": self.information_level,
            },
            "flags": list(self.flags), "notes": list(self.notes),
        }

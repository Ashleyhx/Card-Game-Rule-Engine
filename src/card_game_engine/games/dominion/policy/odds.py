"""Next-shuffle price-point odds: how often a 5-card hand from a deck reaches $5 / $6 / $8.

A *steady-state* approximation used to compare buys: the whole deck is shuffled fresh and one
5-card hand is drawn. The real draw order (cards already drawn this shuffle, the discard pile) is
ignored on purpose - the question is "what does this card do to my deck's typical hand?".

The hand is played with a simple deterministic rule built only from the rule engine's card data:

1. Non-terminal Actions with something to give (+Cards, +$, a village's extra Actions, Merchant)
   are played first, while you have an Action. Cellar and Throne Room are never played.
2. Then, with an Action left, the terminal with the highest immediate value
   ``+$ + (+Cards) x (money density of the whole deck)`` (Library draws until 7 cards in hand,
   setting nothing aside; Moneylender counts as +$2 if the deck has Copper). Repeat while Actions last.
3. All Treasures are played; each Merchant played adds $1 when a Silver is in hand; each Moneylender
   played trashes a Copper from the final hand, if there is one, for +$3 (net +$2).

Choice/manual effects (Chapel, Cellar, Remodel, Mine, Workshop, Artisan, Harbinger's and Sentry's
choices, Vassal's discard, Poacher's discard, attacks, Bureaucrat's/Bandit's gains) add no coins.

Exactness: draws are grouped multisets (``planning.draws.grouped_draws``, hypergeometric weights) and
every probability is an exact Fraction. Cards the rule never looks at individually (Treasures,
Victory cards, Curses, Actions it never plays) are drawn as one lumped "other" label; by
exchangeability the "other" cards that end up in hand are a uniformly random subset of the "other"
pool, so their coins are enumerated once per subset size (``planning.draws.enumerate_draws``).
Actions with identical play behaviour (e.g. Moat and Witch: terminal +2 Cards) share a label.
Past ``EXACT_LIMIT`` enumerated outcomes a seeded Monte Carlo estimate is used instead.

Layering: depends only on ``rules`` and ``planning.draws`` / ``planning.analysis`` (which depend
only on ``rules``), so ``policy`` can use it although ``planning.evaluation`` reads ``policy``
weights: the module graph stays acyclic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import factorial
from operator import add, sub
from typing import Any, Mapping

from ..planning.analysis import EXACT_LIMIT, Number, amount, number
from ..planning.draws import enumerate_draws, grouped_draws
from ..rules.cards import CardDef, CardRegistry
from ..rules.zones import DiscardKnowledge, DrawPileKnowledge

PRICE_POINTS = (5, 6, 8)
HAND_SIZE = 5
MC_HANDS = 2_000   # sampled shuffles when the exact enumeration is too large
MC_SEED = 0
DEAD = "-"         # a non-Treasure in the "other" pool
_CACHE_LIMIT = 2_048


@dataclass(frozen=True)
class ShuffleOdds:
    """Odds for one deck composition. Values are Fractions when ``method == "exact"``."""

    deck_size: int
    p_at_least: dict[int, Number]  # price point -> P(hand reaches it)
    expected_coins: Number
    method: str  # "exact" | "monte_carlo"

    def to_dict(self) -> dict[str, Any]:
        return {
            "deck_size": self.deck_size,
            "method": self.method,
            "p_at_least": {str(k): number(v) for k, v in self.p_at_least.items()},
            "expected_coins": amount(self.expected_coins),
        }


class _TooLarge(Exception):
    pass


@lru_cache(maxsize=1 << 16)
def _grouped(pile: tuple[int, ...], m: int) -> list[tuple[tuple[int, ...], int]]:
    """Multisets of ``m`` cards from ``pile`` (counts per label) with weight ``prod C(n_i, k_i) * m!``."""
    kinds = [i for i, n in enumerate(pile) if n]
    out = []
    for ks, w in grouped_draws(tuple(pile[i] for i in kinds), m):
        drawn = [0] * len(pile)
        for i, k in zip(kinds, ks):
            drawn[i] = k
        out.append((tuple(drawn), w * factorial(m)))
    return out


def plays(card: CardDef) -> bool:
    """Whether the play rule ever plays ``card`` (otherwise it is just a dead card in hand)."""
    e = card.effects
    if not card.is_action:
        return False
    if card.is_terminal:
        return bool(e.cards or e.coins or card.special in ("moneylender", "library"))
    return e.actions >= 1 and bool(e.cards or e.coins or e.actions >= 2 or card.special == "merchant")


class ShuffleOddsModel:
    """Computes (and memoises per deck composition) :class:`ShuffleOdds` with the play rule above."""

    def __init__(self, registry: CardRegistry):
        self.registry = registry
        self._cache: dict[tuple, ShuffleOdds] = {}
        # Play-out end states, shared by compositions that differ only inside the "other" pool
        # (e.g. the deck + Silver, + Gold, + Province, + Workshop all share one).
        self._ends: dict[tuple, dict[tuple, int]] = {}

    def odds(self, deck: Mapping[str, int]) -> ShuffleOdds:
        key = tuple(sorted((self.registry.resolve(c), n) for c, n in deck.items() if n > 0))
        if key not in self._cache:
            if len(self._cache) >= _CACHE_LIMIT:
                self._cache.clear()
                self._ends.clear()
            self._cache[key] = _Deck({c: self.registry.get(c) for c, _ in key}, dict(key), self._ends).odds()
        return self._cache[key]


class _Deck:
    """One composition. Hands and piles are count tuples over the labels
    ``[Actions the rule plays..., other, dead Action]``."""

    def __init__(self, defs: dict[str, CardDef], deck: dict[str, int], shared_ends: dict[tuple, dict[tuple, int]]):
        self.size = sum(deck.values())
        self.density = sum(d.treasure_value * deck[c] for c, d in defs.items()) / self.size if self.size else 0.0
        self.has_copper = deck.get("Copper", 0) > 0
        self.actions: list[CardDef] = []
        by_signature: dict[tuple, int] = {}
        counts: dict[int, int] = {}
        self.other: dict[str, int] = {}   # the lumped pool, by Treasure name (or DEAD)
        for name in sorted(deck):
            d = defs[name]
            if plays(d):
                e = d.effects
                i = by_signature.setdefault((e.cards, e.actions, e.coins, d.special), len(self.actions))
                if i == len(self.actions):
                    self.actions.append(d)
                counts[i] = counts.get(i, 0) + deck[name]
            else:
                label = name if d.is_treasure else DEAD
                self.other[label] = self.other.get(label, 0) + deck[name]
        n = len(self.actions)
        self.other_i = n
        self.pile = tuple([counts.get(i, 0) for i in range(n)] + [sum(self.other.values()), 0])
        self.values = {c: defs[c].treasure_value for c in self.other if c != DEAD}
        self.nt = [i for i, c in enumerate(self.actions) if not c.is_terminal]
        self.inert = [i for i, c in enumerate(self.actions) if c.is_terminal] + [self.other_i]  # see _ends
        self.inert_pool = tuple(self.pile[i] for i in self.inert)
        self._budget = EXACT_LIMIT
        self._shared_ends = shared_ends

    def odds(self) -> ShuffleOdds:
        try:
            dist = self._exact()
            method = "exact"
        except _TooLarge:
            dist = self._sampled()
            method = "monte_carlo"
        return ShuffleOdds(
            deck_size=self.size,
            p_at_least={t: sum((p for c, p in dist.items() if c >= t), Fraction(0)) for t in PRICE_POINTS},
            expected_coins=sum((c * p for c, p in dist.items()), Fraction(0)),
            method=method,
        )

    # ------------------------------------------------------------------ play rule
    def _next_plays(self, hand: tuple[int, ...], actions: int) -> list[int]:
        """All non-terminals in hand (all get played, so their order is irrelevant), else the best terminal."""
        if actions < 1:
            return []
        held = [i for i in range(len(self.actions)) if hand[i]]
        batch = [i for i in held if not self.actions[i].is_terminal for _ in range(hand[i])]
        if batch:
            return batch
        best, best_value = [], 0.0
        for i in held:
            value = self._terminal_value(i, sum(hand))
            if value > best_value:
                best, best_value = [i], value
        return best

    def _terminal_value(self, i: int, hand_size: int) -> float:
        c = self.actions[i]
        n_draw = max(0, 8 - hand_size) if c.special == "library" else c.effects.cards
        return c.effects.coins + n_draw * self.density + (2 if c.special == "moneylender" and self.has_copper else 0)

    def _process_key(self) -> tuple:
        """Everything the play-out depends on: labels, pile and the terminal preferences per hand size
        (the only place the money density enters)."""
        labels = tuple((c.effects, c.special) for c in self.actions)
        prefs = tuple(tuple(i for v, i in sorted((-self._terminal_value(i, h), i) for i, c in enumerate(self.actions)
                                                 if c.is_terminal) if v < 0) for h in range(9))
        return labels, self.pile, prefs

    def _apply(self, hand: tuple[int, ...], batch: list[int]) -> tuple[list[int], int, int]:
        """Play ``batch``: (hand after, cards to draw, change in Actions)."""
        after = list(hand)
        n_draw = delta = 0
        for i in batch:
            after[i] -= 1
            n_draw += self.actions[i].effects.cards
            delta += self.actions[i].effects.actions - 1
        if self.actions[batch[0]].special == "library":
            n_draw = max(0, 7 - sum(after))
        return after, n_draw, delta

    def _coins(self, others: tuple[str, ...], bonus: int, merchants: int, lenders: int) -> int:
        """Final $: +$ from Actions, Treasures among the "other" cards, Merchant and Moneylender bonuses."""
        coins = bonus + sum(self.values.get(c, 0) for c in others)
        if merchants and "Silver" in others:
            coins += merchants
        return coins + 2 * min(lenders, others.count("Copper"))  # trash a Copper (-$1) for +$3

    # ------------------------------------------------------------------ exact
    def _draw(self, pile: tuple[int, ...], m: int) -> list[tuple[tuple[int, ...], int]]:
        outcomes = _grouped(pile, m)
        self._budget -= len(outcomes)
        if self._budget < 0:
            raise _TooLarge
        return outcomes

    def _ends(self) -> dict[tuple, int]:
        """End states ``(other cards in hand, +$, Merchants, Moneylenders played) -> W``, ``P = W / (N! I!)``.

        Forward pass over states ``(hand, pile, Actions)``; each transition shrinks ``hand + 2 * pile``,
        so states are visited in decreasing order of it. A state's mass M means P = M / perm(N, D)
        (D cards drawn so far). Drawing m more cards has P = prod C(n_i, k_i) m! / perm(N - D, m) and
        perm(N, D) perm(N - D, m) = perm(N, D + m), so masses stay integers: M' += M prod C(n_i, k_i) m!.

        While non-terminals are being played (all of them get played), terminals and "other" cards are
        inert and lumped together (mode 0); once none is left in hand, the r inert cards in hand are
        a uniformly random subset of the I inert cards (exchangeability), P = prod C r! (I - r)! / I!,
        and the pass continues per card label (mode 1, masses over perm(N, D) I!). At an end with n cards
        left in the pile, P = M n! / (N! I!).
        """
        nt, inert, inert_pool = self.nt, self.inert, self.inert_pool
        n_inert = sum(inert_pool)
        lumped = tuple(self.pile[i] for i in nt) + (n_inert,)
        start = min(HAND_SIZE, self.size)
        levels: dict[int, dict[tuple, int]] = {}
        level = levels.setdefault(start + 2 * (self.size - start), {})
        for drawn, w in self._draw(lumped, start):
            level[0, drawn, tuple(map(sub, lumped, drawn)), 1] = w
        ends: dict[tuple, int] = {}
        while levels:
            for (mode, hand, pile, actions), mass in levels.pop(max(levels)).items():
                if mode == 0 and any(hand[:-1]):  # play every non-terminal in hand
                    n_draw = sum(n * self.actions[i].effects.cards for i, n in zip(nt, hand))
                    actions += sum(n * (self.actions[i].effects.actions - 1) for i, n in zip(nt, hand))
                    after = (0,) * len(nt) + hand[-1:]
                    m = min(n_draw, sum(pile))
                    level = levels.setdefault(hand[-1] + 2 * sum(pile) - m, {})
                    for drawn, w in self._draw(pile, m):
                        key = (0, tuple(map(add, after, drawn)), tuple(map(sub, pile, drawn)), actions)
                        level[key] = level.get(key, 0) + mass * w
                elif mode == 0:  # identify the inert cards in hand
                    r = hand[-1]
                    level = levels.setdefault(r + 2 * sum(pile), {})
                    rest = factorial(n_inert - r)
                    for sub_hand, w in self._draw(inert_pool, r):
                        full_hand, full_pile = [0] * len(self.pile), [0] * len(self.pile)
                        for j, i in enumerate(nt):
                            full_pile[i] = pile[j]
                        for j, i in enumerate(inert):
                            full_hand[i], full_pile[i] = sub_hand[j], inert_pool[j] - sub_hand[j]
                        key = (1, tuple(full_hand), tuple(full_pile), actions)
                        level[key] = level.get(key, 0) + mass * w * rest
                else:
                    self._step(hand, pile, actions, mass, levels, ends)
        return ends

    def _step(self, hand: tuple[int, ...], pile: tuple[int, ...], actions: int, mass: int,
              levels: dict[int, dict[tuple, int]], ends: dict[tuple, int]) -> None:
        """One play (or the end) from a per-label state; see ``_ends``."""
        batch = self._next_plays(hand, actions)
        if not batch:
            end = (hand[self.other_i],) + self._bonuses(self._played(hand, pile))
            ends[end] = ends.get(end, 0) + mass * factorial(sum(pile))
            return
        after, n_draw, delta = self._apply(hand, batch)
        actions += delta
        left = sum(pile)
        m = min(n_draw, left)
        if not actions:  # nothing else gets played: only the "other" cards drawn matter
            k, bonuses = after[self.other_i], self._bonuses(self._played(hand, pile) + batch)
            n_other = pile[self.other_i]
            rest = factorial(left - m)
            for drawn, w in self._draw((left - n_other, n_other), m):
                end = (k + drawn[1],) + bonuses
                ends[end] = ends.get(end, 0) + mass * w * rest
            return
        level = levels.setdefault(sum(after) + 2 * left - m, {})
        for drawn, w in self._draw(pile, m):
            key = (1, tuple(map(add, after, drawn)), tuple(map(sub, pile, drawn)), actions)
            level[key] = level.get(key, 0) + mass * w

    def _played(self, hand: tuple[int, ...], pile: tuple[int, ...]) -> list[int]:
        """Per-label count of Actions played so far: whatever left the pile and is not in hand."""
        return [self.pile[i] - hand[i] - pile[i] for i in range(len(self.actions))]

    def _bonuses(self, played: list[int]) -> tuple[int, int, int]:
        """(+$, Merchants, Moneylenders) of the played Actions; ``played`` = per-label counts, then extra labels."""
        bonus = merchants = lenders = 0
        counts = played[:len(self.actions)]
        for i in played[len(self.actions):]:
            counts[i] += 1
        for c, n in zip(self.actions, counts):
            bonus += n * c.effects.coins
            merchants += n * (c.special == "merchant")
            lenders += n * (c.special == "moneylender")
        return bonus, merchants, lenders

    def _exact(self) -> dict[int, Fraction]:
        total = factorial(self.size) * factorial(sum(self.inert_pool))
        pool = DrawPileKnowledge(rest=self.other)
        subsets: dict[int, list] = {}
        dist: dict[int, Fraction] = {}
        key = self._process_key()
        if key not in self._shared_ends:
            self._shared_ends[key] = self._ends()
        for (k, *bonuses), weight in self._shared_ends[key].items():
            if k not in subsets:
                subsets[k] = enumerate_draws(pool, DiscardKnowledge(), k)
            p = Fraction(weight, total)
            for o in subsets[k]:
                c = self._coins(o.cards, *bonuses)
                dist[c] = dist.get(c, 0) + p * o.p_exact
        return dist

    # ------------------------------------------------------------------ Monte Carlo fallback
    def _sampled(self) -> dict[int, float]:
        rng = random.Random(MC_SEED)
        cards = [i for i, n in enumerate(self.pile) for _ in range(n)]
        others = [c for c, n in sorted(self.other.items()) for _ in range(n)]
        tally: dict[int, int] = {}
        for _ in range(MC_HANDS):
            order = rng.sample(cards, len(cards))
            hand, pos, actions, played = [0] * len(self.pile), HAND_SIZE, 1, [0] * len(self.actions)
            for i in order[:HAND_SIZE]:
                hand[i] += 1
            while batch := self._next_plays(tuple(hand), actions):
                hand, n_draw, delta = self._apply(tuple(hand), batch)
                played += batch
                actions += delta
                for i in order[pos:pos + n_draw]:
                    hand[i] += 1
                pos += n_draw
            c = self._coins(tuple(rng.sample(others, hand[self.other_i])), *self._bonuses(played))
            tally[c] = tally.get(c, 0) + 1
        return {c: n / MC_HANDS for c, n in tally.items()}

"""Probability queries about future draws: "how many <target> among the next n cards?".

A :class:`Target` is one card ("Copper") or one card type ("Treasure"). The functions here work on
weighted ``(weight, draw pile, discard pile)`` sources (e.g. the worlds of a belief state) and never
re-implement draw mechanics: every card is relabelled as a target match, a non-match or "?" and the
relabelled piles go through ``enumerate_draws`` (or ``sample_draw``), so the reshuffle rule, short
draws and piles of unknown size behave exactly as everywhere else. Relabelling keeps the distribution
of the match count exact while making the number of outcomes tiny.

"?" cards never count as matches. When one can be drawn, results are bounds: the known-only values
are lower bounds and the ``upper`` values treat every drawn "?" as a match, so the truth lies in
between (for ``P(X >= k)`` and ``E[X]``; reversed for ``P(X = 0)``).
"""

from __future__ import annotations

import difflib
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from ..rules.cards import CardRegistry
from ..rules.zones import UNKNOWN_CARD, DiscardKnowledge, DrawPileKnowledge
from .draws import count_draw_outcomes, enumerate_draws, sample_draw

TARGET_TYPES = ("Treasure", "Action", "Victory", "Curse", "Attack", "Reaction")
EXACT_LIMIT = 200_000   # relabelled draw outcomes enumerated before falling back to Monte Carlo
MC_SAMPLES = 20_000

HIT, MISS = "+", "-"     # relabelled cards: a target match / a known non-match ("?" stays "?")

Source = tuple[Fraction, DrawPileKnowledge, DiscardKnowledge]
Number = Fraction | float


def number(v: Number | None) -> dict[str, Any]:
    """``{"p_exact": "a/b", "p": float}`` (``p_exact`` None for an estimate; both None when unknown)."""
    if v is None:
        return {"p_exact": None, "p": None}
    return {"p_exact": f"{v.numerator}/{v.denominator}", "p": float(v)} if isinstance(v, Fraction) \
        else {"p_exact": None, "p": float(v)}


def amount(v: Number | None) -> dict[str, Any]:
    """``{"exact": "a/b", "value": float}`` for an expectation (``exact`` None for an estimate or unknown)."""
    d = number(v)
    return {"exact": d["p_exact"], "value": d["p"]}


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    """A card identity (``kind == "card"``) or a card type (``kind == "type"``); ``cards`` = matching names."""

    kind: str
    name: str
    cards: frozenset[str]

    def matches(self, card: str) -> bool:
        """True if ``card`` is a target card; "?" never matches."""
        return card != UNKNOWN_CARD and card in self.cards

    def describe(self) -> str:
        return self.name if self.kind == "card" else f"{self.name} card(s)"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "cards": sorted(self.cards)}


def card_target(name: str) -> Target:
    return Target("card", name, frozenset([name]))


def type_target(type_name: str, registry: CardRegistry) -> Target:
    return Target("type", type_name, frozenset(c.name for c in registry.all() if type_name in c.types))


def parse_target(text: str, registry: CardRegistry) -> Target:
    """A card name or a type name (case-insensitive; card names win, e.g. "Curse" is the card)."""
    key = (text or "").strip()
    if key in registry:
        return card_target(registry.resolve(key))
    for t in TARGET_TYPES:
        if key.casefold() == t.casefold():
            return type_target(t, registry)
    types = {t.casefold(): t for t in TARGET_TYPES}
    hints = registry.suggest(key) + [types[m] for m in difflib.get_close_matches(key.casefold(), list(types), n=2)]
    hint = f" Did you mean: {', '.join(dict.fromkeys(hints))}?" if hints else ""
    raise ValueError(f"Unknown target '{text}': give a card name or a type ({', '.join(TARGET_TYPES)}).{hint}")


# ---------------------------------------------------------------------------
# Relabelling
# ---------------------------------------------------------------------------

def _label(card: str, target: Target) -> str:
    return UNKNOWN_CARD if card == UNKNOWN_CARD else HIT if target.matches(card) else MISS


def _relabel_counts(items: Iterable[tuple[str, int]], target: Target) -> dict[str, int]:
    out: dict[str, int] = {}
    for card, n in items:
        lab = _label(card, target)
        out[lab] = out.get(lab, 0) + n
    return out


def _relabel(draw: DrawPileKnowledge, discard: DiscardKnowledge, target: Target
             ) -> tuple[DrawPileKnowledge, DiscardKnowledge]:
    d = (DrawPileKnowledge(tuple(_label(c, target) for c in draw.top), _relabel_counts(draw.rest, target),
                           draw.unknown, source="analysis")
         if draw.size_known else DrawPileKnowledge(size_known=False, source="analysis"))
    x = (DiscardKnowledge(_relabel_counts(discard.cards, target), discard.unknown, source="analysis")
         if discard.size_known else DiscardKnowledge(size_known=False, source="analysis"))
    return d, x


def _merged(sources: Iterable[Source], target: Target) -> list[tuple[Fraction, DrawPileKnowledge, DiscardKnowledge]]:
    """Relabel every source, merge identical relabelled piles, normalise weights exactly."""
    merged: dict[tuple, Fraction] = {}
    for weight, draw, discard in sources:
        if weight < 0:
            raise ValueError("source weights must be >= 0")
        if weight:
            key = _relabel(draw, discard, target)
            merged[key] = merged.get(key, Fraction(0)) + Fraction(weight)
    total = sum(merged.values(), Fraction(0))
    if total == 0:
        raise ValueError("no source with positive weight")
    return sorted(((w / total, d, x) for (d, x), w in merged.items()),
                  key=lambda s: (-s[0], repr(s[1]), repr(s[2])))


@dataclass(frozen=True)
class _Count:
    hits: int        # known target cards drawn
    unknown: int     # "?" drawn, plus draws owed to a pile of unknown size
    reshuffled: bool
    short: bool      # fewer than n cards drawn (deck and discard ran out)
    boundary: bool   # drawing hit a pile of unknown size


def _count(o) -> _Count:
    return _Count(o.cards.count(HIT), o.cards.count(UNKNOWN_CARD) + o.owed, o.reshuffled, o.exhausted,
                  o.boundary is not None)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CountDistribution:
    """Distribution of X = number of target cards among the next ``draw_count`` drawn cards.

    ``joint`` holds ``((known matches, "?" drawn), probability)``; everything else derives from it.
    Exact ``Fraction`` values unless ``method == "monte_carlo"`` (floats). ``determinable`` is False when
    some draw reaches a pile of unknown size: probabilities are then withheld (None), only ``min``/``max``.
    """

    target: Target
    draw_count: int
    joint: tuple[tuple[tuple[int, int], Number], ...]
    p_reshuffle: Number
    p_short: Number
    min: int
    max: int
    method: str                  # "exact" | "monte_carlo"
    determinable: bool = True
    samples: int | None = None
    seed: int | None = None

    def _sum(self, keep) -> Number | None:
        if not self.determinable:
            return None
        zero: Number = Fraction(0) if self.method == "exact" else 0.0
        return sum((p for (k, u), p in self.joint if keep(k, u)), zero)

    @property
    def bounded(self) -> bool:
        """True if a "?" card can be drawn: values are then [lower, upper] bounds."""
        return any(u for (_, u), p in self.joint if p)

    @property
    def deterministic(self) -> bool:
        return self.determinable and self.min == self.max and not self.bounded

    def p_exactly(self, k: int) -> Number | None:
        """P(known matches == k) - exact when not ``bounded``."""
        return self._sum(lambda h, u: h == k)

    def p_at_least(self, k: int) -> Number | None:
        """P(known matches >= k) - exact, or the lower bound of P(X >= k) when ``bounded``."""
        return self._sum(lambda h, u: h >= k)

    def p_zero(self) -> Number | None:
        return self.p_exactly(0)

    def upper_p_exactly(self, k: int) -> Number | None:
        """P(known matches + "?" drawn == k): every drawn "?" counted as a match."""
        return self._sum(lambda h, u: h + u == k)

    def upper_p_at_least(self, k: int) -> Number | None:
        """Upper bound of P(X >= k) (every drawn "?" counted as a match)."""
        return self._sum(lambda h, u: h + u >= k)

    def range_at_least(self, k: int) -> tuple[Number, Number] | None:
        """``[lower, upper]`` of P(X >= k)."""
        return None if not self.determinable else (self.p_at_least(k), self.upper_p_at_least(k))  # type: ignore

    def range_exactly(self, k: int) -> tuple[Number, Number] | None:
        """``[lower, upper]`` of P(X == k): certainly k (no "?") vs k reachable by the "?" cards."""
        if not self.determinable:
            return None
        return self._sum(lambda h, u: h == k and not u), self._sum(lambda h, u: h <= k <= h + u)  # type: ignore

    @property
    def expected(self) -> Number | None:
        """E[known matches] - exact, or the lower bound of E[X] when ``bounded``."""
        return None if not self.determinable else self._sum_weighted(lambda h, u: h)

    @property
    def upper_expected(self) -> Number | None:
        return None if not self.determinable else self._sum_weighted(lambda h, u: h + u)

    def _sum_weighted(self, value) -> Number:
        zero: Number = Fraction(0) if self.method == "exact" else 0.0
        return sum((p * value(k, u) for (k, u), p in self.joint), zero)

    def unknown_distribution(self) -> tuple[tuple[int, Number], ...]:
        """P("?" drawn == u) for every possible u."""
        out: dict[int, Number] = {}
        for (_, u), p in self.joint:
            out[u] = out.get(u, 0) + p
        return tuple(sorted(out.items()))

    def to_dict(self) -> dict[str, Any]:
        n, det = self.draw_count, self.determinable
        out: dict[str, Any] = {
            "target": self.target.to_dict(), "draw_count": n, "method": self.method,
            "determinable": det, "bounded": self.bounded, "deterministic": self.deterministic,
            "guaranteed": {"min": self.min, "max": self.max},
            "distribution": [{"k": k, **number(self.p_exactly(k))} for k in range(n + 1)] if det else None,
            "p_zero": number(self.p_zero()), "p_at_least_one": number(self.p_at_least(1)),
            "expected": amount(self.expected),
            "p_reshuffle": number(self.p_reshuffle if det else None),
            "p_short_draw": number(self.p_short if det else None),
            "unknown_drawn": [{"u": u, **number(p)} for u, p in self.unknown_distribution()] if det else None,
            "upper": None,
        }
        if det and self.bounded:
            out["upper"] = {
                "distribution": [{"k": k, **number(self.upper_p_exactly(k))} for k in range(n + 1)],
                "p_at_least_one": number(self.upper_p_at_least(1)), "expected": amount(self.upper_expected),
            }
            out["bounds"] = {
                "p_zero": [number(1 - self.upper_p_at_least(1)), number(self.p_zero())],  # type: ignore[operator]
                "p_at_least_one": [number(v) for v in self.range_at_least(1)],  # type: ignore[union-attr]
                "expected": [amount(self.expected), amount(self.upper_expected)],
            }
        if self.method == "monte_carlo":
            out["samples"], out["seed"] = self.samples, self.seed
        return out


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def _scan(merged: list[Source], n: int) -> tuple[int, int, bool]:
    """Exact (min, max, hits a pile of unknown size) over every outcome of the relabelled sources."""
    counts = [_count(o) for _, d, x in merged for o in enumerate_draws(d, x, n)]
    return (min(c.hits for c in counts), max(c.hits + c.unknown for c in counts),
            any(c.boundary for c in counts))


def guaranteed_range(sources: Iterable[Source], n: int, target: Target) -> tuple[int, int]:
    """(min, max) of X over every outcome with positive probability; max counts drawn "?" as matches."""
    lo, hi, _ = _scan(_merged(sources, target), n)
    return lo, hi


def draw_count_distribution(sources: Iterable[Source], n: int, target: Target, limit: int = EXACT_LIMIT,
                            samples: int = MC_SAMPLES, seed: int = 0) -> CountDistribution:
    """Distribution of the number of ``target`` cards among the next ``n`` draws (mixture over sources).

    Exact (``enumerate_draws``) unless the relabelled outcome count exceeds ``limit``; then a seeded Monte
    Carlo estimate (``sample_draw``) labelled ``monte_carlo``. ``min``/``max`` are always exact.
    """
    if n < 0:
        raise ValueError("draw count must be >= 0")
    merged = _merged(sources, target)
    lo, hi, boundary = _scan(merged, n)
    joint: dict[tuple[int, int], Number] = {}
    reshuffle: Number
    short: Number
    if sum(count_draw_outcomes(d, x, n) for _, d, x in merged) <= limit:
        method, reshuffle, short = "exact", Fraction(0), Fraction(0)
        for w, d, x in merged:
            for o in enumerate_draws(d, x, n):
                c, p = _count(o), w * o.p_exact
                joint[(c.hits, c.unknown)] = joint.get((c.hits, c.unknown), Fraction(0)) + p
                reshuffle += p if c.reshuffled else 0
                short += p if c.short else 0
        used_samples = used_seed = None
    else:
        method, rng = "monte_carlo", random.Random(seed)
        tally: dict[tuple[int, int], int] = {}
        n_reshuffle = n_short = 0
        weights = [float(w) for w, _, _ in merged]
        for _ in range(samples):
            _, d, x = rng.choices(merged, weights)[0]
            c = _count(sample_draw(d, x, n, rng))
            tally[(c.hits, c.unknown)] = tally.get((c.hits, c.unknown), 0) + 1
            n_reshuffle += c.reshuffled
            n_short += c.short
        joint = {k: v / samples for k, v in tally.items()}
        reshuffle, short = n_reshuffle / samples, n_short / samples
        used_samples, used_seed = samples, seed
    return CountDistribution(target, n, tuple(sorted(joint.items())), reshuffle, short, lo, hi, method,
                             not boundary, used_samples, used_seed)

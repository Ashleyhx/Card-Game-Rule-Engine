"""Exact grouped draw enumeration and seeded sampling."""

import random
from collections import Counter
from fractions import Fraction
from itertools import combinations

import pytest

from card_game_engine.games.dominion.planning import (
    BOUNDARY_UNKNOWN_DRAW_PILE, count_draw_outcomes, enumerate_draws, sample_draw,
)
from card_game_engine.games.dominion.rules import DiscardKnowledge, DrawPileKnowledge

EMPTY = DiscardKnowledge()


def probs(outcomes):
    return {o.cards: o.p_exact for o in outcomes}


def brute_force(rest: dict, n: int) -> dict:
    """Reference distribution over ordered-index combinations (tests only)."""
    cards = [c for c, k in sorted(rest.items()) for _ in range(k)]
    combos = list(combinations(range(len(cards)), n))
    tally = Counter(tuple(sorted(cards[i] for i in combo)) for combo in combos)
    return {k: Fraction(v, len(combos)) for k, v in tally.items()}


def test_hand_verifiable_probabilities():
    out = enumerate_draws(DrawPileKnowledge(rest={"Copper": 3, "Estate": 2}), EMPTY, 2)
    assert probs(out) == {("Copper", "Copper"): Fraction(3, 10), ("Copper", "Estate"): Fraction(6, 10),
                          ("Estate", "Estate"): Fraction(1, 10)}
    assert sum(o.p_exact for o in out) == 1
    assert out[0].cards == ("Copper", "Estate") and out[0].probability == pytest.approx(0.6)
    after = {o.cards: o.draw_after.rest_counter for o in out}
    assert after[("Copper", "Estate")] == {"Copper": 2, "Estate": 1}


@pytest.mark.parametrize("rest,n", [({"Copper": 7, "Estate": 3}, 5), ({"A": 2, "B": 2, "C": 2}, 3),
                                    ({"Copper": 4, "Silver": 2, "Estate": 3, "Smithy": 1}, 5)])
def test_grouped_matches_brute_force(rest, n):
    out = enumerate_draws(DrawPileKnowledge(rest=rest), EMPTY, n)
    expected = brute_force(rest, n)
    assert probs(out) == expected
    assert len(out) == len(expected) == count_draw_outcomes(DrawPileKnowledge(rest=rest), EMPTY, n)
    assert sum(o.p_exact for o in out) == 1


def test_count_is_cheap_for_large_piles():
    pile = DrawPileKnowledge(rest={f"Card{i}": 3 for i in range(12)})
    assert count_draw_outcomes(pile, EMPTY, 5) == len(enumerate_draws(pile, EMPTY, 5))


def test_known_order_is_deterministic():
    out = enumerate_draws(DrawPileKnowledge(top=("Gold", "Copper", "Estate")), EMPTY, 2)
    assert len(out) == 1 and out[0].cards == ("Copper", "Gold") and out[0].p_exact == 1
    assert out[0].draw_after.top == ("Estate",)


def test_top_prefix_then_random():
    out = enumerate_draws(DrawPileKnowledge(top=("Silver",), rest={"Copper": 2, "Estate": 1}), EMPTY, 2)
    assert probs(out) == {("Copper", "Silver"): Fraction(2, 3), ("Estate", "Silver"): Fraction(1, 3)}


def test_reshuffle_mid_draw():
    discard = DiscardKnowledge(cards={"Copper": 2, "Estate": 1})
    out = enumerate_draws(DrawPileKnowledge(top=("Gold",)), discard, 3)
    assert probs(out) == {("Copper", "Estate", "Gold"): Fraction(2, 3), ("Copper", "Copper", "Gold"): Fraction(1, 3)}
    for o in out:
        assert o.reshuffled and not o.exhausted and o.boundary is None
        assert o.draw_after.size == 1 and o.discard_after.size == 0


def test_exhaustion_draws_fewer():
    out = enumerate_draws(DrawPileKnowledge(rest={"Copper": 1}), EMPTY, 3)
    assert len(out) == 1 and out[0].cards == ("Copper",) and out[0].exhausted and not out[0].reshuffled
    out = enumerate_draws(DrawPileKnowledge(rest={"Copper": 1}), DiscardKnowledge(cards={"Estate": 1}), 3)
    assert out[0].cards == ("Copper", "Estate") and out[0].exhausted and out[0].reshuffled
    assert out[0].draw_after.size == 0 and out[0].p_exact == 1


def test_unknown_cards_are_placeholders():
    out = enumerate_draws(DrawPileKnowledge(rest={"Copper": 1}, unknown=2), EMPTY, 2)
    assert probs(out) == {("?", "?"): Fraction(1, 3), ("?", "Copper"): Fraction(2, 3)}
    after = {o.cards: (o.draw_after.rest_counter, o.draw_after.unknown) for o in out}
    assert after[("?", "?")] == ({"Copper": 1}, 0)


def test_insufficient_boundary():
    out = enumerate_draws(DrawPileKnowledge(size_known=False), EMPTY, 3)
    assert len(out) == 1 and out[0].cards == () and out[0].owed == 3
    assert out[0].boundary == BOUNDARY_UNKNOWN_DRAW_PILE and out[0].p_exact == 1
    out = enumerate_draws(DrawPileKnowledge(top=("Gold",)), DiscardKnowledge(size_known=False), 3)
    assert out[0].cards == ("Gold",) and out[0].owed == 2 and out[0].reshuffled
    assert out[0].boundary == BOUNDARY_UNKNOWN_DRAW_PILE


def test_zero_draws():
    pile = DrawPileKnowledge(top=("Gold",), rest={"Copper": 2})
    out = enumerate_draws(pile, EMPTY, 0)
    assert len(out) == 1 and out[0].cards == () and out[0].draw_after == pile


def test_sample_reproducible_and_valid():
    pile, discard = DrawPileKnowledge(top=("Gold",), rest={"Copper": 4, "Estate": 3}), DiscardKnowledge(cards={"Silver": 2})
    a = [sample_draw(pile, discard, 5, random.Random(7)) for _ in range(3)]
    assert a[0] == a[1] == a[2] and a[0].probability is None
    possible = probs(enumerate_draws(pile, discard, 5))
    rng = random.Random(1)
    assert all(sample_draw(pile, discard, 5, rng).cards in possible for _ in range(50))


def test_sample_rngs_independent():
    pile = DrawPileKnowledge(rest={"Copper": 7, "Estate": 3, "Silver": 2})
    solo = random.Random(1)
    alone = [sample_draw(pile, EMPTY, 5, solo).cards for _ in range(20)]
    r1, r2 = random.Random(1), random.Random(2)
    interleaved, other = [], []
    for _ in range(20):
        interleaved.append(sample_draw(pile, EMPTY, 5, r1).cards)
        other.append(sample_draw(pile, EMPTY, 5, r2).cards)
    assert interleaved == alone
    assert other != alone

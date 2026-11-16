"""Phase 2c Part A: deck analysis and draw-probability queries (planning.analysis + inference.analysis)."""

import copy
import dataclasses
import itertools
import json
from collections import Counter
from fractions import Fraction
from math import comb

import pytest

from card_game_engine.games.dominion.inference import (
    EndTurnObservation,
    ObserveHand,
    RecordDraw,
    RecordPurchase,
    RecordTrash,
    SessionError,
    analyze_deck,
    analyze_draw,
    get_guaranteed_cards,
    infer_belief_state,
    initialize_session,
    predict_next_hand,
    record_event,
)
from card_game_engine.games.dominion.planning import (
    CountDistribution,
    Target,
    draw_count_distribution,
    guaranteed_range,
    parse_target,
)
from card_game_engine.games.dominion.rules import DiscardKnowledge, DrawPileKnowledge, load_registry

from .conftest import FIRST_GAME

C, E, S, G = "Copper", "Estate", "Silver", "Gold"
REG = load_registry()
EMPTY = DiscardKnowledge()


def src(draw, discard=EMPTY):
    return [(Fraction(1), draw, discard)]


def dist(counts: CountDistribution):
    return [counts.p_exactly(k) for k in range(counts.draw_count + 1)]


def snap(hand=(), **player):
    return {"schema_version": 1, "num_players": 2, "kingdom": FIRST_GAME, "turn_number": 5,
            "player": {"phase": "action", "hand": list(hand), **player}}


def from_snapshot(**player):
    return initialize_session({"snapshot": snap(**player)})


def rec(session, *events):
    for e in events:
        session = record_event(session, e)
    return session


# ---------------------------------------------------------------- targets

def test_parse_target_cards_types_and_errors():
    assert parse_target("copper", REG) == Target("card", C, frozenset([C]))
    t = parse_target("TREASURE", REG)
    assert t.kind == "type" and t.name == "Treasure" and {C, S, G} <= t.cards and "Estate" not in t.cards
    assert t.matches(G) and not t.matches(E) and not t.matches("?")
    assert parse_target("Curse", REG).kind == "card"  # card names win over type names
    assert parse_target("victory", REG).matches("Gardens")
    assert parse_target("attack", REG).matches("Militia") and parse_target("Reaction", REG).matches("Moat")
    with pytest.raises(ValueError, match="Did you mean: Silver"):
        parse_target("Silvr", REG)
    with pytest.raises(ValueError, match="Did you mean: Treasure"):
        parse_target("treasur", REG)
    with pytest.raises(ValueError, match="Unknown target"):
        parse_target("?", REG)
    assert parse_target("Action", REG).to_dict()["kind"] == "type"


# ---------------------------------------------------------------- hand-verifiable (pure functions)

def test_guaranteed_minimum():
    pile = DrawPileKnowledge(rest={C: 4, E: 1})
    copper, treasure = parse_target(C, REG), parse_target("Treasure", REG)
    assert guaranteed_range(src(pile), 5, copper) == (4, 4)
    assert guaranteed_range(src(pile), 3, copper) == (2, 3)
    assert guaranteed_range(src(pile), 5, treasure) == (4, 4)
    assert guaranteed_range(src(pile), 3, treasure) == (2, 3)
    d = draw_count_distribution(src(pile), 5, copper)
    assert d.deterministic and d.p_exactly(4) == 1 and d.expected == 4
    d3 = draw_count_distribution(src(pile), 3, copper)
    assert dist(d3) == [0, 0, Fraction(3, 5), Fraction(2, 5)]  # P(Estate among 3 of 5) = 3/5
    assert not d3.deterministic


def test_exact_hypergeometric_probabilities():
    d = draw_count_distribution(src(DrawPileKnowledge(rest={C: 3, E: 2})), 2, parse_target(C, REG))
    assert dist(d) == [Fraction(1, 10), Fraction(6, 10), Fraction(3, 10)]
    assert d.p_zero() == Fraction(1, 10) and d.p_at_least(1) == Fraction(9, 10) and d.p_at_least(2) == Fraction(3, 10)
    assert d.expected == Fraction(6, 5) and d.method == "exact" and not d.bounded
    assert (d.min, d.max) == (0, 2) and d.p_reshuffle == 0 and d.p_short == 0
    assert sum(dist(d)) == 1


def test_shuffle_boundaries():
    piles = src(DrawPileKnowledge(top=(G, C)), DiscardKnowledge({S: 2, E: 2}))
    for name in (G, C):
        assert guaranteed_range(piles, 4, parse_target(name, REG)) == (1, 1)
    d = draw_count_distribution(piles, 4, parse_target(S, REG))
    hyper = [Fraction(comb(2, k) * comb(2, 2 - k), comb(4, 2)) for k in range(3)]
    assert hyper == [Fraction(1, 6), Fraction(2, 3), Fraction(1, 6)]
    assert dist(d) == hyper + [0, 0] and d.p_reshuffle == 1 and d.p_short == 0
    assert d.expected == 1
    # 7 draws with only 6 cards in total: a certain short draw, every count fixed.
    for name, n in ((G, 1), (C, 1), (S, 2), (E, 2), ("Treasure", 4), ("Victory", 2)):
        d7 = draw_count_distribution(piles, 7, parse_target(name, REG))
        assert d7.p_short == 1 and d7.p_reshuffle == 1 and d7.min == d7.max == n and d7.deterministic
        assert d7.p_exactly(n) == 1


def test_mixture_and_unknown_cards_are_bounds():
    copper = parse_target(C, REG)
    mix = [(Fraction(1, 3), DrawPileKnowledge(rest={C: 2}), EMPTY), (Fraction(2, 3), DrawPileKnowledge(rest={E: 2}), EMPTY)]
    d = draw_count_distribution(mix, 1, copper)
    assert dist(d) == [Fraction(2, 3), Fraction(1, 3)]
    # Weights are normalised exactly; zero weights are ignored.
    d2 = draw_count_distribution([(Fraction(1), DrawPileKnowledge(rest={C: 2}), EMPTY),
                                  (Fraction(2), DrawPileKnowledge(rest={E: 2}), EMPTY),
                                  (Fraction(0), DrawPileKnowledge(rest={G: 2}), EMPTY)], 1, copper)
    assert dist(d2) == dist(d)
    # Gold on top, then 2 of {Copper, Copper, ?, ?}: Treasure count bounded.
    pile = src(DrawPileKnowledge(top=(G,), rest={C: 2}, unknown=2))
    t = draw_count_distribution(pile, 3, parse_target("Treasure", REG))
    assert t.bounded and not t.deterministic and (t.min, t.max) == (1, 3)
    assert t.range_at_least(2) == (Fraction(5, 6), 1)
    assert (t.expected, t.upper_expected) == (2, 3)
    assert t.range_exactly(1) == (0, Fraction(1, 6))
    assert sum(p for _, p in t.joint) == 1
    assert dict(t.unknown_distribution()) == {0: Fraction(1, 6), 1: Fraction(2, 3), 2: Fraction(1, 6)}


def test_unknown_size_is_not_determinable():
    d = draw_count_distribution(src(DrawPileKnowledge(size_known=False)), 3, parse_target(C, REG))
    assert not d.determinable and d.p_zero() is None and d.expected is None and d.range_at_least(1) is None
    out = d.to_dict()
    assert out["distribution"] is None and out["p_zero"] == {"p_exact": None, "p": None}
    assert out["p_reshuffle"]["p"] is None and (d.min, d.max) == (0, 3)
    # Known deck, then a reshuffle of a discard pile of unknown size.
    d2 = draw_count_distribution(src(DrawPileKnowledge(top=(G,)), DiscardKnowledge(size_known=False)), 3,
                                 parse_target(G, REG))
    assert not d2.determinable and (d2.min, d2.max) == (1, 3)


def test_monte_carlo_fallback_is_seeded_and_labelled():
    pile = src(DrawPileKnowledge(rest={C: 3, E: 2}))
    a = draw_count_distribution(pile, 2, parse_target(C, REG), limit=1, samples=4000, seed=7)
    b = draw_count_distribution(pile, 2, parse_target(C, REG), limit=1, samples=4000, seed=7)
    assert a == b and a.method == "monte_carlo" and a.to_dict()["samples"] == 4000
    assert a.to_dict()["distribution"][1]["p_exact"] is None
    assert abs(a.p_exactly(1) - 0.6) < 0.05 and (a.min, a.max) == (0, 2)  # range stays exact


# ---------------------------------------------------------------- the same cases through a session

def test_session_hand_verifiable_cases():
    s = from_snapshot(draw_pile={"composition": {C: 4, E: 1}}, discard_pile={"composition": {}})
    g5 = get_guaranteed_cards(s, 5, at="now")
    assert [b.describe() for b in g5.guarantees] == ["exactly 4 Copper", "exactly 1 Estate", "exactly 4 Treasure",
                                                      "exactly 1 Victory"]
    assert g5.determinable and not g5.bounded and g5.p_reshuffle == 0 and g5.impossible == ()
    g3 = get_guaranteed_cards(s, 3, at="now")
    assert (g3.bound(C).min, g3.bound(C).max) == (2, 3) and (g3.bound("Treasure").min, g3.bound("Treasure").max) == (2, 3)
    assert g3.bound(E).describe() == "at least 0 Estate" and g3.bound(E) not in g3.guarantees
    a = analyze_draw(s, 5, "copper", at="now")
    assert a.deterministic and a.min == a.max == 4 and a.p_exactly(4) == 1

    s = from_snapshot(draw_pile={"composition": {C: 3, E: 2}}, discard_pile={"composition": {}})
    a = analyze_draw(s, 2, C, at="now")
    assert [a.p_exactly(k) for k in range(3)] == [Fraction(1, 10), Fraction(6, 10), Fraction(3, 10)]
    assert a.p_at_least(1) == Fraction(9, 10)
    out = a.to_dict()
    assert [d["p_exact"] for d in out["distribution"]] == ["1/10", "3/5", "3/10"]
    assert out["p_at_least_one"] == {"p_exact": "9/10", "p": 0.9} and out["expected"]["exact"] == "6/5"

    s = from_snapshot(draw_pile={"order": [G, C]}, discard_pile={"composition": {S: 2, E: 2}})
    a = analyze_draw(s, 4, S, at="now")
    assert [a.p_exactly(k) for k in range(3)] == [Fraction(1, 6), Fraction(2, 3), Fraction(1, 6)]
    assert a.counts.p_reshuffle == 1
    g = get_guaranteed_cards(s, 4, at="now")
    assert {b.describe() for b in g.guarantees} >= {"exactly 1 Gold", "exactly 1 Copper", "at least 2 Treasure"}
    g7 = get_guaranteed_cards(s, 7, at="now")
    assert g7.p_short == 1 and all(b.min == b.max for b in g7.cards + g7.types)
    assert {b.target.name: b.min for b in g7.cards} == {C: 1, E: 2, G: 1, S: 2}


def test_at_semantics_follow_predict_next_hand():
    s = initialize_session()  # opening hand pending: auto = now
    a = analyze_draw(s, 5, C)
    assert a.at == "now" and a.turn == 1
    assert [a.p_exactly(k) for k in range(6)] == [Fraction(comb(7, k) * comb(3, 5 - k), comb(10, 5)) for k in range(6)]
    assert any("pending draw" in n for n in a.notes) and any("no Clean-up" in n for n in a.notes)
    s = record_event(s, ObserveHand([C] * 4 + [E]))
    assert analyze_draw(s, 5, C).at == "next_turn" and predict_next_hand(s).which == "next_turn"
    nxt = analyze_draw(s, 5, C, at="next-turn")
    assert nxt.deterministic and nxt.min == 3 and nxt.turn == 2
    # now: the current draw pile (3 Copper, 2 Estate) and an empty discard pile -> 7 draws are short.
    now = analyze_draw(s, 7, C, at="now")
    assert now.min == now.max == 3 and now.counts.p_short == 1 and now.counts.p_reshuffle == 0
    # next turn: 5 from the draw pile, then 2 from the reshuffled hand (4 Copper, Estate).
    later = analyze_draw(s, 7, C, at="next_turn")
    assert later.counts.p_reshuffle == 1 and later.p_exactly(5) == Fraction(comb(4, 2), comb(5, 2))
    assert any("Clean-up" in n for n in later.notes)
    with pytest.raises(ValueError, match="at must be"):
        analyze_draw(s, 5, C, at="tomorrow")
    with pytest.raises(ValueError, match="positive"):
        analyze_draw(s, 0, C)


# ---------------------------------------------------------------- brute force

def _perms(cards):
    """Distinct orderings of a multiset (equally likely under a uniform shuffle)."""
    c = Counter(cards)
    n = sum(c.values())

    def rec_(prefix):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for card in sorted(c):
            if c[card]:
                c[card] -= 1
                prefix.append(card)
                yield from rec_(prefix)
                prefix.pop()
                c[card] += 1

    return list(rec_([]))


def _draw(state, n):
    """Every way of drawing n cards from a concrete state (deck order known): [(p, drawn, state, reshuffled)]."""
    deck, discard, hand = state
    k = min(n, len(deck))
    drawn, deck = deck[:k], deck[k:]
    if k == n or not discard:
        return [(Fraction(1), drawn, (deck, discard, hand), False)]
    orders = _perms(discard)
    out = []
    for order in orders:
        m = min(n - k, len(order))
        out.append((Fraction(1, len(orders)), drawn + order[:m], (order[m:], (), hand), True))
    return out


class Brute:
    """Exact reference model: a distribution over concrete (deck order, discard, hand) states."""

    def __init__(self, deck):
        self.states = {(order, (), ()): Fraction(1, len(_perms(deck))) for order in _perms(deck)}

    def _merge(self, items):
        out = {}
        for p, st in items:
            st = (st[0], tuple(sorted(st[1])), tuple(sorted(st[2])))
            out[st] = out.get(st, Fraction(0)) + p
        total = sum(out.values())
        self.states = {k: v / total for k, v in out.items()}

    def draw_to_hand(self, n, observed=None):
        items = []
        for st, p in self.states.items():
            for q, drawn, (deck, discard, hand), _ in _draw(st, n):
                new_hand = hand + drawn
                if observed is None or sorted(new_hand) == sorted(observed):
                    items.append((p * q, (deck, discard, new_hand)))
        self._merge(items)

    def gain(self, card):
        self._merge([(p, (d, x + (card,), h)) for (d, x, h), p in self.states.items()])

    def end_turn(self):
        self._merge([(p, (d, x + h, ())) for (d, x, h), p in self.states.items()])

    def query(self, n, target, at, pending):
        states = self.states
        if at == "next_turn":
            saved = self.states
            if pending:
                self.draw_to_hand(pending)
            self.end_turn()
            states, self.states = self.states, saved
        out, reshuffle, short = {}, Fraction(0), Fraction(0)
        for st, p in states.items():
            for q, drawn, _, reshuffled in _draw(st, n):
                k = sum(target.matches(c) for c in drawn)
                out[k] = out.get(k, Fraction(0)) + p * q
                reshuffle += p * q if reshuffled else 0
                short += p * q if len(drawn) < n else 0
        return [out.get(k, Fraction(0)) for k in range(n + 1)], reshuffle, short


def _check(session, brute, queries):
    pending = infer_belief_state(session).pending_draws
    for n, name, at in queries:
        a = analyze_draw(session, n, name, at)
        expected, reshuffle, short = brute.query(n, a.target, a.at, pending)
        assert [a.p_exactly(k) for k in range(n + 1)] == expected, (n, name, at)
        assert (a.counts.p_reshuffle, a.counts.p_short) == (reshuffle, short), (n, name, at)
        support = [k for k, p in enumerate(expected) if p]
        assert (a.min, a.max) == (min(support), max(support)) and a.method == "exact"


QUERIES = [(n, t, at) for n in (2, 5, 7, 13) for t in (C, S, "Treasure") for at in ("now", "next_turn")]


def test_repeated_observations_and_unobserved_hand_match_brute_force():
    s, b = initialize_session(), Brute([C] * 7 + [E] * 3)
    _check(s, b, QUERIES)
    s = record_event(s, ObserveHand([C] * 4 + [E]))                       # T1 observed
    b.draw_to_hand(5, [C] * 4 + [E])
    _check(s, b, QUERIES)
    s = rec(s, RecordPurchase(S), EndTurnObservation())                    # T2 pending (deduced)
    b.gain(S)
    b.end_turn()
    _check(s, b, QUERIES)
    s = rec(s, RecordPurchase(S), EndTurnObservation())                    # T2 hand never entered
    b.draw_to_hand(5)
    b.gain(S)
    b.end_turn()
    _check(s, b, QUERIES)
    s = record_event(s, ObserveHand([C, C, C, E, S]))                      # T3 observed after a reshuffle
    b.draw_to_hand(5, [C, C, C, E, S])
    _check(s, b, QUERIES)
    s = rec(s, EndTurnObservation(), EndTurnObservation())                 # T4 unobserved -> T5 mixture
    b.end_turn()
    b.draw_to_hand(5)
    b.end_turn()
    assert len(infer_belief_state(s).worlds) > 1
    _check(s, b, QUERIES)
    s = record_event(s, RecordDraw(count=2))                               # 2 of T5 drawn unseen
    b.draw_to_hand(2)
    _check(s, b, [(n, t, at) for n in (3, 8) for t in (S, E) for at in ("now", "next_turn")])


# ---------------------------------------------------------------- incomplete history

PARTIAL = {"hand": ["Smithy", "Militia", C, C, S],
           "draw_pile": {"top": [G], "composition": {C: 2, E: 1}, "unknown_count": 2},
           "discard_pile": {"composition": {C: 2}, "unknown_count": 1}}


def _completions():
    for deck_unknown in itertools.combinations_with_replacement((C, S, E, G), 2):
        for disc in (C, S, E, G):
            player = copy.deepcopy(PARTIAL)
            player["draw_pile"] = {"top": [G], "composition": dict(Counter({C: 2, E: 1}) + Counter(deck_unknown))}
            player["discard_pile"] = {"composition": dict(Counter({C: 2}) + Counter([disc]))}
            yield from_snapshot(**player)


@pytest.mark.parametrize("n,name,at", [(3, "Treasure", "now"), (7, S, "now"), (8, C, "now"), (5, G, "next_turn"),
                                       (9, "Victory", "next_turn")])
def test_incomplete_history_bounds_contain_the_truth(n, name, at):
    partial = analyze_draw(from_snapshot(**copy.deepcopy(PARTIAL)), n, name, at)
    c = partial.counts
    assert partial.bounded and partial.determinable and not partial.deterministic
    assert any("bounds" in note for note in partial.notes)
    out = partial.to_dict()
    assert out["bounded"] and out["upper"] is not None and "bounds" in out
    for truth_session in _completions():
        truth = analyze_draw(truth_session, n, name, at)
        assert not truth.bounded
        for k in range(n + 1):
            lo, hi = c.range_at_least(k)
            assert lo <= truth.p_at_least(k) <= hi
            lo, hi = c.range_exactly(k)
            assert lo <= truth.p_exactly(k) <= hi
        assert c.expected <= truth.counts.expected <= c.upper_expected
        assert partial.min <= truth.min and truth.max <= partial.max


def _observed_bounds_hold(partial, completions, events, queries):
    """Every completion of the "?" cards that allows ``events`` has its exact answers inside the bounds."""
    s = rec(partial, *events)
    checked = 0
    for player in completions:
        try:
            truth_session = rec(from_snapshot(**player), *events)
        except SessionError:
            continue  # this identity of the "?" cards makes the observation impossible
        checked += 1
        for n, name in queries:
            a, truth = analyze_draw(s, n, name, "now"), analyze_draw(truth_session, n, name, "now")
            assert not truth.bounded
            for k in range(n + 1):
                lo, hi = a.counts.range_at_least(k)
                assert lo <= truth.p_at_least(k) <= hi
            assert a.counts.expected <= truth.counts.expected <= a.counts.upper_expected
            assert a.min <= truth.min and truth.max <= a.max
    assert checked
    return s


def test_observing_a_card_a_question_mark_may_be_keeps_honest_bounds():
    # Draw pile {Copper, ?}; a Copper is drawn. If the "?" is an Estate the next card is surely not a Copper,
    # if it is a Copper it surely is: the "?" has no prior, so only [0, 1] is honest.
    partial = from_snapshot(draw_pile={"composition": {C: 1}, "unknown_count": 1}, discard_pile={"composition": {}})
    completions = [{"draw_pile": {"composition": dict(Counter([C, u]))}, "discard_pile": {"composition": {}}}
                   for u in (C, E, S, G)]
    s = _observed_bounds_hold(partial, completions, [RecordDraw(cards=[C])], [(1, C), (1, E), (1, "Treasure")])
    assert analyze_draw(s, 1, C, "now").counts.range_at_least(1) == (0, 1)
    assert any("'?'" in f and "no prior" in f for f in infer_belief_state(s).flags)


def test_observation_through_known_top_and_mid_draw_reshuffle_with_question_marks():
    # Draw pile: Gold on top, then {Copper}; discard {Copper 2, Estate, ?}. Drawing 4 takes Gold and Copper,
    # reshuffles the discard pile and draws 2 of {Copper, Copper, Estate, ?}: observed Copper, Estate.
    # Only {Copper 2, Estate} - {Copper, Estate} = 1 Copper is certainly left; the other card is "?"
    # (if the "?" is a Copper the pile is {Copper, Copper}, if an Estate {Copper, Estate}).
    def player(u=None):
        discard = {"composition": dict(Counter([C, C, E] + ([u] if u else [])))}
        if u is None:
            discard["unknown_count"] = 1
        return {"draw_pile": {"top": [G], "composition": {C: 1}}, "discard_pile": discard}

    s = _observed_bounds_hold(from_snapshot(**player()), [player(u) for u in (C, E, S, G)],
                              [RecordDraw(cards=[G, C, C, E])], [(1, C), (1, E), (2, C), (2, "Treasure")])
    pile = infer_belief_state(s).zone_summary("draw_pile")
    assert (pile["size"], pile["certain"], pile["uncertain_count"]) == (2, {C: 1}, 1)
    assert analyze_draw(s, 1, C, "now").counts.range_at_least(1) == (Fraction(1, 2), 1)
    assert analyze_draw(s, 2, C, "now").min == 1
    with pytest.raises(SessionError):  # the known top card must be among the drawn cards
        rec(from_snapshot(**player()), RecordDraw(cards=[C, C, C, E]))


def test_insufficient_history_is_not_determinable():
    s = from_snapshot(hand=[C, C, C, E, E])  # no deck_count, no owned cards: draw pile size unknown
    for at in ("now", "next_turn"):
        a = analyze_draw(s, 5, C, at)
        assert not a.determinable and not a.deterministic
        out = a.to_dict()
        assert out["determinable"] is False and out["distribution"] is None
        assert out["p_at_least_one"] == {"p_exact": None, "p": None} and out["expected"]["value"] is None
        assert any(n.startswith("Not determinable") for n in a.notes)
    g = get_guaranteed_cards(s)
    assert not g.determinable and g.p_reshuffle is None and g.guarantees == ()
    # Size known but contents unknown: honest bounds, not "not determinable".
    a = analyze_draw(from_snapshot(hand=[C], deck_count=6), 3, C, "now")
    assert a.determinable and a.bounded and a.counts.range_at_least(1) == (0, 1)


# ---------------------------------------------------------------- analyze_deck

def test_analyze_deck_deduced_vs_probabilistic():
    s = initialize_session()
    d = analyze_deck(s)
    draw = d.zone("draw_pile")
    assert draw.status == "deduced" and draw.size == 10 and draw.card(C).certain == 7 and draw.card(E).certain == 3
    assert draw.card("Treasure").certain == 7 and d.zone("hand").size == 0 and d.pending_draws == 5
    assert d.zone("owned").card(C).certain == 7

    s = record_event(s, RecordDraw(count=5))  # opening hand drawn unseen
    d = analyze_deck(s)
    hand = d.zone("hand").card(C)
    assert hand.status == "probabilistic" and hand.certain is None and (hand.min, hand.max) == (2, 5)
    assert hand.expected == Fraction(7, 2)
    assert dict(hand.distribution) == {k: Fraction(comb(7, k) * comb(3, 5 - k), comb(10, 5)) for k in range(2, 6)}
    assert sum(p for _, p in hand.distribution) == 1
    deck = d.zone("draw_pile").card(C)
    assert deck.status == "probabilistic" and deck.expected == Fraction(7, 2)
    assert d.zone("owned").card(C).certain == 7 and d.zone("draw_pile").size == 5

    s = record_event(s, ObserveHand([C] * 4 + [E]))
    d = analyze_deck(s)
    assert d.zone("hand").card(C).status == "observed" and d.zone("draw_pile").card(C).status == "deduced"
    s = rec(s, RecordTrash(E))
    assert analyze_deck(s).zone("trash").card(E).certain == 1
    assert analyze_deck(s).zone("owned").card(E).certain == 2


def test_analyze_deck_unknown_cards_and_top_order():
    d = analyze_deck(from_snapshot(**copy.deepcopy(PARTIAL)))
    draw = d.zone("draw_pile")
    assert draw.status == "unknown" and draw.top == (G,) and draw.unknown == 2 and draw.size == 6
    assert (draw.card(C).min, draw.card(C).max, draw.card(C).status) == (2, 4, "unknown")
    assert (draw.card(G).min, draw.card(G).max) == (1, 3) and draw.card(G).certain is None
    assert d.zone("owned").status == "unknown" and d.zone("owned").size is None
    unknown = analyze_deck(from_snapshot(hand=[C]))
    assert unknown.zone("draw_pile").size is None and unknown.zone("draw_pile").cards == ()


# ---------------------------------------------------------------- properties

def _mixture_session():
    return rec(initialize_session(), ObserveHand([C] * 4 + [E]), RecordPurchase(S), EndTurnObservation(),
               RecordPurchase(S), EndTurnObservation(), ObserveHand([C, C, C, E, S]), EndTurnObservation(),
               EndTurnObservation())


def test_no_mutation_frozen_results_and_determinism():
    s = _mixture_session()
    before, belief = s.to_dict(), infer_belief_state(s).to_dict()
    results = [analyze_deck(s), analyze_draw(s, 5, S), get_guaranteed_cards(s)]
    assert s.to_dict() == before and infer_belief_state(s).to_dict() == belief
    for r in results:
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.worlds = 0  # type: ignore[misc]
    again = [analyze_deck(s), analyze_draw(s, 5, S), get_guaranteed_cards(s)]
    assert [r.to_dict() for r in results] == [r.to_dict() for r in again] and results == again


def test_json_round_trip_and_exact_strings():
    s = _mixture_session()
    a = analyze_draw(s, 5, S)
    for r in (analyze_deck(s), a, get_guaranteed_cards(s), analyze_draw(from_snapshot(**copy.deepcopy(PARTIAL)), 3, C)):
        data = r.to_dict()
        assert json.loads(json.dumps(data)) == data
    data = a.to_dict()
    probs = [Fraction(d["p_exact"]) for d in data["distribution"]]
    assert probs == [a.p_exactly(k) for k in range(6)] and sum(probs) == 1
    assert all(abs(d["p"] - float(Fraction(d["p_exact"]))) < 1e-12 for d in data["distribution"])
    assert Fraction(data["expected"]["exact"]) == a.counts.expected
    assert data["target"] == {"kind": "card", "name": S, "cards": [S]} and data["at"] == "now"

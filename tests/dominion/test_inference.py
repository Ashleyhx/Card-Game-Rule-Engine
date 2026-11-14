"""Phase 2b Part A: initial state, belief tracking and cross-turn inference (session API)."""

import copy
import itertools
import json
from collections import Counter
from fractions import Fraction
from math import comb

import pytest

from card_game_engine.games.dominion.inference import (
    EVENT_TYPES,
    EndTurnObservation,
    HandNotDetermined,
    IllegalEvent,
    InconsistentObservation,
    ObserveHand,
    RecordDiscard,
    RecordDraw,
    RecordGain,
    RecordPlay,
    RecordPurchase,
    RecordSupply,
    RecordTrash,
    Session,
    SessionError,
    correct_event,
    event_from_dict,
    infer_belief_state,
    initialize_session,
    plan_next_action,
    predict_next_hand,
    record_event,
    record_observation,
    remove_event,
    reset_session,
    to_game_state,
)
from card_game_engine.games.dominion.planning import PlanConfig
from card_game_engine.games.dominion.rules import has_errors, validate_state

from .conftest import FIRST_GAME

C, E, S, G = "Copper", "Estate", "Silver", "Gold"


def rec(session, *events):
    for e in events:
        session = record_event(session, e)
    return session


def dist(prediction):
    """{sorted cards tuple: Fraction} for an exact prediction."""
    return {cards: p for cards, p, _ in prediction.distribution}


def hand(**n):
    return tuple(sorted(Counter(n).elements()))


def spec_through_turn2():
    """Spec example: T1 4C+1E buy Silver; T2 hand inferred (not entered), buy Silver; T2 ends."""
    s = initialize_session()
    return rec(s, ObserveHand([C] * 4 + [E]), RecordPurchase(S), EndTurnObservation(),
               RecordPurchase(S), EndTurnObservation())


def spec_turn3_observed():
    return record_event(spec_through_turn2(), ObserveHand([C, C, C, E, S]))


def zone_total(belief):
    w = belief.worlds[0]
    return (w.hand_counter + w.draw.rest_counter + Counter(w.draw.top) + w.discard.counter
            + Counter(belief.in_play))


# 1 ---------------------------------------------------------------------------
def test_standard_initial_composition_and_opening_distribution():
    s = initialize_session()
    b = infer_belief_state(s)
    assert (b.turn, b.phase, b.pending_draws) == (1, "action", 5)
    assert len(b.worlds) == 1
    w = b.worlds[0]
    assert dict(w.draw.rest) == {C: 7, E: 3} and w.draw.top == () and w.draw.unknown == 0  # no invented order
    assert w.discard.size == 0 and w.hand == ()
    assert b.owned == {C: 7, E: 3}
    assert b.zone_status("draw_pile") == "deduced" and b.information_level == "known_composition"

    p = predict_next_hand(s)
    assert p.which == "start_of_turn" and p.turn == 1 and p.method == "exact" and not p.deterministic
    d = dist(p)
    assert d[(C,) * 5] == Fraction(comb(7, 5), comb(10, 5)) == Fraction(21, 252)
    for coppers in range(2, 6):
        assert d[hand(Copper=coppers, Estate=5 - coppers)] == Fraction(comb(7, coppers) * comb(3, 5 - coppers), 252)
    assert sum(d.values()) == 1 and len(d) == 4
    assert p.expected_coins == Fraction(7, 2)


# 2 ---------------------------------------------------------------------------
def test_complementary_second_hand_is_deduced_exactly():
    s = record_event(initialize_session(), ObserveHand([C, C, C, C, E]))
    b = infer_belief_state(s)
    assert dict(b.worlds[0].draw.rest) == {C: 3, E: 2}
    assert b.is_deterministic("draw_pile") and b.zone_status("hand") == "observed"
    p = predict_next_hand(s)
    assert p.which == "next_turn" and p.turn == 2
    assert p.deterministic and p.hand == hand(Copper=3, Estate=2)
    assert dist(p) == {hand(Copper=3, Estate=2): Fraction(1)}
    assert not p.lower_bound and b.flags == ()
    assert p.to_dict()["distribution"] == [{"cards": [C, C, C, E, E], "p_exact": "1/1", "p": 1.0}]


# 3 ---------------------------------------------------------------------------
def test_purchases_go_to_discard_and_hand_and_play_wait_for_cleanup():
    s = rec(initialize_session(), ObserveHand([C, C, C, C, E]), RecordPurchase(S))
    b = infer_belief_state(s)
    w = b.worlds[0]
    assert dict(w.discard.cards) == {S: 1}  # hand Estate and played Coppers are NOT in the discard yet
    assert sorted(b.in_play) == [C] * 4 and b.determined_hand() == [E]
    assert b.owned == {C: 7, E: 3, S: 1} and b.game.supply[S] == 39
    assert b.game.turn.coins == 1 and b.game.turn.buys == 0
    b2 = infer_belief_state(record_event(s, EndTurnObservation()))
    assert dict(b2.worlds[0].discard.cards) == {C: 4, E: 1, S: 1}
    assert (b2.turn, b2.pending_draws, b2.in_play, b2.worlds[0].hand) == (2, 5, (), ())
    assert zone_total(b2) == Counter(b2.owned)


# 4 ---------------------------------------------------------------------------
def test_reshuffle_at_turn_3_with_known_composition():
    s = spec_through_turn2()
    b = infer_belief_state(s)
    assert any("hand deduced exactly: 3 Copper, 2 Estate" in n for n in b.notes)  # T2 never entered
    w = b.worlds[0]
    assert len(b.worlds) == 1 and w.draw.size == 0
    assert dict(w.discard.cards) == {C: 7, E: 3, S: 2}
    p = predict_next_hand(s)
    assert p.which == "start_of_turn" and p.turn == 3 and p.p_reshuffle == 1
    b3 = infer_belief_state(record_event(s, ObserveHand([C, C, C, E, S])))
    assert dict(b3.worlds[0].draw.rest) == {C: 4, E: 2, S: 1} and b3.worlds[0].discard.size == 0
    assert zone_total(b3) == Counter(b3.owned)


# 5 ---------------------------------------------------------------------------
def test_turn3_distribution_is_exact_hypergeometric():
    p = predict_next_hand(spec_through_turn2())
    d = dist(p)
    total = comb(12, 5)
    for c in range(6):
        for e in range(4):
            for sv in range(3):
                if c + e + sv == 5:
                    expected = Fraction(comb(7, c) * comb(3, e) * comb(2, sv), total)
                    assert d.get(hand(Copper=c, Estate=e, Silver=sv), Fraction(0)) == expected
    assert d[hand(Copper=3, Estate=1, Silver=1)] == Fraction(35 * 3 * 2, 792) == Fraction(35, 132)
    assert d[(C,) * 5] == Fraction(21, 792)
    assert d[hand(Copper=3, Silver=2)] == Fraction(35, 792)
    assert sum(d.values()) == 1
    assert p.expected_coins == Fraction(5 * (7 + 2 * 2), 12)
    assert p.p_at_least_one[S] == 1 - Fraction(comb(10, 5), total)


# 6 ---------------------------------------------------------------------------
def test_turn4_conditional_inference():
    p = predict_next_hand(spec_turn3_observed())
    assert p.which == "next_turn" and p.turn == 4 and p.p_reshuffle == 0
    assert p.p_at_least_one[S] == Fraction(5, 7)
    d = dist(p)  # 5 of the remaining 4C, 2E, 1S
    assert d[hand(Copper=4, Estate=1)] == Fraction(comb(4, 4) * comb(2, 1), comb(7, 5))
    assert d[hand(Copper=2, Estate=2, Silver=1)] == Fraction(comb(4, 2), comb(7, 5))
    assert sum(d.values()) == 1


def test_new_observation_changes_prediction():
    before = predict_next_hand(spec_through_turn2())  # turn-3 hand from 7C 3E 2S
    assert before.p_at_least_one[S] == Fraction(15, 22)
    seen_silver = predict_next_hand(spec_turn3_observed())
    assert seen_silver.p_at_least_one[S] == Fraction(5, 7)
    no_silver = predict_next_hand(record_event(spec_through_turn2(), ObserveHand([C, C, C, E, E])))
    assert no_silver.p_at_least_one[S] == 1 - Fraction(1, comb(7, 5)) == Fraction(20, 21)


# 7 + 8 ---------------------------------------------------------------------------
def _turn5_after_unobserved_turn4():
    return rec(spec_turn3_observed(), EndTurnObservation(), EndTurnObservation())


def _brute_force_turn5():
    """Uniform over orderings of the 7-card T4 pile, then over 3-card subsets of the 10-card reshuffle."""
    out: dict[tuple, Fraction] = {}
    orders = sorted(set(itertools.permutations([C] * 4 + [E] * 2 + [S])))
    for order in orders:
        t4, left = list(order[:5]), list(order[5:])
        discard = [C, C, C, E, S] + t4
        subsets = list(itertools.combinations(range(10), 3))
        for idx in subsets:
            key = tuple(sorted(left + [discard[i] for i in idx]))
            out[key] = out.get(key, Fraction(0)) + Fraction(1, len(orders) * len(subsets))
    return out


def test_cross_turn_correlation_matches_brute_force_and_differs_from_independent_draws():
    s = _turn5_after_unobserved_turn4()
    b = infer_belief_state(s)
    assert len(b.worlds) > 1 and sum(w.weight for w in b.worlds) == 1
    assert any("not observed" in f for f in b.flags)
    p = predict_next_hand(s)
    assert p.which == "start_of_turn" and p.p_reshuffle == 1
    assert dist(p) == _brute_force_turn5()
    assert sum(dist(p).values()) == 1
    naive = {}  # independent: 5 fresh cards from the whole 12-card deck
    for combo in itertools.combinations([C] * 7 + [E] * 3 + [S] * 2, 5):
        k = tuple(sorted(combo))
        naive[k] = naive.get(k, Fraction(0)) + Fraction(1, comb(12, 5))
    assert dist(p) != naive


def test_probabilities_normalised_everywhere():
    for s in (initialize_session(), spec_through_turn2(), spec_turn3_observed(), _turn5_after_unobserved_turn4()):
        assert sum(w.weight for w in infer_belief_state(s).worlds) == 1
        p = predict_next_hand(s)
        assert sum(dist(p).values()) == 1
        assert all(0 <= v <= 1 for v in p.p_coins_at_least.values())


def test_unobserved_draw_then_observation_is_bayes_conditioning():
    snap = {"schema_version": 1, "num_players": 2, "kingdom": FIRST_GAME, "turn_number": 5,
            "player": {"phase": "action", "hand": [], "pending_draws": 2,
                       "owned_cards": {C: 2, E: 1, S: 1},
                       "draw_pile": {"composition": {C: 2, E: 1, S: 1}}, "discard_pile": {"composition": {}}}}
    s = rec(initialize_session({"snapshot": snap}), RecordDraw(count=2))
    prior = {w.hand: w.weight for w in infer_belief_state(s).worlds}
    assert prior == {((C, 2),): Fraction(1, 6), ((C, 1), (E, 1)): Fraction(1, 3),
                     ((C, 1), (S, 1)): Fraction(1, 3), ((E, 1), (S, 1)): Fraction(1, 6)}
    s = record_event(s, RecordDraw([S]))  # third card seen: Silver
    post = {w.hand: w.weight for w in infer_belief_state(s).worlds}
    brute: dict = {}
    orders = sorted(set(itertools.permutations([C, C, E, S])))
    matching = [o for o in orders if o[2] == S]
    for o in matching:
        key = tuple(sorted(Counter(o[:3]).items()))
        brute[key] = brute.get(key, Fraction(0)) + Fraction(1, len(matching))
    assert post == brute == {((C, 2), (S, 1)): Fraction(1, 3), ((C, 1), (E, 1), (S, 1)): Fraction(2, 3)}


def test_later_hand_observation_conditions_the_mixture():
    s = rec(spec_turn3_observed(), EndTurnObservation(), RecordDraw(count=5))  # T4 drawn, unseen
    assert len(infer_belief_state(s).worlds) > 1
    s = record_event(s, ObserveHand([C, C, C, C, E]))  # assertion about the current hand
    b = infer_belief_state(s)
    assert len(b.worlds) == 1 and dict(b.worlds[0].draw.rest) == {E: 1, S: 1}
    assert b.zone_status("hand") == "observed"


# 9 ---------------------------------------------------------------------------
def test_deterministic_versus_uncertain():
    s2 = rec(initialize_session(), ObserveHand([C, C, C, C, E]), EndTurnObservation())
    p2 = predict_next_hand(s2)
    assert p2.deterministic and p2.hand == hand(Copper=3, Estate=2)
    p3 = predict_next_hand(spec_through_turn2())
    assert not p3.deterministic and p3.hand is None and len(p3.distribution) > 1
    b = infer_belief_state(rec(spec_turn3_observed(), EndTurnObservation(), RecordDraw(count=5)))
    assert b.zone_status("hand") == "probabilistic" and not b.is_deterministic("hand")
    assert b.determined_hand() is None
    assert "expected" in b.to_dict()["inferred"]["hand"]


def test_purchase_with_undetermined_hand_is_evidence():
    s = record_event(initialize_session(), RecordPurchase(S))  # opening hand never entered
    b = infer_belief_state(s)
    weights = {w.hand: w.weight for w in b.worlds}
    assert weights == {((C, 5),): Fraction(1, 11), ((C, 4), (E, 1)): Fraction(5, 11),
                       ((C, 3), (E, 2)): Fraction(5, 11)}  # 2 Copper + 3 Estate cannot pay $3
    assert any("bought Silver while the hand was not determined" in f for f in b.flags)
    with pytest.raises(InconsistentObservation, match="impossible resource change"):
        record_event(initialize_session(), RecordPurchase(G))


def test_discard_from_unknown_hand_conditions_worlds():
    s = rec(initialize_session(), RecordDraw(count=5), RecordDiscard([E, E]))
    b = infer_belief_state(s)
    assert {w.hand: w.weight for w in b.worlds} == {((C, 3),): Fraction(5, 6), ((C, 2), (E, 1)): Fraction(1, 6)}
    assert all(w.discard.counter == Counter({E: 2}) for w in b.worlds)


# 10 ---------------------------------------------------------------------------
def _assert_rejected(session, event, exc, *fragments):
    before = session.to_dict()
    belief_before = infer_belief_state(session).to_dict()
    with pytest.raises(exc) as info:
        record_event(session, event)
    assert info.value.event_index == len(session.events)
    for f in fragments:
        assert f in str(info.value)
    assert session.to_dict() == before and infer_belief_state(session).to_dict() == belief_before
    return info.value


def test_contradictions_detected_with_index_and_no_state_change():
    s = spec_through_turn2()
    err = _assert_rejected(s, ObserveHand([S, S, S, C, C]), InconsistentObservation, "3 Silver", "only 2")
    assert f"Event #{len(s.events)}" in str(err)

    s1 = rec(initialize_session(), ObserveHand([C, C, C, C, E]), EndTurnObservation())
    _assert_rejected(s1, ObserveHand([C, C, C, C, E]), InconsistentObservation, "4 Copper", "at most 3")

    s2 = rec(initialize_session(), ObserveHand([C, C, C, C, E]), RecordPurchase(S), EndTurnObservation())
    _assert_rejected(s2, RecordPurchase(G), InconsistentObservation, "impossible resource change", "$6", "$3")

    _assert_rejected(initialize_session(), ObserveHand([C] * 6), InconsistentObservation, "6 card(s)")


def test_inconsistent_versus_incomplete_and_illegal():
    s = _turn5_after_unobserved_turn4()  # incomplete history: flagged, not an error
    assert infer_belief_state(s).flags
    _assert_rejected(initialize_session(), ObserveHand(["Coppper"] * 5), IllegalEvent, "Copper")
    _assert_rejected(rec(spec_turn3_observed(), EndTurnObservation(), RecordDraw(count=5)),
                     RecordPlay("Smithy"), IllegalEvent, "not determined")
    _assert_rejected(initialize_session(), RecordDraw(), IllegalEvent, "exactly one")
    assert not issubclass(IllegalEvent, InconsistentObservation) and issubclass(IllegalEvent, SessionError)
    with pytest.raises(IllegalEvent):
        record_observation(initialize_session(), RecordPurchase(S))


def test_max_worlds_collapse_is_flagged_and_honest():
    s = record_event(initialize_session({"max_worlds": 3}), RecordDraw(count=5))  # 4 outcomes > 3
    b = infer_belief_state(s)
    assert len(b.worlds) == 1
    w = b.worlds[0]
    assert dict(w.hand) == {C: 2, "?": 3} and dict(w.draw.rest) == {C: 2} and w.draw.unknown == 3
    assert any("max_worlds" in f for f in b.flags)
    assert b.zone_status("hand") == "unknown"
    assert predict_next_hand(s).lower_bound
    b2 = infer_belief_state(record_event(s, ObserveHand([C, C, C, C, E])))  # "?" is consistent with anything
    assert b2.determined_hand() == [C, C, C, C, E]


# 11 ---------------------------------------------------------------------------
def test_replay_reproducible_and_round_trip():
    a, b = _turn5_after_unobserved_turn4(), _turn5_after_unobserved_turn4()
    assert a == b and a.to_dict() == b.to_dict()
    assert infer_belief_state(a).to_dict() == infer_belief_state(b).to_dict()
    data = json.loads(json.dumps(a.to_dict()))
    c = Session.from_dict(data)
    assert c == a and infer_belief_state(c).to_dict() == infer_belief_state(a).to_dict()
    assert predict_next_hand(c).to_dict() == predict_next_hand(a).to_dict()
    fresh = Session(a.config, a.events)  # no cache: full replay
    assert infer_belief_state(fresh) == infer_belief_state(a)
    json.dumps(infer_belief_state(a).to_dict())
    json.dumps(predict_next_hand(a).to_dict())
    assert data["events"][-1] == {"kind": "end_turn"}


def test_event_json_round_trip():
    events = [ObserveHand([C, E]), RecordDraw([S]), RecordDraw(count=2), RecordPlay("Throne Room", "Smithy"),
              RecordPurchase(S), RecordGain(G, "deck"), RecordTrash(C, "in_play"), RecordDiscard([E]),
              RecordSupply("Province", 7), EndTurnObservation()]
    assert {e.kind for e in events} == set(EVENT_TYPES)
    for e in events:
        assert event_from_dict(json.loads(json.dumps(e.to_dict()))) == e
    assert ObserveHand({C: 2, E: 1}).cards == (C, C, E)


# 12 ---------------------------------------------------------------------------
def test_no_mutation_of_inputs():
    snap = {"schema_version": 1, "num_players": 2, "kingdom": FIRST_GAME, "turn_number": 3,
            "player": {"phase": "action", "hand": [S, C, C, C, E], "owned_cards": {C: 7, E: 3, S: 2},
                       "discard_pile": {"composition": {}}}}
    config = {"snapshot": snap, "max_worlds": 100}
    events = [EndTurnObservation(), RecordDraw(count=5)]
    saved_config, saved_events = copy.deepcopy(config), copy.deepcopy(events)
    s = initialize_session(config)
    s_dict = s.to_dict()
    belief0 = infer_belief_state(s).to_dict()
    s2 = rec(s, *events)
    predict_next_hand(s2)
    to_game_state(s2).turn.hand.append(G)  # mutating the output must not leak back
    with pytest.raises(HandNotDetermined):
        plan_next_action(s2)
    assert config == saved_config and events == saved_events
    assert s.to_dict() == s_dict and infer_belief_state(s).to_dict() == belief0
    assert G not in to_game_state(s2).turn.hand


# 13 + 14 -------------------------------------------------------------------------
def test_plan_on_inferred_belief_state(engine):
    s = spec_turn3_observed()
    state = to_game_state(s)
    assert not has_errors(validate_state(state, engine))
    assert dict(state.draw_pile.rest) == {C: 4, E: 2, S: 1} and state.deck_count == 7 and state.discard_count == 0
    result = plan_next_action(s, PlanConfig(max_nodes=3000))
    assert result.recommended is not None
    legal = {a.describe() for a in engine.legal_actions(state)}
    assert result.recommended.action["label"] in legal
    assert result.information_level == "known_composition"


def test_plan_refuses_when_hand_not_determined():
    with pytest.raises(HandNotDetermined, match="pending"):
        plan_next_action(spec_through_turn2())
    with pytest.raises(HandNotDetermined, match="not determined"):
        plan_next_action(rec(spec_turn3_observed(), EndTurnObservation(), RecordDraw(count=5)))


def test_plan_search_limits_respected(engine):
    s = record_event(spec_turn3_observed(), RecordPlay(C))  # Buy phase, many candidate buys
    result = plan_next_action(s, {"max_nodes": 5})
    assert result.stats["budget_exhausted"] is True
    assert result.recommended is not None
    legal = {a.describe() for a in engine.legal_actions(to_game_state(s))}
    assert result.recommended.action["label"] in legal


def test_to_game_state_valid_for_mixture(engine):
    s = _turn5_after_unobserved_turn4()
    state = to_game_state(s)
    assert not has_errors(validate_state(state, engine))
    assert state.draw_pile.size + state.discard_pile.size == 12 - len(state.turn.hand)
    assert any("mixture" in a for a in state.assumptions)


# snapshot start / edits ----------------------------------------------------------
def test_mid_game_partial_snapshot_start(engine):
    snap = {"schema_version": 1, "num_players": 2, "kingdom": FIRST_GAME, "turn_number": 7,
            "player": {"phase": "action", "hand": [G, S, C, C, E], "owned_cards": {C: 7, E: 3, S: 2, G: 1},
                       "discard_pile": {"composition": {C: 3, E: 1}}}}
    s = initialize_session({"snapshot": snap})
    b = infer_belief_state(s)
    assert b.turn == 7 and b.determined_hand() == sorted([G, S, C, C, E])
    assert dict(b.worlds[0].draw.rest) == {C: 2, E: 1, S: 1} and b.zone_status("draw_pile") == "deduced"
    assert any(f.startswith("started_from_snapshot") for f in b.flags)
    s = rec(s, RecordPurchase(G), EndTurnObservation())
    p = predict_next_hand(s)  # 4 known cards then reshuffle of 10
    assert p.p_reshuffle == 1 and sum(dist(p).values()) == 1
    assert not has_errors(validate_state(to_game_state(s), engine))

    unknown = {"schema_version": 1, "num_players": 2, "kingdom": FIRST_GAME,
               "player": {"phase": "action", "hand": [C, C, C, E, E], "deck_count": 6}}
    s2 = initialize_session({"snapshot": unknown})
    b2 = infer_belief_state(s2)
    assert b2.zone_status("draw_pile") == "unknown" and b2.zone_status("discard_pile") == "unknown"
    assert b2.worlds[0].draw.unknown == 6 and b2.owned is None
    s2 = rec(s2, EndTurnObservation())
    p2 = predict_next_hand(s2)
    assert p2.lower_bound and p2.distribution[0][0] == ("?",) * 5
    b3 = infer_belief_state(record_event(s2, ObserveHand([S, C, C, C, C])))  # consistent with "?"
    assert b3.determined_hand() == [C, C, C, C, S] and b3.worlds[0].draw.unknown == 1


def test_correct_remove_and_reset():
    s = rec(initialize_session(), ObserveHand([C, C, C, C, E]), RecordPurchase(S), EndTurnObservation())
    fixed = correct_event(s, 0, ObserveHand([C, C, C, E, E]))
    assert predict_next_hand(fixed).hand == hand(Copper=4, Estate=1)
    with pytest.raises(InconsistentObservation):  # 2C+3E cannot pay for the recorded Silver
        correct_event(s, 0, ObserveHand([C, C, E, E, E]))
    assert s.events[0] == ObserveHand([C, C, C, C, E])
    removed = remove_event(s, 1)
    assert S not in infer_belief_state(removed).worlds[0].discard.counter
    assert len(removed.events) == 2
    with pytest.raises(IndexError):
        remove_event(s, 3)
    r = reset_session(s)
    assert r.events == () and infer_belief_state(r) == infer_belief_state(initialize_session())


def test_gain_trash_supply_events():
    s = rec(initialize_session(), ObserveHand([C, C, C, C, E]), RecordGain(G, "deck"), RecordTrash(E, "hand"))
    b = infer_belief_state(s)
    assert b.worlds[0].draw.top == (G,) and b.owned == {C: 7, E: 2, G: 1} and b.trash == (E,)
    assert b.game.supply[G] == 29
    s = record_event(s, RecordSupply("Province", 5))
    b2 = infer_belief_state(s)
    assert b2.game.supply["Province"] == 5 and b2.owned == b.owned and b2.worlds == b.worlds
    _assert_rejected(s, RecordSupply("Province", 9), InconsistentObservation, "starts with only 8")
    _assert_rejected(s, RecordTrash(S, "hand"), InconsistentObservation, "Silver")


def test_action_draws_and_mid_turn_reshuffle_exclude_cards_in_play():
    snap = {"schema_version": 1, "num_players": 2, "kingdom": FIRST_GAME, "turn_number": 4,
            "player": {"phase": "action", "hand": ["Smithy", C, C, C, C],
                       "owned_cards": {C: 7, E: 3, "Smithy": 1},
                       "draw_pile": {"composition": {E: 1}}, "discard_pile": {"composition": {C: 3, E: 2}}}}
    s = record_event(initialize_session({"snapshot": snap}), RecordPlay("Smithy"))
    b = infer_belief_state(s)
    assert b.pending_draws == 3 and b.in_play == ("Smithy",)
    unseen = infer_belief_state(record_event(s, RecordDraw(count=3)))
    # 1 Estate from the deck, then 2 of the reshuffled 3C+2E; Smithy (in play) is never shuffled in
    assert {w.hand: w.weight for w in unseen.worlds} == {
        ((C, 6), (E, 1)): Fraction(3, 10), ((C, 5), (E, 2)): Fraction(6, 10), ((C, 4), (E, 3)): Fraction(1, 10)}
    assert all("Smithy" not in (w.zone_counter("draw_pile") or {}) for w in unseen.worlds)
    seen = infer_belief_state(record_event(s, RecordDraw([E, C, E])))
    assert len(seen.worlds) == 1 and dict(seen.worlds[0].draw.rest) == {C: 2, E: 1}
    ended = infer_belief_state(rec(s, RecordDraw([E, C, E]), EndTurnObservation()))
    assert ended.worlds[0].discard.counter["Smithy"] == 1


def test_unknown_card_cannot_match_a_card_known_to_be_elsewhere(engine):
    # max_worlds=1 collapses the turn-3 unobserved hand into "?" cards. The only Smithy is then gained to the
    # discard pile, so a turn-4 hand containing Smithy is impossible (it must not match a "?" in the draw pile).
    events = [ObserveHand([C, C, C, E, E]), RecordPurchase(S), EndTurnObservation(), EndTurnObservation(),
              RecordGain("Smithy"), EndTurnObservation()]
    for max_worlds in (1, 5000):
        s = rec(initialize_session({"max_worlds": max_worlds}), *events)
        with pytest.raises(InconsistentObservation) as err:
            record_event(s, ObserveHand([S, C, C, "Smithy", E]))
        assert err.value.event_index == len(events) and "Smithy" in str(err.value)
    s = rec(initialize_session({"max_worlds": 1}), *events, ObserveHand([S, C, C, C, E]))
    assert not has_errors(validate_state(to_game_state(s), engine))

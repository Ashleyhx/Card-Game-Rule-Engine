"""Pure planning transitions: immutability, card conservation, resources, manual-step handling."""

import copy
from collections import Counter

import pytest

from card_game_engine.games.dominion.planning import (
    BOUNDARY_UNKNOWN_DRAW_PILE, card_totals, enumerate_draws, from_game_state, pending_draws, simulate_action,
    simulate_draw, simulate_end_turn, simulate_purchase, simulate_treasure,
)
from card_game_engine.games.dominion.rules import (
    Buy, EndActionPhase, IllegalActionError, PlayAction, state_from_dict,
)

KINGDOM = ["Bandit", "Bureaucrat", "Chapel", "Library", "Market", "Militia", "Smithy", "Throne Room", "Village", "Witch"]


def plan(engine, **player):
    snap = {"schema_version": 1, "num_players": 2, "kingdom": KINGDOM,
            "player": {"phase": "action", "hand": [], **player}}
    return from_game_state(state_from_dict(snap), engine)


def draw_first(ps, engine):
    """Resolve pending draws with the most likely outcome."""
    return simulate_draw(ps, enumerate_draws(ps.draw, ps.discard, pending_draws(ps))[0], engine)


def test_full_turn_is_pure_and_conserves_cards(engine):
    ps = plan(engine, hand=["Village", "Smithy", "Copper", "Copper", "Estate"],
              draw_pile={"composition": {"Copper": 3, "Silver": 2, "Estate": 2}},
              discard_pile={"composition": {"Estate": 1}})
    totals = card_totals(ps)
    steps = [
        lambda s: simulate_action(s, PlayAction("Village"), engine),
        lambda s: draw_first(s, engine),
        lambda s: simulate_action(s, PlayAction("Smithy"), engine),
        lambda s: draw_first(s, engine),
        lambda s: simulate_action(s, EndActionPhase(), engine),
        lambda s: simulate_treasure(s, None, engine),
        lambda s: simulate_end_turn(s, engine),
        lambda s: draw_first(s, engine),
    ]
    for step in steps:
        before = copy.deepcopy(ps)
        nxt = step(ps)
        assert ps == before
        assert card_totals(nxt) == totals
        ps = nxt
    assert ps.turn_ended and len(ps.next_hand) == 5 and pending_draws(ps) == 0


def test_village_and_market_resources(engine):
    ps = plan(engine, hand=["Village", "Market"], draw_pile={"composition": {"Copper": 5}})
    v = simulate_action(ps, PlayAction("Village"), engine)
    assert (v.game.turn.actions, pending_draws(v)) == (2, 1)
    m = simulate_action(draw_first(v, engine), PlayAction("Market"), engine)
    t = m.game.turn
    assert (t.actions, t.buys, t.coins, pending_draws(m)) == (2, 2, 1, 1)


def test_purchase_goes_to_discard(engine):
    ps = plan(engine, phase="buy", in_play=["Gold", "Silver"], coins=5, discard_pile={"composition": {"Copper": 1}},
              draw_pile={"composition": {}})
    after = simulate_purchase(ps, "silver", engine)
    t = after.game.turn
    assert (t.coins, t.buys, t.bought) == (2, 0, ["Silver"])
    assert after.discard.counter == {"Copper": 1, "Silver": 1}
    assert after.game.supply["Silver"] == ps.game.supply["Silver"] - 1
    assert card_totals(after) - card_totals(ps) == Counter({"Silver": 1})
    assert after.game.discard_count == 2


def test_illegal_actions_raise(engine):
    ps = plan(engine, hand=["Copper"])
    with pytest.raises(IllegalActionError):
        simulate_action(ps, PlayAction("Smithy"), engine)
    with pytest.raises(IllegalActionError):
        simulate_action(ps, Buy("Province"), engine)
    buy = simulate_action(ps, EndActionPhase(), engine)
    with pytest.raises(IllegalActionError):
        simulate_purchase(buy, "Province", engine)


def test_manual_card_is_a_boundary(engine):
    ps = simulate_action(plan(engine, hand=["Chapel", "Estate"]), PlayAction("Chapel"), engine)
    assert ps.boundary == "manual:Chapel"
    assert engine.legal_actions(ps.game) == []


@pytest.mark.parametrize("action,coins", [(PlayAction("Militia"), 2), (PlayAction("Throne Room", throne_target="Militia"), 4)])
def test_opponent_only_steps_cleared(engine, action, coins):
    ps = simulate_action(plan(engine, hand=["Throne Room", "Militia"]), action, engine)
    assert ps.game.turn.unresolved == [] and ps.boundary is None
    assert ps.game.turn.coins == coins
    assert engine.legal_actions(ps.game)


def test_witch_attack_cleared_draws_pending(engine):
    ps = simulate_action(plan(engine, hand=["Witch"], deck_count=4), PlayAction("Witch"), engine)
    assert ps.game.turn.unresolved == [] and pending_draws(ps) == 2 and ps.boundary is None


def test_library_draws_to_seven_keeping_everything(engine):
    ps = plan(engine, hand=["Library", "Copper", "Copper", "Estate", "Estate"],
              draw_pile={"composition": {"Copper": 5, "Smithy": 1}})
    played = simulate_action(ps, PlayAction("Library"), engine)
    assert pending_draws(played) == 3 and played.game.turn.unresolved == [] and played.boundary is None
    for outcome in enumerate_draws(played.draw, played.discard, 3):
        after = simulate_draw(played, outcome, engine)
        assert len(after.game.turn.hand) == 7 and card_totals(after) == card_totals(ps)


@pytest.mark.parametrize("action", [PlayAction("Library"), PlayAction("Throne Room", throne_target="Library")])
def test_library_counts_unknown_cards_in_hand(engine, action):
    ps = plan(engine, hand=["Village", "Throne Room", "Library", "Copper", "Copper"], draw_pile={"unknown_count": 10})
    ps = draw_first(simulate_action(ps, PlayAction("Village"), engine), engine)
    assert ps.unknown_in_hand == 1
    played = simulate_action(ps, action, engine)
    t = played.game.turn
    assert len(t.hand) + played.unknown_in_hand + pending_draws(played) == 7


def test_bureaucrat_silver_onto_deck(engine):
    ps = plan(engine, hand=["Bureaucrat", "Throne Room"], draw_pile={"order": ["Gold"]})
    once = simulate_action(ps, PlayAction("Bureaucrat"), engine)
    assert once.draw.top == ("Silver", "Gold") and once.game.turn.unresolved == []
    assert card_totals(once) - card_totals(ps) == Counter({"Silver": 1})
    twice = simulate_action(ps, PlayAction("Throne Room", throne_target="Bureaucrat"), engine)
    assert twice.draw.top == ("Silver", "Silver", "Gold") and twice.game.deck_count == 3


def test_bandit_gold_to_discard(engine):
    ps = plan(engine, hand=["Bandit"], draw_pile={"order": []}, discard_pile={"composition": {}})
    after = simulate_action(ps, PlayAction("Bandit"), engine)
    assert after.discard.counter == {"Gold": 1} and after.boundary is None
    assert card_totals(after) - card_totals(ps) == Counter({"Gold": 1})


def test_unknown_cards_drawn_as_placeholders(engine):
    ps = simulate_action(plan(engine, hand=["Smithy"], deck_count=3), PlayAction("Smithy"), engine)
    outcomes = enumerate_draws(ps.draw, ps.discard, 3)
    assert [o.cards for o in outcomes] == [("?", "?", "?")]
    after = simulate_draw(ps, outcomes[0], engine)
    assert after.unknown_in_hand == 3 and after.game.turn.hand == [] and pending_draws(after) == 0
    assert card_totals(after) == card_totals(ps) == Counter({"?": 3, "Smithy": 1})


def test_unknown_draw_pile_is_a_boundary(engine):
    ps = simulate_action(plan(engine, hand=["Smithy"]), PlayAction("Smithy"), engine)
    after = simulate_draw(ps, enumerate_draws(ps.draw, ps.discard, 3)[0], engine)
    assert after.boundary == BOUNDARY_UNKNOWN_DRAW_PILE and pending_draws(after) == 3


def test_end_turn_moves_cards_and_draws_next_hand_with_reshuffle(engine):
    ps = plan(engine, phase="buy", hand=["Estate"], in_play=["Silver", "Copper"], coins=3,
              draw_pile={"order": ["Gold", "Copper"]}, discard_pile={"composition": {"Estate": 2}})
    bought = simulate_purchase(ps, "Silver", engine)
    ended = simulate_end_turn(bought, engine)
    assert ended.turn_ended and pending_draws(ended) == 5 and ended.game.turn.hand == []
    assert ended.discard.counter == {"Estate": 3, "Silver": 2, "Copper": 1}
    assert card_totals(ended) == card_totals(bought)
    outcomes = enumerate_draws(ended.draw, ended.discard, 5)
    assert all(o.reshuffled for o in outcomes) and sum(o.p_exact for o in outcomes) == 1
    nxt = simulate_draw(ended, outcomes[0], engine)
    assert len(nxt.next_hand) == 5 and {"Gold", "Copper"} <= set(nxt.next_hand)
    assert card_totals(nxt) == card_totals(bought)


def test_simulate_draw_requires_pending(engine):
    ps = plan(engine, hand=["Copper"], draw_pile={"order": ["Gold"]})
    with pytest.raises(ValueError):
        simulate_draw(ps, enumerate_draws(ps.draw, ps.discard, 1)[0], engine)


def test_key_is_canonical_and_hashable(engine):
    a = plan(engine, hand=["Copper", "Smithy", "Estate"], draw_pile={"composition": {"Copper": 2}})
    b = plan(engine, hand=["Estate", "Copper", "Smithy"], draw_pile={"composition": {"Copper": 2}})
    assert a.key() == b.key() and hash(a.key()) == hash(b.key())
    c = plan(engine, hand=["Copper", "Smithy", "Estate"], draw_pile={"composition": {"Copper": 3}})
    assert a.key() != c.key()

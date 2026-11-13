"""Legal-action enumeration, constraint validation and transitions."""

import pytest

from card_game_engine.games.dominion.rules import (
    Buy, EndActionPhase, EndTurn, IllegalActionError, Phase, PlayAction, PlayAllTreasures, PlayTreasure,
)
from tests.dominion.conftest import OTHER_KINGDOM


def kinds(actions):
    return {(a.kind, getattr(a, "card", None)) for a in actions}


# ---------------------------------------------------------------- Action phase

def test_action_phase_lists_each_action_card_and_end(engine, make_state):
    s = make_state(hand=["Village", "Smithy", "Copper", "Estate", "Copper"])
    legal = engine.legal_actions(s)
    assert kinds(legal) == {("play_action", "Village"), ("play_action", "Smithy"), ("end_action_phase", None)}


def test_no_actions_left_only_end_phase(engine, make_state):
    s = make_state(hand=["Smithy"], in_play=["Smithy"], actions=0)
    assert engine.legal_actions(s) == [EndActionPhase()]
    assert "No Actions remaining." in engine.check(s, PlayAction("Smithy"))


def test_cannot_play_card_not_in_hand_or_non_action(engine, make_state):
    s = make_state(hand=["Copper"])
    assert any("not in your hand" in r for r in engine.check(s, PlayAction("Smithy")))
    assert any("not an Action" in r for r in engine.check(s, PlayAction("Copper")))


def test_treasures_not_playable_in_action_phase(engine, make_state):
    s = make_state(hand=["Copper"])
    assert any("Buy phase" in r for r in engine.check(s, PlayTreasure("Copper")))


def test_village_effects_and_pending_draw(engine, make_state):
    s = make_state(hand=["Village", "Smithy"])
    t = engine.apply(s, PlayAction("Village")).state
    assert t.turn.actions == 2 and t.turn.pending_draws == 1
    assert t.turn.in_play == ["Village"] and t.turn.hand == ["Smithy"]
    # Must draw before anything else is legal
    assert engine.legal_actions(t) == []
    t2 = engine.resolve_pending(t, drawn=["Copper"])
    assert sorted(t2.turn.hand) == ["Copper", "Smithy"]
    assert PlayAction("Smithy") in engine.legal_actions(t2)


def test_cannot_draw_more_than_pending(engine, make_state):
    t = engine.apply(make_state(hand=["Village"]), PlayAction("Village")).state
    with pytest.raises(ValueError):
        engine.resolve_pending(t, drawn=["Copper", "Copper"])


def test_festival_market(engine, make_state):
    s = make_state(hand=["Festival", "Market"])
    s = engine.apply(s, PlayAction("Festival")).state
    assert (s.turn.actions, s.turn.buys, s.turn.coins) == (2, 2, 2)
    s = engine.apply(s, PlayAction("Market")).state
    assert (s.turn.actions, s.turn.buys, s.turn.coins, s.turn.pending_draws) == (2, 3, 3, 1)


def test_apply_illegal_raises(engine, make_state):
    s = make_state(hand=["Copper"])
    with pytest.raises(IllegalActionError):
        engine.apply(s, PlayAction("Smithy"))


def test_original_state_unchanged_by_apply(engine, make_state):
    s = make_state(hand=["Village"])
    engine.apply(s, PlayAction("Village"))
    assert s.turn.hand == ["Village"] and s.turn.actions == 1


def test_manual_effect_blocks_until_resolved(engine, make_state):
    s = make_state(hand=["Remodel", "Estate"])
    t = engine.apply(s, PlayAction("Remodel")).state
    assert t.turn.unresolved and engine.legal_actions(t) == []
    t2 = engine.resolve_pending(t, hand=[])
    assert engine.legal_actions(t2) == [EndActionPhase()]


# ---------------------------------------------------------------- special cards

def test_merchant_bonus_first_silver_only(engine, make_state):
    s = make_state(hand=["Merchant", "Merchant", "Silver", "Silver"])
    s = engine.resolve_pending(engine.apply(s, PlayAction("Merchant")).state, drawn=[])
    s = engine.resolve_pending(engine.apply(s, PlayAction("Merchant")).state, drawn=[])
    s = engine.apply(s, EndActionPhase()).state
    s = engine.apply(s, PlayTreasure("Silver")).state
    assert s.turn.coins == 2 + 2  # two Merchants each add +$1 to the first Silver
    s = engine.apply(s, PlayTreasure("Silver")).state
    assert s.turn.coins == 6


def test_merchant_after_silver_gives_nothing(engine, make_state):
    s = make_state(phase="buy", hand=["Silver"], in_play=["Silver"], coins=2, silver_played=True,
                   merchants_played=1)
    s = engine.apply(s, PlayTreasure("Silver")).state
    assert s.turn.coins == 4


def test_moneylender_variants(engine, make_state):
    s = make_state(kingdom=OTHER_KINGDOM, hand=["Moneylender", "Copper", "Estate"],
                   owned_cards={"Moneylender": 1, "Copper": 7, "Estate": 3})
    legal = engine.legal_actions(s)
    assert PlayAction("Moneylender", trash_copper=True) in legal
    assert PlayAction("Moneylender", trash_copper=False) in legal
    t = engine.apply(s, PlayAction("Moneylender", trash_copper=True)).state
    assert t.turn.coins == 3 and "Copper" not in t.turn.hand
    assert t.trash == ["Copper"] and t.owned_cards["Copper"] == 6
    t2 = engine.apply(s, PlayAction("Moneylender", trash_copper=False)).state
    assert t2.turn.coins == 0


def test_moneylender_without_copper(engine, make_state):
    s = make_state(kingdom=OTHER_KINGDOM, hand=["Moneylender", "Estate"])
    assert engine.legal_actions(s) == [PlayAction("Moneylender", trash_copper=False), EndActionPhase()]


def test_throne_room_targets_and_double_effect(engine, make_state):
    s = make_state(kingdom=OTHER_KINGDOM, hand=["Throne Room", "Festival", "Copper"])
    legal = engine.legal_actions(s)
    assert PlayAction("Throne Room", throne_target="Festival") in legal
    assert PlayAction("Throne Room") in legal
    t = engine.apply(s, PlayAction("Throne Room", throne_target="Festival")).state
    # 1 - 1 (Throne Room) + 2 + 2 = 4 actions; Festival's play does not use Actions
    assert (t.turn.actions, t.turn.buys, t.turn.coins) == (4, 3, 4)
    assert sorted(t.turn.in_play) == ["Festival", "Throne Room"]


def test_throne_room_needs_another_copy_to_target_itself(engine, make_state):
    s = make_state(kingdom=OTHER_KINGDOM, hand=["Throne Room", "Copper"])
    assert engine.legal_actions(s) == [PlayAction("Throne Room"), EndActionPhase()]
    s2 = make_state(kingdom=OTHER_KINGDOM, hand=["Throne Room", "Throne Room"])
    assert PlayAction("Throne Room", throne_target="Throne Room") in engine.legal_actions(s2)


def test_throne_room_smithy_draws_six(engine, make_state):
    s = make_state(hand=["Throne Room", "Smithy"], kingdom=["Throne Room", "Smithy", "Cellar", "Chapel", "Moat",
                                                          "Village", "Market", "Mine", "Witch", "Artisan"])
    t = engine.apply(s, PlayAction("Throne Room", throne_target="Smithy")).state
    assert t.turn.pending_draws == 6 and t.turn.actions == 0


def test_poacher_discards_per_empty_pile(engine, make_state):
    s = make_state(kingdom=OTHER_KINGDOM, hand=["Poacher"], supply={"Chapel": 0, "Witch": 0})
    t = engine.apply(s, PlayAction("Poacher")).state
    assert t.turn.coins == 1 and t.turn.actions == 1
    assert any("discard 2" in u for u in t.turn.unresolved)
    s2 = make_state(kingdom=OTHER_KINGDOM, hand=["Poacher"])
    assert engine.apply(s2, PlayAction("Poacher")).state.turn.unresolved == []


def test_library_draws_to_seven(engine, make_state):
    s = make_state(kingdom=OTHER_KINGDOM, hand=["Library", "Copper", "Copper", "Estate", "Estate"])
    t = engine.apply(s, PlayAction("Library")).state
    assert t.turn.pending_draws == 3  # 4 cards left in hand -> draw 3


def test_bandit_gains_gold(engine, make_state):
    s = make_state(kingdom=OTHER_KINGDOM, hand=["Bandit"], owned_cards={"Bandit": 1, "Copper": 7, "Estate": 3})
    t = engine.apply(s, PlayAction("Bandit")).state
    assert t.supply["Gold"] == 29 and t.owned_cards["Gold"] == 1
    assert any("Bandit attack" in u for u in t.turn.unresolved)


def test_bandit_no_gold_left(engine, make_state):
    s = make_state(kingdom=OTHER_KINGDOM, hand=["Bandit"], supply={"Gold": 0})
    tr = engine.apply(s, PlayAction("Bandit"))
    assert tr.state.supply["Gold"] == 0
    assert any("no Gold left" in n for n in tr.notes)


def test_witch_creates_manual_step(engine, make_state):
    s = make_state(kingdom=OTHER_KINGDOM, hand=["Witch"])
    t = engine.apply(s, PlayAction("Witch")).state
    assert t.turn.pending_draws == 2 and any("Curse" in u for u in t.turn.unresolved)


# ---------------------------------------------------------------- Buy phase

def test_buy_phase_enumeration(engine, make_state):
    s = make_state(phase="buy", hand=["Copper", "Copper", "Silver", "Estate"])
    legal = engine.legal_actions(s)
    assert PlayAllTreasures() in legal
    assert PlayTreasure("Copper") in legal and PlayTreasure("Silver") in legal
    # With $0, only $0 cards are buyable
    buys = {a.card for a in legal if a.kind == "buy"}
    assert buys == {"Copper", "Curse"}
    assert EndTurn() in legal


def test_play_all_treasures_then_buy(engine, make_state):
    s = make_state(phase="buy", hand=["Copper", "Copper", "Silver", "Silver", "Estate"])
    s = engine.apply(s, PlayAllTreasures()).state
    assert s.turn.coins == 6 and s.turn.hand == ["Estate"]
    assert Buy("Gold") in engine.legal_actions(s)
    assert Buy("Province") not in engine.legal_actions(s)
    assert "Costs $8; you have $6." in engine.check(s, Buy("Province"))
    s = engine.apply(s, Buy("Gold")).state
    assert s.turn.coins == 0 and s.turn.buys == 0 and s.supply["Gold"] == 29
    assert engine.legal_actions(s) == [EndTurn()]


def test_no_treasures_after_buying(engine, make_state):
    s = make_state(phase="buy", hand=["Copper", "Silver"], coins=0)
    s = engine.apply(s, Buy("Copper")).state  # buying a $0 card with $0 is legal
    assert any("after buying" in r for r in engine.check(s, PlayTreasure("Silver")))
    assert any("after buying" in r for r in engine.check(s, PlayAllTreasures()))


def test_multiple_buys_split_coins(engine, make_state):
    s = make_state(phase="buy", hand=[], in_play=["Festival", "Gold", "Gold"], coins=8, buys=2, actions=2,
                   kingdom=OTHER_KINGDOM)
    s = engine.apply(s, Buy("Silver")).state
    assert s.turn.coins == 5 and s.turn.buys == 1
    assert Buy("Laboratory") in engine.legal_actions(s)
    s = engine.apply(s, Buy("Laboratory")).state
    assert s.turn.coins == 0 and s.turn.buys == 0


def test_cannot_buy_from_empty_pile_or_outside_supply(engine, make_state):
    s = make_state(phase="buy", coins=10, supply={"Smithy": 0})
    assert "The Smithy pile is empty." in engine.check(s, Buy("Smithy"))
    assert any("not in this game's Supply" in r for r in engine.check(s, Buy("Witch")))


def test_end_turn_only_from_buy_phase(engine, make_state):
    s = make_state(hand=["Copper"])
    assert engine.check(s, EndTurn())
    s = engine.apply(s, EndActionPhase()).state
    assert s.turn.phase == Phase.BUY
    t = engine.apply(s, EndTurn()).state
    assert t.turn.phase == Phase.TURN_OVER and engine.legal_actions(t) == []


# ---------------------------------------------------------------- end of game & scoring

def test_game_end_on_last_province(engine, make_state):
    s = make_state(phase="buy", coins=8, supply={"Province": 1})
    assert not engine.game_end_status(s).game_over
    s = engine.apply(s, Buy("Province")).state
    end = engine.game_end_status(s)
    assert end.game_over and "Province" in end.reasons[0]
    notes = engine.apply(s, EndTurn()).notes
    assert any("GAME OVER" in n for n in notes)


def test_game_end_three_piles(engine, make_state):
    s = make_state(phase="buy", coins=3, supply={"Cellar": 0, "Moat": 0, "Village": 1})
    assert not engine.game_end_status(s).game_over
    s = engine.apply(s, Buy("Village")).state
    assert engine.game_end_status(s).game_over


def test_two_empty_piles_not_game_over(engine, make_state):
    s = make_state(supply={"Cellar": 0, "Curse": 0})
    assert not engine.game_end_status(s).game_over


def test_victory_points_including_gardens(engine):
    owned = {"Copper": 20, "Estate": 3, "Duchy": 2, "Province": 1, "Curse": 2, "Gardens": 2}
    # 30 cards -> each Gardens worth 3
    assert engine.victory_points(owned) == 3 + 6 + 6 - 2 + 6

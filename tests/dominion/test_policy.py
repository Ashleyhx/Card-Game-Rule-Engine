"""Decision policy behaviour on representative situations."""

import json
from importlib import resources

import pytest

from card_game_engine.games.dominion.policy import HeuristicPolicy, build_context
from card_game_engine.games.dominion.rules import Buy, EndActionPhase, PlayAction, PlayAllTreasures, DominionRuleEngine, state_from_dict
from tests.dominion.conftest import OTHER_KINGDOM


def load_example(name):
    path = resources.files("card_game_engine.games.dominion.examples").joinpath(f"{name}.json")
    return state_from_dict(json.loads(path.read_text()))


@pytest.fixture(scope="module")
def policy():
    return HeuristicPolicy(DominionRuleEngine())


def best_buy(advice):
    ranked = advice.projected or advice.ranked
    return ranked[0].action


def test_only_ranks_legal_actions(policy, make_state):
    s = make_state(phase="buy", hand=["Silver", "Copper"], coins=0)
    legal = policy.engine.legal_actions(s)
    for r in policy.recommend(s).ranked:
        assert r.action in legal


def test_treasures_first_then_projected_buys(policy):
    a = policy.recommend(load_example("opening_5_2"))
    assert a.best.action == PlayAllTreasures()
    assert a.projected_coins == 5
    assert all(policy.engine.card(r.action.card).cost <= 5 for r in a.projected if isinstance(r.action, Buy))


def test_village_before_smithy(policy):
    a = policy.recommend(load_example("village_smithy_action"))
    assert a.best.action == PlayAction("Village")
    assert any("Sequencing" in r for r in a.best.reasons)
    smithy = next(r for r in a.ranked if r.action == PlayAction("Smithy"))
    assert any("play Village first" in r for r in smithy.reasons)


def test_throne_room_prefers_smithy(policy):
    a = policy.recommend(load_example("throne_room_moneylender"))
    assert a.best.action == PlayAction("Throne Room", throne_target="Smithy")


def test_penultimate_province_rule(policy):
    a = policy.recommend(load_example("endgame_penultimate_province"))
    assert best_buy(a) == Buy("Duchy")
    prov = next(r for r in a.projected if r.action == Buy("Province"))
    assert any("Penultimate Province" in r for r in prov.reasons)


def test_winning_last_province(policy):
    a = policy.recommend(load_example("endgame_winning_province"))
    assert best_buy(a) == Buy("Province")
    assert any("winning buy" in r for r in a.projected[0].reasons)
    assert a.confidence == "high"


def test_three_pile_win(policy):
    a = policy.recommend(load_example("three_pile_ending"))
    assert best_buy(a) == Buy("Village")


def test_avoid_losing_game_end(policy, make_state):
    s = make_state(phase="buy", coins=8, supply={"Province": 1},
                   owned_cards={"Copper": 7, "Estate": 3, "Province": 2}, opponents=[{"name": "X", "vp": 40}])
    a = policy.recommend(s)
    assert a.best.action != Buy("Province")
    prov = next(r for r in a.ranked if r.action == Buy("Province"))
    assert prov.score < 0 and any("you would lose" in r for r in prov.reasons)


def test_game_ending_gardens_buy_recounts_owned_gardens(policy, make_state):
    # 19 cards with 2 Gardens (5 VP). Buying the last Gardens makes 20 cards: 3 Gardens x 2 + 3 Estates = 9 VP.
    s = make_state(kingdom=OTHER_KINGDOM, phase="buy", coins=4, supply={"Gardens": 1, "Curse": 0, "Poacher": 0},
                   owned_cards={"Copper": 7, "Estate": 3, "Silver": 7, "Gardens": 2}, opponents=[{"name": "X", "vp": 8}])
    gardens = next(r for r in policy.recommend(s).ranked if r.action == Buy("Gardens"))
    assert any("winning buy" in r and "9 vs 8" in r for r in gardens.reasons)


def test_unknown_vp_game_end_warns(policy, make_state):
    s = make_state(phase="buy", coins=8, supply={"Province": 1})
    prov = next(r for r in policy.recommend(s).ranked if r.action == Buy("Province"))
    assert any("VP totals are unknown" in w for w in prov.warnings)


def test_gold_over_province_very_early_without_economy(policy, make_state):
    s = make_state(phase="buy", coins=8, turn_number=3, owned_cards={"Copper": 7, "Estate": 3, "Silver": 1},
                   opponents=[{"name": "X", "vp": 3}])
    assert policy.recommend(s).best.action == Buy("Gold")


def test_duchy_over_gold_late(policy, make_state):
    s = make_state(phase="buy", coins=6, supply={"Province": 3},
                   owned_cards={"Copper": 7, "Estate": 3, "Silver": 4, "Gold": 2, "Province": 5},
                   opponents=[{"name": "X", "vp": 20}])
    assert policy.recommend(s).best.action == Buy("Duchy")


def test_terminal_collision_penalty(policy, make_state):
    s = make_state(phase="buy", coins=4, turn_number=8,
                   owned_cards={"Copper": 7, "Estate": 3, "Silver": 2, "Smithy": 2, "Militia": 1})
    smithy = next(r for r in policy.recommend(s).ranked if r.action == Buy("Smithy"))
    assert any("Terminal collision" in r for r in smithy.reasons)


def test_witch_without_curses(policy, make_state):
    s = make_state(kingdom=OTHER_KINGDOM, phase="buy", coins=5, supply={"Curse": 0},
                   owned_cards={"Copper": 7, "Estate": 3, "Silver": 2})
    witch = next(r for r in policy.recommend(s).ranked if r.action == Buy("Witch"))
    assert any("No Curses left" in r for r in witch.reasons)


def test_moneylender_trash_preferred_early(policy, make_state):
    s = make_state(kingdom=OTHER_KINGDOM, hand=["Moneylender", "Copper", "Copper", "Estate", "Estate"],
                   owned_cards={"Copper": 7, "Estate": 3, "Moneylender": 1}, turn_number=3)
    assert policy.recommend(s).best.action == PlayAction("Moneylender", trash_copper=True)


def test_end_action_phase_when_nothing_playable(policy, make_state):
    s = make_state(hand=["Copper", "Estate"])
    a = policy.recommend(s)
    assert a.best.action == EndActionPhase() and a.confidence == "high"


def test_blocked_state_gives_no_ranking(policy, make_state):
    s = make_state(hand=["Copper"], pending_draws=2)
    a = policy.recommend(s)
    assert a.ranked == [] and a.blocking


def test_plan_covers_multi_buy(policy):
    a = policy.recommend(load_example("multi_buy_festival"))
    assert a.plan[0] == "Play all Treasures"
    assert a.plan[-1].startswith("End turn")
    assert sum(1 for p in a.plan if p.startswith("Buy ")) >= 2


def test_plan_stops_at_hidden_information(policy):
    a = policy.recommend(load_example("village_smithy_action"))
    assert a.plan[0] == "Play Village" and "Draw 1" in a.plan[1]


def test_uncertainty_reported_for_minimal_snapshot(policy):
    a = policy.recommend(load_example("unknown_deck_minimal"))
    text = " ".join(a.uncertainties)
    assert "Deck composition unknown" in text and "Supply counts were not provided" in text
    assert a.confidence in ("medium", "low")


def test_deterministic(policy):
    s = load_example("witch_midgame_buy")
    a1, a2 = policy.recommend(s).to_dict(), policy.recommend(s).to_dict()
    assert a1 == a2


def test_stage_detection(make_state):
    eng = DominionRuleEngine()
    assert build_context(make_state(turn_number=1, owned_cards={"Copper": 7, "Estate": 3}), eng).stage == "early"
    assert build_context(make_state(supply={"Province": 5}), eng).stage == "dance"
    assert build_context(make_state(supply={"Province": 4}), eng).stage == "late"
    assert build_context(make_state(supply={"Province": 2}), eng).stage == "endgame"
    assert build_context(make_state(supply={"Cellar": 0, "Moat": 0, "Village": 2}), eng).stage == "endgame"


@pytest.mark.parametrize("name", [
    "opening_5_2", "village_smithy_action", "witch_midgame_buy", "throne_room_moneylender",
    "multi_buy_festival", "endgame_penultimate_province", "endgame_winning_province",
    "three_pile_ending", "unknown_deck_minimal",
])
def test_every_example_produces_advice(policy, name):
    a = policy.recommend(load_example(name))
    assert a.best is not None and a.plan
    json.dumps(a.to_dict())  # serialisable


def test_visible_gold_means_not_early(policy, make_state):
    # Regression: no turn number / deck list, but Golds in play -> economy exists, Province beats Gold.
    s = make_state(phase="buy", hand=["Estate"], in_play=["Gold", "Gold", "Silver"], coins=8)
    assert build_context(s, policy.engine).stage == "mid"
    assert policy.recommend(s).best.action == Buy("Province")

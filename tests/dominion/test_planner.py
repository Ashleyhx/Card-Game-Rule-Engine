"""Lookahead planner: hand-verifiable expectations, probabilities, budgets, missing information, API shape."""

import json
from importlib import resources

import pytest

from card_game_engine.games.dominion.planning import (
    PlanConfig, compare_actions, evaluate_action_sequences, get_objective, plan_next_action,
)
from card_game_engine.games.dominion.policy import HeuristicPolicy
from card_game_engine.games.dominion.rules import (
    Buy, EndTurn, PlayAction, action_from_dict, action_to_dict, state_from_dict,
)
from tests.dominion.conftest import FIRST_GAME

EXAMPLES = "card_game_engine.games.dominion.examples"


def snap(phase="action", planning=None, **player):
    data = {"schema_version": 1, "num_players": 2, "kingdom": FIRST_GAME,
            "player": {"phase": phase, "hand": [], **player}}
    if planning:
        data["planning"] = planning
    return state_from_dict(data)


def by_label(candidates):
    return {c.action["label"]: c for c in candidates}


def all_candidates(result):
    return [result.recommended, *result.alternatives]


def village_smithy():
    # Hand Village, Smithy, Copper; draw pile = {Copper 2, Silver 1, Estate 3} (6 cards, $4 in total); discard empty.
    return snap(hand=["Village", "Smithy", "Copper"],
                draw_pile={"composition": {"Copper": 2, "Silver": 1, "Estate": 3}},
                discard_pile={"composition": {}})


def test_village_before_smithy_has_higher_expected_money(engine):
    # Village first: draw 1, then Smithy draws 3 -> 4 random cards of 6, each worth $4/6 on average:
    #   E[$] = 1 (Copper in hand) + 4 * 4/6 = 11/3.
    # Smithy first: draw 3 (Village is stranded with 0 Actions): E[$] = 1 + 3 * 4/6 = 3.
    # No Action: $1.
    result = plan_next_action(village_smithy(), engine=engine)
    cands = by_label(all_candidates(result))
    assert result.recommended.action == action_to_dict(PlayAction("Village"))
    assert cands["Play Village"].expected_value == pytest.approx(11 / 3)
    assert cands["Play Smithy"].expected_value == pytest.approx(3.0)
    assert cands["End Action phase (go to Buy phase)"].expected_value == pytest.approx(1.0)
    assert result.method == "exact" and cands["Play Village"].value_provenance == "expected"
    assert cands["Play Village"].principal_line[:3] == ["Play Village", "draw Estate", "Play Smithy"]
    assert any("11/3" in s or "3.67" in s for s in result.explanation)


def test_lookahead_beats_greedy_heuristic_on_stranded_draw(engine):
    # Draw pile is only Estates: Smithy draws 3 Estates and strands Militia -> $3 (Coppers only).
    # Militia: +$2 and the three Coppers -> $5.
    state = snap(hand=["Smithy", "Militia", "Copper", "Copper", "Copper"], actions=1,
                 draw_pile={"composition": {"Estate": 3}}, discard_pile={"composition": {}})
    result = plan_next_action(state, engine=engine)
    cands = by_label(all_candidates(result))
    assert result.recommended.action == action_to_dict(PlayAction("Militia"))
    assert cands["Play Militia"].expected_value == pytest.approx(5.0)
    assert cands["Play Smithy"].expected_value == pytest.approx(3.0)
    assert cands["Play Smithy"].outcome_distribution[0].metrics["action_cards_stranded"] == 1
    # The V1 greedy heuristic prefers Smithy here: this case documents where lookahead helps.
    assert HeuristicPolicy(engine).recommend(state).best.action == PlayAction("Smithy")


def smithy_three_coppers():
    # Hand Smithy + 3 Coppers; draw pile {Silver 2, Copper 2, Estate 3} (7 cards); Smithy draws 3 of them.
    return snap(hand=["Smithy", "Copper", "Copper", "Copper"],
                draw_pile={"composition": {"Silver": 2, "Copper": 2, "Estate": 3}},
                discard_pile={"composition": {}})


def test_threshold_objective_is_exact_hypergeometric_probability(engine):
    # $6 needs >= $3 from 3 cards drawn out of C(7,3) = 35 equally likely sets (s Silvers, c Coppers):
    #   s2c1: C(2,2)C(2,1)=2   s2e1: 1*3=3   s1c2: 2*1=2   s1c1e1: 2*2*3=12   -> 19/35.
    state = smithy_three_coppers()
    result = plan_next_action(state, PlanConfig(objective="threshold:6"), engine)
    assert result.objective == "threshold:6" and result.value_name == "P($>=6)"
    assert result.recommended.action_key == "Play Smithy"
    assert result.recommended.expected_value == pytest.approx(19 / 35, abs=1e-12)
    assert result.recommended.threshold_probabilities[6] == pytest.approx(19 / 35, abs=1e-12)
    # Same probability reported by the money objective's distribution; E[$] = 3 + 3 * (2*2 + 2*1)/7 = 39/7.
    money = plan_next_action(state, engine=engine).recommended
    assert money.expected_value == pytest.approx(39 / 7)
    assert money.threshold_probabilities[6] == pytest.approx(19 / 35, abs=1e-12)
    assert money.threshold_probabilities[8] == pytest.approx(2 / 35, abs=1e-12)  # s2c1 only: $3 + $5


def test_outcome_distributions_are_normalised(engine):
    full = plan_next_action(smithy_three_coppers(), PlanConfig(top_outcomes=50), engine)
    for cand in all_candidates(full):
        assert cand.distribution_method == "exact"
        assert sum(o.probability for o in cand.outcome_distribution) == pytest.approx(1.0, abs=1e-9)
        assert cand.other_probability == pytest.approx(0.0, abs=1e-9)
    top1 = plan_next_action(smithy_three_coppers(), PlanConfig(top_outcomes=1), engine).recommended
    assert len(top1.outcome_distribution) == 1
    assert top1.outcome_distribution[0].probability + top1.other_probability == pytest.approx(1.0, abs=1e-9)


def test_same_seed_is_reproducible_also_when_sampling(engine):
    state = village_smithy()
    a = plan_next_action(state, PlanConfig(seed=3), engine).to_dict()
    b = plan_next_action(state, PlanConfig(seed=3), engine).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    forced = PlanConfig(max_outcomes_per_chance=1, samples_per_chance=4, seed=11)
    r1 = plan_next_action(state, forced, engine)
    r2 = plan_next_action(state, forced, engine)
    assert json.dumps(r1.to_dict(), sort_keys=True) == json.dumps(r2.to_dict(), sort_keys=True)
    smithy = by_label(all_candidates(r1))["Play Smithy"]
    assert smithy.method in ("monte_carlo", "mixed") and smithy.value_provenance == "monte_carlo"
    assert smithy.stderr is not None
    assert r1.method in ("monte_carlo", "mixed")
    assert any("sampled" in s for s in r1.limitations)
    if smithy.distribution_method is not None:
        assert smithy.distribution_method == "monte_carlo"


def test_node_budget_and_depth_limit(engine):
    state = village_smithy()
    legal = engine.legal_actions(state)
    tiny = plan_next_action(state, PlanConfig(max_nodes=5), engine)
    assert tiny.stats["budget_exhausted"] is True
    assert action_from_dict(tiny.recommended.action) in legal
    assert any("budget" in s for s in tiny.limitations)

    shallow = plan_next_action(state, PlanConfig(max_depth=2), engine)
    assert shallow.stats["max_depth_reached"] <= 2
    assert all(c.depth_reached <= 2 for c in all_candidates(shallow))
    assert action_from_dict(shallow.recommended.action) in legal
    village = by_label(all_candidates(shallow))["Play Village"]
    assert village.flags["p_stopped_mid_turn"] == pytest.approx(1.0)
    assert any("mid-turn" in s for s in shallow.limitations)


def test_unknown_deck_contents_are_flagged_not_invented(engine):
    # deck_count only: 2 cards of unknown identity, discard unknown. Smithy draws both "?" cards and then
    # would have to reshuffle an unknown discard pile -> boundary.
    state = snap(hand=["Smithy", "Copper"], deck_count=2)
    result = plan_next_action(state, engine=engine)
    assert result.information_level == "partially_known"
    smithy = by_label(all_candidates(result))["Play Smithy"]
    assert smithy.boundaries == {"unknown_draw_pile": pytest.approx(1.0)}
    assert smithy.flags["p_unknown_cards_in_hand"] == pytest.approx(1.0)
    leaf = smithy.outcome_distribution[0].metrics
    assert leaf["unknown_cards_in_hand"] == 2 and leaf["coins_generated"] == 1  # "?" counts $0
    assert any("unknown_draw_pile" in s for s in result.limitations)
    assert any("deck_count" in s for s in result.information_notes)


def test_insufficient_information_draws_hit_boundaries(engine):
    state = snap(hand=["Smithy", "Copper"])  # no deck size at all
    result = plan_next_action(state, engine=engine)
    assert result.information_level == "insufficient"
    smithy = by_label(all_candidates(result))["Play Smithy"]
    assert smithy.boundaries == {"unknown_draw_pile": pytest.approx(1.0)}
    leaf = smithy.outcome_distribution[0].metrics
    assert leaf["cards_drawn_this_turn"] == 0 and leaf["unknown_cards_in_hand"] == 0  # nothing fabricated
    assert any("insufficient" in s for s in result.limitations)
    assert any("unknown_draw_pile" in s and "1.00" in s for s in result.limitations)


def test_next_hand_boundary_does_not_claim_a_lower_bound_for_this_turns_money(engine):
    state = snap(phase="buy", hand=["Copper"])  # draw pile unknown: only the next hand hits the boundary
    result = plan_next_action(state, PlanConfig(horizon="next_hand"), engine=engine)
    end = by_label(all_candidates(result))["Play all Treasures"]
    assert end.boundaries == {"unknown_draw_pile": pytest.approx(1.0)} and end.value_provenance != "lower_bound"
    notes = [s for s in result.limitations if s.startswith("Play all Treasures: hidden information")]
    assert notes and all("lower bound" not in s for s in notes)


def test_blocked_state_gives_no_recommendation(engine):
    state = snap(hand=["Copper"], pending_draws=2, deck_count=5)
    result = plan_next_action(state, engine=engine)
    assert result.recommended is None and result.blocked
    assert "blocked" in result.explanation[0]
    json.dumps(result.to_dict())


def load_examples():
    files = resources.files(EXAMPLES)
    return [(f.name, state_from_dict(json.loads(f.read_text())))
            for f in sorted(files.iterdir(), key=lambda f: f.name) if f.name.endswith(".json")]


@pytest.mark.parametrize("name,state", load_examples(), ids=lambda v: v if isinstance(v, str) else "")
def test_examples_recommend_legal_actions_and_serialise(engine, name, state):
    result = plan_next_action(state, PlanConfig(max_nodes=300, max_depth=4), engine)
    legal = engine.legal_actions(state)
    if result.recommended is None:
        assert not legal or result.blocked
    else:
        assert action_from_dict(result.recommended.action) in legal
    json.dumps(result.to_dict())


def test_multi_buy_money_vs_purchase(engine):
    state = snap(phase="buy", coins=8, buys=2)
    money = plan_next_action(state, engine=engine)
    # Money counts coins generated ($8 whatever is bought); its documented tie-break spends the most coins
    # and then keeps Buys, which picks a single Province.
    assert money.objective == "money"
    assert money.recommended.action == action_to_dict(Buy("Province"))
    assert money.recommended.expected_value == pytest.approx(8.0)
    purchase = plan_next_action(state, PlanConfig(objective="purchase"), engine)
    assert purchase.objective == "purchase" and purchase.recommended.value_provenance == "heuristic"
    assert action_from_dict(purchase.recommended.action) in engine.legal_actions(state)
    assert any("heuristic" in s for s in purchase.limitations)
    json.dumps(purchase.to_dict())


def test_next_hand_money_reshuffle_changes_value_only_after_reshuffle(engine):
    # Buy phase, $6 in play (Silver, Silver, Copper, Copper); draw pile = 2 Estates; discard = 1 Copper.
    # Clean-up discards the 4 played Treasures -> discard {Copper 3, Silver 2} (+ Gold if bought).
    # Next hand = 2 Estates + 3 cards from the reshuffled discard:
    #   no buy:   3 * (3*1 + 2*2) / 5        = 21/5 = 4.2
    #   Buy Gold: 3 * (3*1 + 2*2 + 3) / 6    = 5.0
    reshuffle = snap(phase="buy", in_play=["Silver", "Silver", "Copper", "Copper"], coins=6,
                     draw_pile={"composition": {"Estate": 2}}, discard_pile={"composition": {"Copper": 1}})
    cfg = PlanConfig(objective="next_hand_money")
    res = compare_actions(reshuffle, [Buy("Gold"), EndTurn()], cfg, engine)
    assert res.horizon == "next_hand"
    vals = {c.action_key: c.expected_value for c in res.candidates}
    assert vals["Buy Gold"] == pytest.approx(5.0)
    assert vals["End turn (Clean-up)"] == pytest.approx(21 / 5)
    assert all(c.distribution_method == "exact" for c in res.candidates)

    # No reshuffle: draw pile holds exactly 5 known cards -> the next hand is {Copper 2, Estate 3} either way.
    no_reshuffle = snap(phase="buy", in_play=["Silver", "Silver", "Copper", "Copper"], coins=6,
                        draw_pile={"composition": {"Copper": 2, "Estate": 3}}, discard_pile={"composition": {}})
    res = compare_actions(no_reshuffle, [Buy("Gold"), EndTurn()], cfg, engine)
    assert [c.expected_value for c in res.candidates] == [pytest.approx(2.0), pytest.approx(2.0)]


def test_snapshot_planning_horizon_is_the_default(engine):
    state = snap(phase="buy", coins=3, draw_pile={"composition": {"Copper": 5}}, discard_pile={"composition": {}},
                 planning={"horizon": "next_hand"})
    result = plan_next_action(state, engine=engine)
    assert result.horizon == "next_hand"
    assert result.recommended.outcome_distribution[0].metrics["next_hand_coins"] == 5
    assert plan_next_action(state, PlanConfig(horizon="turn"), engine).horizon == "turn"
    with pytest.raises(ValueError, match="next_hand"):
        plan_next_action(state, PlanConfig(objective="next_hand_money", horizon="turn"), engine)


def test_compare_actions_dicts_objects_and_illegal(engine):
    state = village_smithy()
    res = compare_actions(state, [action_to_dict(PlayAction("Smithy")), PlayAction("Village"), Buy("Gold")],
                          engine=engine)
    assert [c.action_key for c in res.candidates] == ["Play Village", "Play Smithy", "Buy Gold"]
    village, smithy, gold = res.candidates
    assert village.expected_value == pytest.approx(11 / 3) and smithy.expected_value == pytest.approx(3.0)
    assert gold.legal is False and gold.expected_value is None
    assert gold.illegal_reasons == engine.check(state, Buy("Gold"))
    assert any("difference +0.67" in n for n in res.difference_notes)
    assert any("Buy Gold" in n and "illegal" in n for n in res.difference_notes)
    json.dumps(res.to_dict())


def test_evaluate_action_sequences_covers_every_legal_first_action(engine):
    state = snap(phase="buy", hand=["Copper", "Silver"])
    cands = evaluate_action_sequences(state, engine=engine)
    assert sorted(c.action_key for c in cands) == sorted(a.describe() for a in engine.legal_actions(state))
    vals = {c.action_key: c.expected_value for c in cands}
    # Playing one Treasure first loses nothing (the other is played next); buying first forfeits the Treasures.
    assert vals["Play all Treasures"] == vals["Play Copper"] == vals["Play Silver"] == pytest.approx(3.0)
    assert vals["Buy Copper"] == pytest.approx(0.0)
    assert cands[0].expected_value == pytest.approx(3.0)


def test_config_and_objective_validation():
    cfg = PlanConfig(objective="threshold:8", horizon="turn", seed=4)
    assert PlanConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown PlanConfig"):
        PlanConfig.from_dict({"depth": 3})
    with pytest.raises(ValueError, match="money, threshold:<T>, purchase, next_hand_money"):
        PlanConfig(objective="greed")
    with pytest.raises(ValueError):
        PlanConfig(horizon="game")
    with pytest.raises(ValueError):
        get_objective("threshold:x")
    assert get_objective("threshold:6").name == "threshold:6"


def load_example(name):
    return state_from_dict(json.loads(resources.files(EXAMPLES).joinpath(f"{name}.json").read_text()))


def test_unknown_draw_pile_values_are_lower_bounds_and_comparison_not_conclusive(engine):
    # unknown_deck_minimal: nothing is known about the deck. Market draws a card of unknown identity -> boundary;
    # its $4 (Market $1 + Copper + Silver) counts the owed card as $0, so it is a lower bound. Militia: exactly $5.
    result = plan_next_action(load_example("unknown_deck_minimal"), engine=engine)
    cands = by_label(all_candidates(result))
    militia, market = cands["Play Militia"], cands["Play Market"]
    assert result.recommended is militia
    assert militia.value_provenance == "exact" and militia.threshold_provenance == "exact"
    assert market.expected_value == pytest.approx(4.0)
    assert market.value_provenance == "lower_bound" and market.threshold_provenance == "lower_bound"
    assert market.flags["p_value_lower_bound"] == pytest.approx(1.0)
    assert any("P($>=5) >= 0.00" in s for s in market.explanation)
    assert not any("(exact)" in s for s in market.explanation)
    assert market.outcome_distribution[0].label.startswith("coins=$4+")
    assert any("Not conclusive" in s and "Play Market" in s and "lower bound" in s for s in result.explanation)
    assert any("E[$] >= 4.00" in s or ">= 4.00" in s for s in result.explanation)
    d = market.to_dict()
    assert d["value_provenance"] == "lower_bound" and d["threshold_provenance"] == "lower_bound"
    json.dumps(result.to_dict())

    res = compare_actions(load_example("unknown_deck_minimal"), [PlayAction("Militia"), PlayAction("Market")],
                          engine=engine)
    assert any("Not conclusive" in n for n in res.difference_notes)


def test_unknown_cards_in_hand_make_a_lower_bound_without_boundary(engine):
    # Draw pile = 2 Copper + 1 unknown card; Smithy draws all three: $1 + $2 + "?" (counted $0) -> E[$] >= 3.
    state = snap(hand=["Smithy", "Copper"], draw_pile={"composition": {"Copper": 2}, "unknown_count": 1},
                 discard_pile={"composition": {}})
    result = plan_next_action(state, engine=engine)
    smithy = by_label(all_candidates(result))["Play Smithy"]
    assert not smithy.boundaries
    assert smithy.expected_value == pytest.approx(3.0) and smithy.value_provenance == "lower_bound"
    assert smithy.threshold_provenance == "lower_bound"
    # The recommended lower bound (>= 3) beats an exact $1: that comparison is conclusive, so no caveat.
    assert result.recommended is smithy
    assert not any("Not conclusive" in s for s in result.explanation)


def test_manual_choice_boundary_is_partial(engine):
    # Workshop's gain is a player choice the planner does not model -> boundary manual:Workshop.
    result = plan_next_action(snap(hand=["Workshop", "Copper"]), engine=engine)
    workshop = by_label(all_candidates(result))["Play Workshop"]
    assert workshop.boundaries == {"manual:Workshop": pytest.approx(1.0)}
    assert workshop.value_provenance == "partial" and workshop.threshold_provenance == "partial"
    assert any("as if that choice did nothing" in s for s in workshop.explanation)
    assert any("Not conclusive" in s and "partial" in s for s in result.explanation)
    assert any("manual:Workshop" in s and "partial" in s for s in result.limitations)


def test_money_objective_says_purchase_is_not_judged(engine):
    from card_game_engine.games.dominion.planning.planner import MONEY_BUY_NOTE

    state = snap(phase="buy", coins=8, buys=2)
    money = plan_next_action(state, engine=engine)
    assert any(step.startswith("Buy ") for step in money.recommended.principal_line)
    assert MONEY_BUY_NOTE in money.limitations
    assert "--objective purchase" in MONEY_BUY_NOTE and "cardgame dominion recommend" in MONEY_BUY_NOTE
    assert MONEY_BUY_NOTE not in plan_next_action(state, PlanConfig(objective="purchase"), engine).limitations
    no_buy = plan_next_action(snap(phase="buy", coins=3, buys=0), engine=engine)  # no Buy can appear in any line
    assert MONEY_BUY_NOTE not in no_buy.limitations

"""Lookahead planner: hand-verifiable expectations, probabilities, budgets, missing information, API shape."""

import json
from importlib import resources

import pytest

from fractions import Fraction

from card_game_engine.cli import main as cardgame_main
from card_game_engine.games.dominion.planning import (
    PlanConfig, card_totals, compare_actions, draw_count_distribution, evaluate_action_sequences, from_game_state,
    get_objective, parse_target, plan_next_action, projected_hand_count, simulate_projected_cleanup,
)
from card_game_engine.games.dominion.policy import HeuristicPolicy
from card_game_engine.games.dominion.rules import (
    Buy, DiscardKnowledge, DrawPileKnowledge, EndTurn, PlayAction, PlayAllTreasures, PlayTreasure, SnapshotError,
    action_from_dict, action_to_dict, load_registry, state_from_dict,
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
    # Single Play-Treasure actions are collapsed into Play-all-Treasures (as plan_next_action does); Buys are kept.
    legal = [a.describe() for a in engine.legal_actions(state)]
    assert "Play Copper" in legal and "Play Silver" in legal
    assert sorted(c.action_key for c in cands) == sorted(k for k in legal if k not in ("Play Copper", "Play Silver"))
    vals = {c.action_key: c.expected_value for c in cands}
    # Buying first forfeits the Treasures.
    assert vals["Play all Treasures"] == pytest.approx(3.0)
    assert vals["Buy Copper"] == pytest.approx(0.0)
    assert cands[0].expected_value == pytest.approx(3.0)
    # compare_actions without candidates is the same comparison; explicit candidates may name a single Treasure.
    assert [c.action_key for c in compare_actions(state, engine=engine).candidates] == [c.action_key for c in cands]
    res = compare_actions(state, [PlayTreasure("Copper"), PlayAllTreasures()], engine=engine)
    vals = {c.action_key: c.expected_value for c in res.candidates}
    # Playing one Treasure first loses nothing (the other is played next).
    assert vals == {"Play Copper": pytest.approx(3.0), "Play all Treasures": pytest.approx(3.0)}


def test_config_and_objective_validation():
    cfg = PlanConfig(objective="threshold:8", horizon="turn", seed=4)
    assert PlanConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown PlanConfig"):
        PlanConfig.from_dict({"depth": 3})
    with pytest.raises(ValueError, match="money, threshold:<T>, purchase, next_hand_money, future_money"):
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


def test_target_objective_notes_purchase_and_first_hand_only_ties(engine):
    from card_game_engine.games.dominion.planning.planner import TARGET_BUY_NOTE, TARGET_TIE_HINT

    state = snap(phase="buy", coins=3, buys=1, hand=["Estate"], draw_pile={"composition": {"Copper": 5, "Estate": 3}},
                 discard_pile={"composition": {}})
    result = plan_next_action(state, PlanConfig(objective="target:Silver>=1", horizon="hands:2"), engine)
    assert TARGET_BUY_NOTE in result.limitations and result.tie
    tie = next(s for s in result.explanation if s.startswith("TIE"))
    assert TARGET_TIE_HINT in tie and "longer horizon)" not in tie
    silver = by_label(all_candidates(result))["Buy Silver"]
    assert silver.projected_hands[0]["p_target"] == 0 < silver.projected_hands[1]["p_target"]  # only hand 2 differs


# ---------------------------------------------------------------- horizon hands:N, future_money, ties

def turn_one(**overrides):
    # Turn 1: owned 7 Copper + 3 Estate, discard empty -> draw pile = owned - hand - play.
    player = dict(owned_cards={"Copper": 7, "Estate": 3}, discard_count=0)
    player.update(overrides)
    return snap(phase="buy", **player)


def turn_one_treasures_played():
    # Hand Estate, 4 Coppers in play ($4) -> draw pile {Copper 3, Estate 2}.
    return turn_one(hand=["Estate"], in_play=["Copper"] * 4, coins=4)


def test_hands_2_future_money_turn_one_buy_silver_by_hand(engine):
    # Clean-up: hand + play (+ bought card) -> discard {Copper 4, Estate 1 (+ Silver)}.
    # Hand 1 = the whole 5-card draw pile {Copper 3, Estate 2}: $3 for certain, no reshuffle.
    # Hand 1's Clean-up -> discard {Copper 7, Estate 3 (+ Silver)}; the draw pile is empty -> reshuffle, draw 5.
    #   Buy Silver: 11 cards, E[$] = 5 * (7*1 + 1*2) / 11 = 45/11;
    #               P($>=5) = [C(7,3)C(3,1) + C(7,4)  (Silver + >=3 Coppers)  +  C(7,5)  (5 Coppers)] / C(11,5)
    #                       = (105 + 35 + 21) / 462 = 23/66.
    #   Nothing:    10 cards, E[$] = 5 * 7/10 = 7/2;  P($>=5) = C(7,5) / C(10,5) = 21/252 = 1/12.
    # future_money = hand 1 + hand 2: 3 + 45/11 = 78/11 vs 3 + 7/2 = 13/2.
    cfg = PlanConfig(objective="future_money", horizon="hands:2", top_outcomes=50)
    res = compare_actions(turn_one_treasures_played(), [EndTurn(), Buy("Silver")], cfg, engine)
    assert res.horizon == "hands:2" and res.objective == "future_money" and not res.tie
    silver, nothing = res.candidates
    assert silver.action_key == "Buy Silver" and nothing.action_key == "End turn (Clean-up)"
    assert silver.expected_value == pytest.approx(78 / 11) and nothing.expected_value == pytest.approx(13 / 2)
    assert silver.expected_value > nothing.expected_value
    assert silver.value_provenance == "expected" and silver.method == "exact"
    for c in (silver, nothing):
        h1, h2 = c.projected_hands
        assert h1["hand"] == 1 and h1["expected_coins_exact"] == "3" and h1["p_at_least_5_exact"] == "0"
        assert h1["provenance"] == "exact" and h1["p_lower_bound"] == 0
        # hand 1 is deterministic: every leaf of the (fully shown) distribution has the same $3 first hand
        assert c.other_probability == pytest.approx(0.0)
        assert {o.metrics["hand_coins"][0] for o in c.outcome_distribution} == {3}
    assert Fraction(silver.projected_hands[1]["expected_coins_exact"]) == Fraction(45, 11)
    assert Fraction(silver.projected_hands[1]["p_at_least_5_exact"]) == Fraction(23, 66)
    assert Fraction(nothing.projected_hands[1]["expected_coins_exact"]) == Fraction(7, 2)
    assert Fraction(nothing.projected_hands[1]["p_at_least_5_exact"]) == Fraction(1, 12)
    assert silver.projected_hands[1]["expected_coins"] == pytest.approx(45 / 11)
    assert any("hand 2 E[$]=4.09, P($>=5)=0.35" in s for s in silver.explanation)
    assert any("(after reshuffle)" in step and step.startswith("hand 2:") for step in silver.principal_line)
    d = silver.to_dict()
    assert d["projected_hands"][1]["p_at_least_5"] == pytest.approx(23 / 66)
    json.dumps(res.to_dict())
    # The planner itself buys Silver with $4 under future_money (Silver is the best $ per card it can afford).
    plan = plan_next_action(turn_one_treasures_played(), cfg, engine)
    assert plan.recommended.action_key == "Buy Silver" and not plan.tie


def test_hands_2_is_consistent_with_hands_1(engine):
    # reshuffle-style state: draw pile 2 Estates, discard 1 Copper, $6 in play (see the next_hand_money test above).
    state = snap(phase="buy", in_play=["Silver", "Silver", "Copper", "Copper"], coins=6,
                 draw_pile={"composition": {"Estate": 2}}, discard_pile={"composition": {"Copper": 1}})
    actions = [Buy("Gold"), Buy("Silver"), EndTurn()]
    one = compare_actions(state, actions, PlanConfig(objective="future_money", horizon="hands:1"), engine)
    alias = compare_actions(state, actions, PlanConfig(objective="future_money", horizon="next_hand"), engine)
    nh = compare_actions(state, actions, PlanConfig(objective="next_hand_money", horizon="hands:2"), engine)
    two = compare_actions(state, actions, PlanConfig(objective="future_money", horizon="hands:2"), engine)
    v = lambda res: {c.action_key: c.expected_value for c in res.candidates}  # noqa: E731
    hands = lambda res: {c.action_key: c.projected_hands for c in res.candidates}  # noqa: E731
    assert v(one) == v(alias) == v(nh)                      # first projected hand, whichever way it is asked
    assert v(one)["Buy Gold"] == pytest.approx(5.0) and v(one)["End turn (Clean-up)"] == pytest.approx(21 / 5)
    assert hands(one) == hands(alias)
    for key in v(one):
        h1 = hands(one)[key][0]
        assert hands(two)[key][0] == h1 and hands(nh)[key][0] == h1   # hand 1 does not depend on the horizon
        assert len(hands(two)[key]) == 2
        assert v(two)[key] == pytest.approx(sum(h["expected_coins"] for h in hands(two)[key]))
        assert v(two)[key] > v(one)[key]


def test_projected_cleanup_conserves_cards_and_reshuffles_only_when_required(engine):
    from card_game_engine.games.dominion.planning import DominionPlanningModel, PlanNode, get_objective

    state = turn_one_treasures_played()
    root = PlanNode(from_game_state(state, engine))
    model = DominionPlanningModel(engine, get_objective("future_money"), "hands:3", root)
    end = model.apply(root, EndTurn())
    total = card_totals(end.ps)
    (p, first, label), = model.exact_outcomes(end, 200)        # hand 1: the whole draw pile, no reshuffle
    assert p == 1 and "reshuffle" not in label and first.ps.projected == (("Copper",) * 3 + ("Estate",) * 2,)
    assert first.ps.next_hand is None and first.ps.game.turn.pending_draws == 5 and not model.is_terminal(first)
    assert first.ps.discard.counter == {"Copper": 7, "Estate": 3} and first.ps.draw.size == 0
    assert card_totals(first.ps) == total
    outs = model.exact_outcomes(first, 200)
    assert sum(o[0] for o in outs) == 1 and all("(after reshuffle)" in o[2] for o in outs)
    for _, child, _ in outs:                                   # hand 2 drawn and cleaned up; hand 3 pending
        assert len(child.ps.projected) == 2 and card_totals(child.ps) == total
        # hand 3 comes from the 5 cards left after hand 2: no further reshuffle needed
        assert child.ps.draw.size == 5 and child.ps.discard.size == 5
    with pytest.raises(ValueError):
        simulate_projected_cleanup(end.ps, engine)              # hand not drawn yet


def test_future_money_sampling_fallback_is_reproducible(engine):
    cfg = PlanConfig(objective="future_money", horizon="hands:2", max_outcomes_per_chance=1, samples_per_chance=8,
                     seed=11)
    state = turn_one_treasures_played()
    a = compare_actions(state, [Buy("Silver"), EndTurn()], cfg, engine)
    b = compare_actions(state, [Buy("Silver"), EndTurn()], cfg, engine)
    assert a.to_dict() == b.to_dict()
    for c in a.candidates:
        # hand 1 (the whole 5-card pile) has a single outcome and stays exact; hand 2 is sampled -> mixed
        assert c.method == "mixed" and c.value_provenance == "monte_carlo" and c.stderr is not None
        assert c.distribution_method == "monte_carlo"
        assert [h["provenance"] for h in c.projected_hands] == ["monte_carlo", "monte_carlo"]
        assert c.projected_hands[0]["expected_coins"] == pytest.approx(3.0)  # hand 1 is $3 in every sample
        assert c.projected_hands[1]["expected_coins_exact"] is None
    assert a.stats["sampled_chance_nodes"] > 0
    assert any("sampled" in s and "seed 11" in s for s in a.limitations)
    other = compare_actions(state, [Buy("Silver"), EndTurn()], PlanConfig(**{**cfg.to_dict(), "seed": 12}), engine)
    assert other.to_dict() != a.to_dict()


def test_future_money_unknown_cards_are_lower_bounds(engine):
    # Draw pile = 5 Coppers, discard size known but contents unknown (3 cards): hand 1 is exactly $5,
    # hand 2 reshuffles the discard ({4 played Coppers, ...}) together with 3 unknown cards -> lower bound.
    state = snap(phase="buy", in_play=["Copper"] * 4, coins=4, draw_pile={"composition": {"Copper": 5}},
                 discard_count=3)
    res = compare_actions(state, [EndTurn()], PlanConfig(objective="future_money", horizon="hands:2"), engine)
    c = res.candidates[0]
    h1, h2 = c.projected_hands
    assert h1["provenance"] == "exact" and h1["expected_coins"] == pytest.approx(5.0)
    assert h2["provenance"] == "lower_bound" and h2["p_lower_bound"] > 0
    assert c.value_provenance == "lower_bound"
    assert any("hand 2 E[$] >= " in s for s in c.explanation)


def test_horizon_and_future_objective_validation(engine):
    assert [projected_hand_count(h) for h in ("turn", "next_hand", "hands:1", "hands:4")] == [0, 1, 1, 4]
    for bad in ("hands:0", "hands:5", "hands:x", "hands", "game"):
        with pytest.raises(ValueError, match="hands:<N>"):
            PlanConfig(horizon=bad)
    cfg = PlanConfig(objective="future_money", horizon="hands:3")
    assert PlanConfig.from_dict(cfg.to_dict()) == cfg
    state = turn_one(hand=["Copper"] * 4 + ["Estate"])
    with pytest.raises(ValueError, match="future_money.*next_hand"):
        plan_next_action(state, PlanConfig(objective="future_money", horizon="turn"), engine)
    # default horizon of the future objectives is next_hand (= hands:1)
    assert plan_next_action(state, PlanConfig(objective="future_money"), engine).horizon == "next_hand"


def test_tie_is_reported_not_presented_as_better(engine):
    # Turn 1 Buy phase, hand 4 Copper + Estate, draw pile {Copper 3, Estate 2}: the next hand is that whole pile
    # whatever is done now -> every candidate has E[next-hand $] = 3: a tie.
    state = turn_one(hand=["Copper"] * 4 + ["Estate"])
    result = plan_next_action(state, PlanConfig(objective="next_hand_money", horizon="next_hand"), engine)
    assert result.tie is True
    assert all(c.expected_value == pytest.approx(3.0) for c in all_candidates(result))
    assert sorted(result.tied_actions) == sorted(c.action["label"] for c in all_candidates(result))
    assert result.recommended is not None                      # kept for API stability, flagged by ``tie``
    assert result.explanation[0].startswith("TIE: no difference under objective next_hand_money")
    assert "beyond this objective/horizon" in result.explanation[0]
    assert not any(s.startswith("Recommended ") for s in result.explanation)
    d = result.to_dict()
    assert d["tie"] is True and d["tied_actions"] == result.tied_actions
    json.dumps(d)
    res = compare_actions(state, None, PlanConfig(objective="next_hand_money"), engine)
    assert res.tie and set(res.tied_actions) == {c.action_key for c in res.candidates if c.legal}
    assert any("no difference under objective next_hand_money" in n for n in res.difference_notes)
    assert res.to_dict()["tie"] is True
    # No tie: a real difference is still reported as one.
    better = plan_next_action(village_smithy(), engine=engine)
    assert better.tie is False and better.tied_actions == []
    assert better.explanation[0].startswith("Recommended Play Village")


def test_equal_lower_bounds_are_not_reported_as_a_tie(engine):
    # Unknown draw pile: Play Village and Play Smithy both show E[$] >= 2 (drawn cards count $0). The true
    # values may differ, so this is "not conclusive", never "no difference under the objective".
    state = snap(hand=["Village", "Smithy", "Copper", "Copper"], deck_count=6, discard_count=4)
    result = plan_next_action(state, PlanConfig(objective="money"), engine)
    vals = by_label(all_candidates(result))
    assert vals["Play Village"].expected_value == vals["Play Smithy"].expected_value
    assert vals["Play Village"].value_provenance == "lower_bound"
    assert result.tie is False and result.tied_actions == []
    assert not any(s.startswith("TIE") for s in result.explanation)
    assert any(s.startswith("Not conclusive") for s in result.explanation)
    res = compare_actions(state, None, PlanConfig(objective="money"), engine)
    assert res.tie is False and any(n.startswith("Not conclusive") for n in res.difference_notes)


def test_budget_cut_projection_is_a_lower_bound(engine):
    # A tiny node budget cannot project 3 hands: hands that were not reached count $0 -> labelled lower bound.
    cfg = PlanConfig(objective="future_money", horizon="hands:3", max_nodes=6)
    res = compare_actions(turn_one_treasures_played(), [Buy("Silver"), EndTurn()], cfg, engine)
    assert all(c.truncated and c.value_provenance == "lower_bound" for c in res.candidates)
    assert any("node budget cut off" in s for c in res.candidates for s in c.explanation)
    assert any("Not conclusive" in n for n in res.difference_notes) or res.tie


@pytest.mark.parametrize("horizon", ["turn", "next_hand", "hands:1", "hands:4"])
def test_snapshot_horizon_matches_plan_config(horizon):
    snap = {"num_players": 2, "kingdom": FIRST_GAME, "player": {"hand": []}, "planning": {"horizon": horizon}}
    assert state_from_dict(snap).planning_horizon == horizon
    assert PlanConfig(horizon=horizon).horizon == horizon


@pytest.mark.parametrize("horizon", ["hands:0", "hands:5", "hands:x", "week"])
def test_snapshot_horizon_rejects_what_plan_config_rejects(horizon):
    snap = {"num_players": 2, "kingdom": FIRST_GAME, "player": {"hand": []}, "planning": {"horizon": horizon}}
    with pytest.raises(SnapshotError, match="planning.horizon"):
        state_from_dict(snap)
    with pytest.raises(ValueError):
        PlanConfig(horizon=horizon)


# ---------------------------------------------------------------- objective target:<T>>=<K>

def test_target_objective_exact_probability_differs_between_action_plays(engine):
    # village_smithy: hand Village, Smithy, Copper; draw pile {Copper 2, Silver 1, Estate 3} (3 Treasures, 3 Estates).
    # Treasure>=3 in hand + play = the Copper in hand + >= 2 Treasures among the cards drawn.
    #   Village, then Smithy: 4 of the 6 cards; at most 3 are Estates, so 0 Treasures is impossible;
    #     P(1 Treasure) = C(3,1)C(3,3)/C(6,4) = 3/15  ->  P = 12/15 = 4/5.
    #   Smithy first (Village stranded): 3 of 6:  P(>=2) = [C(3,2)C(3,1) + C(3,3)] / C(6,3) = 10/20 = 1/2.
    #   No Action: the Copper alone -> 0.
    state = village_smithy()
    result = plan_next_action(state, PlanConfig(objective="target:Treasure>=3"), engine)
    assert result.objective == "target:Treasure>=3" and result.value_name == "P(Treasure>=3)"
    assert action_from_dict(result.recommended.action) in engine.legal_actions(state)
    cands = by_label(all_candidates(result))
    assert result.recommended.action_key == "Play Village" and not result.tie
    assert cands["Play Village"].expected_value == pytest.approx(4 / 5, abs=1e-12)
    assert cands["Play Smithy"].expected_value == pytest.approx(1 / 2, abs=1e-12)
    assert cands["End Action phase (go to Buy phase)"].expected_value == 0
    assert cands["Play Village"].value_provenance == "expected" and result.method == "exact"
    # Outcome groups carry the target count, so the exact distribution adds up to the same probability.
    full = plan_next_action(state, PlanConfig(objective="target:Treasure>=3", top_outcomes=50), engine)
    village = by_label(all_candidates(full))["Play Village"]
    assert village.other_probability == pytest.approx(0.0)
    assert sum(o.probability for o in village.outcome_distribution
               if o.metrics["target_count"] >= 3) == pytest.approx(4 / 5, abs=1e-12)
    # target:<card> means K = 1: P(Silver among 4 of 6) = 4/6 vs 3/6.
    silver = compare_actions(state, [PlayAction("Smithy"), PlayAction("Village")],
                             PlanConfig(objective="target:silver"), engine)
    assert silver.objective == "target:Silver>=1"
    assert [(c.action_key, c.expected_value) for c in silver.candidates] == [
        ("Play Village", pytest.approx(2 / 3)), ("Play Smithy", pytest.approx(1 / 2))]
    json.dumps(result.to_dict())


def test_target_objective_unknown_cards_are_lower_bounds(engine):
    # Draw pile = 2 Copper + 1 unknown card; Smithy draws all three: 3 known Treasures + "?" (never a match).
    state = snap(hand=["Smithy", "Copper"], draw_pile={"composition": {"Copper": 2}, "unknown_count": 1},
                 discard_pile={"composition": {}})
    result = plan_next_action(state, PlanConfig(objective="target:Treasure>=4"), engine)
    smithy = by_label(all_candidates(result))["Play Smithy"]
    assert smithy.expected_value == 0 and smithy.value_provenance == "lower_bound"
    label = smithy.outcome_distribution[0].label
    assert label.startswith("coins=$3+") and "target cards 3+" in label
    # The lower-bound wording fits the objective: "?" cards are non-matching here, $0 for money objectives.
    bound = [s for s in smithy.explanation if s.startswith("This value is a lower bound")]
    assert bound and all("counted as non-matching" in s and "$0" not in s for s in bound)
    money = by_label(all_candidates(plan_next_action(state, PlanConfig(objective="money"), engine)))["Play Smithy"]
    assert any(s.endswith("which are counted as $0.") for s in money.explanation)


def test_target_objective_hands_2_per_hand_probability_matches_draw_analysis(engine):
    # turn_one_treasures_played: 4 Coppers in play, Estate in hand, draw pile {Copper 3, Estate 2}.
    # Hand 1 is that whole pile: 3 Treasures -> P(>=4 Treasures) = 0 for every candidate -> the utility (first
    # projected hand) is 0 for both: a tie, reported as before. Hand 2 = 5 cards from the reshuffled discard:
    #   Buy Silver: {Copper 7, Silver 1, Estate 3}: [C(8,4)C(3,1) + C(8,5)] / C(11,5) = (210 + 56)/462 = 19/33.
    #   Nothing:    {Copper 7, Estate 3}:           [C(7,4)C(3,1) + C(7,5)] / C(10,5) = (105 + 21)/252 = 1/2.
    cfg = PlanConfig(objective="target:Treasure>=4", horizon="hands:2")
    res = compare_actions(turn_one_treasures_played(), [Buy("Silver"), EndTurn()], cfg, engine)
    assert res.value_name == "P(next hand Treasure>=4)"
    assert res.tie and set(res.tied_actions) == {"Buy Silver", "End turn (Clean-up)"}
    assert all(c.expected_value == 0 and c.value_provenance == "expected" for c in res.candidates)
    silver, nothing = by_label(res.candidates)["Buy Silver"], by_label(res.candidates)["End turn (Clean-up)"]
    treasure = parse_target("Treasure", load_registry())
    for c, discard, expected in ((silver, {"Copper": 7, "Silver": 1, "Estate": 3}, Fraction(19, 33)),
                                 (nothing, {"Copper": 7, "Estate": 3}, Fraction(1, 2))):
        h1, h2 = c.projected_hands
        assert h1["target"] == "Treasure>=4" and h1["p_target_exact"] == "0" and h1["provenance"] == "exact"
        assert Fraction(h2["p_target_exact"]) == expected and h2["p_target"] == pytest.approx(float(expected))
        # Same number from the session analysis' draw engine on the piles after hand 1's Clean-up.
        counts = draw_count_distribution([(Fraction(1), DrawPileKnowledge(), DiscardKnowledge(discard))], 5, treasure)
        assert counts.method == "exact" and counts.p_reshuffle == 1 and counts.p_at_least(4) == expected
    assert any("P(Treasure>=4)=0.58" in s for s in silver.explanation)
    # Horizon turn: the 4 Coppers in play already make it certain whatever is done -> also a tie, at 1.
    turn = compare_actions(turn_one_treasures_played(), [Buy("Silver"), EndTurn()],
                           PlanConfig(objective="target:Treasure>=4"), engine)
    assert turn.tie and all(c.expected_value == 1 and c.value_provenance == "exact" for c in turn.candidates)
    assert all(not c.projected_hands for c in turn.candidates)


@pytest.mark.parametrize("bad,match", [
    ("target:", "Unknown target ''"), ("target:Tresure", "Did you mean: Treasure"),
    ("target:Treasure>=0", "K >= 1"), ("target:Treasure>=x", "K >= 1"), ("target:Treasure>=", "K >= 1"),
    ("target:>=2", "Unknown target"),
])
def test_target_objective_bad_strings(bad, match):
    with pytest.raises(ValueError, match=match):
        PlanConfig(objective=bad)
    with pytest.raises(ValueError, match="Bad target objective"):
        get_objective(bad)


def test_target_objective_parsing():
    assert get_objective("target:Gold").name == "target:Gold>=1"
    assert get_objective("TARGET:action >= 2").name == "target:Action>=2"
    assert get_objective("target:Curse").target.kind == "card"   # card names win over type names
    assert PlanConfig.from_dict(PlanConfig(objective="target:Treasure>=3").to_dict()).objective == "target:Treasure>=3"


def test_cli_plan_target_objective(capsys):
    # small_deck_village_smithy: 2 Coppers in hand; draw pile {Copper 2, Silver 1, Estate 2}.
    # Village + Smithy draw 4 of 5 -> >= 2 Treasures for certain -> 1; Smithy first: 3 of 5 ->
    # [C(3,2)C(2,1) + C(3,3)] / C(5,3) = 7/10.
    code = cardgame_main(["dominion", "plan", "-e", "small_deck_village_smithy", "--objective", "target:Treasure>=4"])
    out = capsys.readouterr().out
    assert code == 0 and ">> RECOMMENDED: Play Village    P(Treasure>=4) = 1.00" in out
    assert "Play Smithy" in out and "P(Treasure>=4) = 0.70" in out
    code = cardgame_main(["dominion", "plan", "-e", "small_deck_village_smithy", "--objective", "target:Treasure>=4",
                          "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and data["objective"] == "target:Treasure>=4" and data["value_name"] == "P(Treasure>=4)"
    assert data["recommended"]["action_key"] == "Play Village" and data["recommended"]["expected_value"] == 1.0
    assert data["alternatives"][0]["expected_value"] == pytest.approx(7 / 10)
    code = cardgame_main(["dominion", "plan", "-e", "opening_5_2", "--objective", "target:Tresure>=2"])
    assert code == 2 and "Did you mean: Treasure" in capsys.readouterr().err

"""Decision policy behaviour on representative situations."""

import ast
import json
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from importlib import resources
from itertools import combinations
from math import comb
from pathlib import Path

import pytest

from card_game_engine.games.dominion.policy import HeuristicPolicy, build_context
from card_game_engine.games.dominion.policy import odds as odds_module
from card_game_engine.games.dominion.policy.odds import ShuffleOddsModel
from card_game_engine.games.dominion.rules import Buy, EndActionPhase, PlayAction, PlayAllTreasures, DominionRuleEngine, state_from_dict
from tests.dominion.conftest import FIRST_GAME, OTHER_KINGDOM


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


# ---------------------------------------------------------------- next-shuffle odds and money density

STARTING_DECK = {"Copper": 7, "Estate": 3}


def buy_state(make_state, coins, owned=STARTING_DECK, **kw):
    return make_state(phase="buy", coins=coins, turn_number=3, owned_cards=dict(owned), **kw)


def ranked_buys(advice):
    return {r.action.card: r for r in advice.ranked if isinstance(r.action, Buy)}


def test_next_shuffle_odds_hand_computed(engine):
    # 7 Copper + 3 Estate: $5 needs all 5 cards Copper: C(7,5)/C(10,5) = 21/252 = 1/12; E[$] = 5 x 7/10.
    # + Silver (11 cards, C(11,5) = 462 hands): $5 = 5 Copper (21) or Silver + >=3 Copper among 4 others
    # (C(7,3)C(3,1) + C(7,4) = 105 + 35), so 161/462 = 23/66; $6 = Silver + 4 Copper: 35/462 = 5/66;
    # E[$] = 5 x 9/11 = 45/11.
    model = ShuffleOddsModel(engine.registry)
    now, silver = model.odds(STARTING_DECK), model.odds({**STARTING_DECK, "Silver": 1})
    assert now.method == "exact" and now.p_at_least == {5: Fraction(1, 12), 6: 0, 8: 0}
    assert now.expected_coins == Fraction(7, 2)
    assert silver.p_at_least == {5: Fraction(23, 66), 6: Fraction(5, 66), 8: 0}
    assert silver.expected_coins == Fraction(45, 11)
    assert silver.to_dict()["p_at_least"]["5"]["p_exact"] == "23/66"


def _brute_force_odds(deck, engine):
    """Reference: every 5-card subset of the labelled deck, every subset for each later draw, the play
    rule of policy/odds.py written out plainly (one non-terminal at a time)."""
    cards = [engine.card(name) for name, n in sorted(deck.items()) for _ in range(n)]
    density = sum(c.treasure_value for c in cards) / len(cards)
    dist = Counter()

    def playable(c):
        e = c.effects
        if not c.is_action:
            return False
        if c.is_terminal:
            return bool(e.cards or e.coins or c.special in ("moneylender", "library"))
        return bool(e.cards or e.coins or e.actions >= 2 or c.special == "merchant")

    def value(c, hand_size):
        draw = max(0, 8 - hand_size) if c.special == "library" else c.effects.cards
        has_copper = any(x.name == "Copper" for x in cards)
        return c.effects.coins + draw * density + (2 if c.special == "moneylender" and has_copper else 0)

    def play(hand, rest, actions, bonus, merchants, lenders, p):
        options = [c for c in hand if playable(c)] if actions else []
        card = next((c for c in options if not c.is_terminal), None)
        if card is None and options:
            card = max(options, key=lambda c: (value(c, len(hand)), [-ord(x) for x in c.name]))
        if card is None:
            coins = bonus + sum(c.treasure_value for c in hand)
            coins += merchants if any(c.name == "Silver" for c in hand) else 0
            coins += 2 * min(lenders, sum(c.name == "Copper" for c in hand))
            dist[coins] += p
            return
        hand = list(hand)
        hand.remove(card)
        n = max(0, 7 - len(hand)) if card.special == "library" else card.effects.cards
        n = min(n, len(rest))
        args = (actions + card.effects.actions - 1, bonus + card.effects.coins,
                merchants + (card.special == "merchant"), lenders + (card.special == "moneylender"))
        for drawn in combinations(range(len(rest)), n):
            left = [c for i, c in enumerate(rest) if i not in drawn]
            play(hand + [rest[i] for i in drawn], left, *args, p / comb(len(rest), n))

    for idx in combinations(range(len(cards)), 5):
        play([cards[i] for i in idx], [c for i, c in enumerate(cards) if i not in idx], 1, 0, 0, 0,
             Fraction(1, comb(len(cards), 5)))
    return dist


@pytest.mark.parametrize("deck", [
    {"Copper": 4, "Silver": 2, "Estate": 2, "Smithy": 1, "Market": 1, "Village": 1},
    {"Copper": 5, "Silver": 1, "Estate": 3, "Militia": 1, "Moat": 1, "Merchant": 1},
    {"Copper": 4, "Silver": 1, "Estate": 2, "Moneylender": 1, "Library": 1, "Festival": 1, "Laboratory": 1},
])
def test_next_shuffle_odds_match_brute_force(engine, deck):
    dist = _brute_force_odds(deck, engine)
    odds = ShuffleOddsModel(engine.registry).odds(deck)
    assert odds.method == "exact"
    assert odds.p_at_least == {t: sum(p for c, p in dist.items() if c >= t) for t in (5, 6, 8)}
    assert odds.expected_coins == sum(c * p for c, p in dist.items())


def test_next_shuffle_monte_carlo_fallback_is_labelled(engine, monkeypatch):
    deck = {"Copper": 7, "Estate": 3, "Silver": 2, "Smithy": 1, "Village": 1}
    exact = ShuffleOddsModel(engine.registry).odds(deck)
    monkeypatch.setattr(odds_module, "EXACT_LIMIT", 1)
    sampled = ShuffleOddsModel(engine.registry).odds(deck)
    assert sampled.method == "monte_carlo" and sampled.to_dict()["p_at_least"]["5"]["p_exact"] is None
    assert abs(sampled.p_at_least[5] - float(exact.p_at_least[5])) < 0.05


def test_silver_beats_cheap_actions_on_three(policy, make_state):
    # The reported case: 3 players, First Game kingdom, 7 Copper + 3 Estate, $3. Silver lifts P($>=5)
    # from 1/12 to 23/66 (DominionStrategy "Silver": Silver/Silver hits $5 on turn 3/4 ~91%); Moat is a
    # weak terminal +2 Cards that only blocks Militia here (DominionStrategy "Moat").
    advice = policy.recommend(buy_state(make_state, 3, num_players=3))
    buys = ranked_buys(advice)
    assert advice.best.action == Buy("Silver")
    assert all(buys["Silver"].score > r.score for c, r in buys.items() if c != "Silver")
    assert any("P($>=5) 8%->35%" in r for r in buys["Silver"].reasons)
    assert any("mild attack" in r and "weaker than Smithy" in r for r in buys["Moat"].reasons)
    shuffle = advice.best.to_dict()["next_shuffle"]
    assert shuffle["buy_nothing"]["p_at_least"]["5"]["p_exact"] == "1/12"
    assert shuffle["after_buy"]["p_at_least"]["5"]["p_exact"] == "23/66"


def test_market_beats_second_terminal(policy, make_state):
    # Deck 7 Copper + 3 Estate + Militia, $5: a second terminal collides (DominionStrategy "Terminal":
    # two terminals in a ~12-card deck collide often); Market is non-terminal (building-the-first-game-
    # engine article: Militia/Silver, then Markets).
    advice = policy.recommend(buy_state(make_state, 5, {**STARTING_DECK, "Militia": 1}))
    buys = ranked_buys(advice)
    assert advice.best.action == Buy("Market")
    for terminal in ("Militia", "Smithy", "Moat"):
        assert buys["Market"].score > buys[terminal].score
        assert any("Terminal collision" in r for r in buys[terminal].reasons)


def test_five_coins_without_terminals_picks_a_real_card(policy, make_state):
    # $5 with no terminal yet: a $4-$5 Action (Militia/Smithy/Market) or Silver are all reasonable
    # (Militia/Silver is the suggested opening for this kingdom in "Building the First Game engine";
    # Smithy-Big Money is a standard benchmark). Weak cards must not win.
    advice = policy.recommend(buy_state(make_state, 5, num_players=3))
    assert advice.best.action not in (Buy("Moat"), Buy("Cellar"), Buy("Workshop"))


@pytest.mark.parametrize("owned", [STARTING_DECK, {**STARTING_DECK, "Silver": 1, "Militia": 1}])
def test_six_coins_early_buys_gold(policy, make_state, owned):
    # Big Money: Gold at $6-7 (DominionStrategy "Money strategies"); one Gold before Markets (First Game).
    assert policy.recommend(buy_state(make_state, 6, owned)).best.action == Buy("Gold")


def test_moat_defence_depends_on_attacks_and_opponents(policy, make_state):
    def moat_defence(kingdom, players):
        advice = policy.recommend(buy_state(make_state, 2, kingdom=kingdom, num_players=players))
        return next(r for r in ranked_buys(advice)["Moat"].reasons if "Moat blocks" in r)

    witch = ["Witch", "Moat", "Village", "Smithy", "Market", "Cellar", "Chapel", "Mine", "Workshop", "Remodel"]
    assert moat_defence(witch, 2).startswith("[+2.0] Moat blocks Witch (a junking attack) from 1 opponent")
    assert moat_defence(witch, 3).startswith("[+3.0]")
    assert moat_defence(FIRST_GAME, 2).startswith("[+0.4] Moat blocks Militia (a mild attack)")


def test_unknown_deck_skips_odds_with_typical_money_value(policy, make_state):
    advice = policy.recommend(make_state(phase="buy", coins=3))
    silver = ranked_buys(advice)["Silver"]
    assert silver.next_shuffle is None and silver.score == 5.0  # the previous all-in early value
    assert any("deck unknown: next-shuffle odds not computed" in r for r in silver.reasons)


def test_odds_helper_keeps_the_import_graph_acyclic():
    # policy.odds -> planning.draws / planning.analysis -> rules only (planning.evaluation reads policy
    # weights), and both import orders work in a fresh interpreter.
    root = Path(__file__).resolve().parents[2] / "src" / "card_game_engine" / "games" / "dominion"
    for module in ("planning/draws.py", "planning/analysis.py"):
        tree = ast.parse((root / module).read_text())
        imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level]
        assert all(m.startswith(("rules", "draws")) or m == "" for m in imported), (module, imported)
    pkg = "card_game_engine.games.dominion"
    for first, second in (("policy", "planning"), ("planning", "policy"), ("inference", "policy")):
        code = f"import {pkg}.{first}, {pkg}.{second}"
        subprocess.run([sys.executable, "-c", code], check=True, cwd=root.parents[2])

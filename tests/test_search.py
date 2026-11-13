"""Generic expectimax search (core/search.py) on small hand-written toy models (no Dominion)."""

import json
import math
import random

import pytest

from card_game_engine.core import ActionValue, PlanningModel, SearchConfig, SearchStats, expectimax


class TableModel:
    """Toy model defined by a table of named nodes.

    spec[name] is one of:
      ("decision", {action: child_name, ...}, heuristic_value)
      ("chance", [(p, child_name, label), ...], heuristic_value)
      ("leaf", value)
    Chance nodes report exact outcomes only when there are at most ``max_outcomes`` of them.
    """

    def __init__(self, spec):
        self.spec = spec

    def decisions(self, s):
        node = self.spec[s]
        return list(node[1]) if node[0] == "decision" else []

    def apply(self, s, a):
        return self.spec[s][1][a]

    def is_chance(self, s):
        return self.spec[s][0] == "chance"

    def outcomes(self, s, max_outcomes):
        outs = self.spec[s][1]
        if len(outs) > max_outcomes:
            return None
        return [(p, child, label) for p, child, label in outs]

    def sample(self, s, rng):
        outs = self.spec[s][1]
        p, child, label = rng.choices(outs, weights=[o[0] for o in outs])[0]
        return child, label

    def is_terminal(self, s):
        return self.spec[s][0] == "leaf"

    def evaluate(self, s):
        node = self.spec[s]
        return node[1] if node[0] == "leaf" else node[2]

    def key(self, s):
        return s

    def action_key(self, a):
        return str(a)


def by_key(results):
    return {av.action_key: av for av in results}


# ---------------------------------------------------------------- basics

COIN_FLIP = {
    "root": ("decision", {"A": "safe", "B": "flip"}, 0.0),
    "safe": ("leaf", 0.6),
    "flip": ("chance", [(0.5, "win", "heads"), (0.5, "lose", "tails")], 0.5),
    "win": ("leaf", 1.0),
    "lose": ("leaf", 0.0),
}


def test_model_satisfies_protocol():
    assert isinstance(TableModel(COIN_FLIP), PlanningModel)


def test_coin_flip_guaranteed_beats_gamble():
    results, stats = expectimax(TableModel(COIN_FLIP), "root", SearchConfig())
    assert [av.action_key for av in results] == ["A", "B"]
    a, b = results
    assert a.value == pytest.approx(0.6) and b.value == pytest.approx(0.5)
    assert a.method == b.method == "exact"
    assert a.stderr is None and b.stderr is None
    assert b.principal_line == ["B", "heads"]  # equal probabilities: tie broken by label
    assert stats.exact_chance_nodes == 1 and stats.sampled_chance_nodes == 0
    assert not stats.budget_exhausted


def test_exact_aggregation_over_three_weighted_outcomes():
    spec = {
        "root": ("decision", {"go": "c"}, 0.0),
        "c": ("chance", [(0.2, "x", "x"), (0.3, "y", "y"), (0.5, "z", "z")], 0.0),
        "x": ("leaf", 10.0), "y": ("leaf", 4.0), "z": ("leaf", -2.0),
    }
    (av,), stats = expectimax(TableModel(spec), "root")
    assert av.value == pytest.approx(0.2 * 10 + 0.3 * 4 + 0.5 * -2)  # 2.2
    assert av.method == "exact" and av.stderr is None
    assert av.principal_line == ["go", "z"]  # most probable outcome
    assert stats.chance_nodes == stats.exact_chance_nodes == 1


@pytest.mark.parametrize("outs", [
    [(0.5, "x", "x"), (0.4, "y", "y")],                      # sums to 0.9
    [(0.6, "x", "x"), (0.6, "y", "y")],                      # sums to 1.2
    [(1.2, "x", "x"), (-0.2, "y", "y")],                     # negative probability
    [],                                                      # empty distribution
])
def test_probabilities_must_be_normalised(outs):
    spec = {"root": ("decision", {"go": "c"}, 0.0), "c": ("chance", outs, 0.0),
            "x": ("leaf", 1.0), "y": ("leaf", 0.0)}
    with pytest.raises(ValueError):
        expectimax(TableModel(spec), "root")


def test_probabilities_within_tolerance_accepted():
    spec = {"root": ("decision", {"go": "c"}, 0.0),
            "c": ("chance", [(0.1, "x", "x")] * 10, 0.0), "x": ("leaf", 1.0)}  # fsum == 1 within 1e-9
    (av,), _ = expectimax(TableModel(spec), "root")
    assert av.value == pytest.approx(1.0)


# ---------------------------------------------------------------- lookahead beats greedy

GREEDY_TRAP = {
    # Immediate evaluation prefers "safe" (5 vs 1); one step deeper "invest" pays 10.
    "root": ("decision", {"safe": "S", "invest": "I"}, 0.0),
    "S": ("decision", {"stop": "S_end"}, 5.0),
    "I": ("decision", {"cash": "I_end"}, 1.0),
    "S_end": ("leaf", 5.0),
    "I_end": ("leaf", 10.0),
}


def test_depth_one_is_greedy():
    results, stats = expectimax(TableModel(GREEDY_TRAP), "root", SearchConfig(max_depth=1))
    assert results[0].action_key == "safe"
    assert by_key(results)["invest"].value == 1.0
    assert stats.max_depth_reached == 1


def test_depth_two_finds_better_plan():
    results, stats = expectimax(TableModel(GREEDY_TRAP), "root", SearchConfig(max_depth=2))
    assert [av.action_key for av in results] == ["invest", "safe"]
    assert results[0].value == 10.0 and results[1].value == 5.0
    assert results[0].principal_line == ["invest", "cash"]
    assert results[0].depth_reached == 2


def test_ties_broken_by_action_key():
    spec = {"root": ("decision", {"zeta": "l1", "alpha": "l2", "mid": "l3"}, 0.0),
            "l1": ("leaf", 0.3), "l2": ("leaf", 0.3), "l3": ("leaf", 0.1 + 0.2)}  # float noise ignored
    results, _ = expectimax(TableModel(spec), "root")
    assert [av.action_key for av in results] == ["alpha", "mid", "zeta"]


def test_terminal_or_chance_root_returns_nothing():
    m = TableModel(COIN_FLIP)
    assert expectimax(m, "win")[0] == []
    assert expectimax(m, "flip")[0] == []


# ---------------------------------------------------------------- Monte Carlo

def dice_spec(actions=("roll",)):
    spec = {"root": ("decision", {a: f"die_{a}" for a in actions}, 0.0)}
    for a in actions:
        spec[f"die_{a}"] = ("chance", [(1 / 6, f"face{i}", str(i)) for i in range(1, 7)], 3.5)
    for i in range(1, 7):
        spec[f"face{i}"] = ("leaf", float(i))
    return spec


MC = dict(max_outcomes_per_chance=1, samples_per_chance=200)


def test_monte_carlo_reproducible_and_seed_dependent():
    m = TableModel(dice_spec())
    (a1,), s1 = expectimax(m, "root", SearchConfig(seed=7, **MC))
    (a2,), _ = expectimax(m, "root", SearchConfig(seed=7, **MC))
    (b,), _ = expectimax(m, "root", SearchConfig(seed=8, **MC))
    assert a1.to_dict() == a2.to_dict()
    assert a1.value != b.value
    assert a1.method == "monte_carlo" and a1.stderr is not None and a1.stderr > 0
    assert s1.sampled_chance_nodes == 1 and s1.exact_chance_nodes == 0


def test_monte_carlo_estimate_near_truth():
    true_sd = math.sqrt(35 / 12)
    within = 0
    for seed in range(20):
        (av,), _ = expectimax(TableModel(dice_spec()), "root", SearchConfig(seed=seed, **MC))
        assert av.stderr == pytest.approx(true_sd / math.sqrt(200), rel=0.25)
        within += abs(av.value - 3.5) <= 3 * av.stderr
    assert within >= 19


def test_root_rng_is_seeded_from_seed_and_action_key():
    m = TableModel(dice_spec())
    (av,), _ = expectimax(m, "root", SearchConfig(seed=3, **MC))
    rng = random.Random("3:roll")
    expected = sum(m.evaluate(m.sample("die_roll", rng)[0]) for _ in range(200)) / 200
    assert av.value == pytest.approx(expected)


def test_root_actions_have_independent_rngs():
    cfg = SearchConfig(seed=11, max_outcomes_per_chance=1, samples_per_chance=50)
    xy = by_key(expectimax(TableModel(dice_spec(("x", "y"))), "root", cfg)[0])
    yx = by_key(expectimax(TableModel(dice_spec(("y", "x"))), "root", cfg)[0])
    xyz = by_key(expectimax(TableModel(dice_spec(("x", "y", "z"))), "root", cfg)[0])
    alone = by_key(expectimax(TableModel(dice_spec(("x",))), "root", cfg)[0])
    assert xy["x"].value == yx["x"].value == xyz["x"].value == alone["x"].value
    assert xy["y"].value == yx["y"].value == xyz["y"].value
    assert xy["x"].value != xy["y"].value  # different keys -> different streams


def test_mixed_method_when_sampled_node_leads_to_exact_nodes():
    spec = {
        "root": ("decision", {"go": "big"}, 0.0),
        "big": ("chance", [(1 / 3, "c1", "a"), (1 / 3, "c2", "b"), (1 / 3, "c3", "c")], 0.0),
        "c1": ("chance", [(0.5, "w", "w"), (0.5, "l", "l")], 0.0),
        "c2": ("leaf", 1.0),
        "c3": ("leaf", 0.0),
        "w": ("leaf", 1.0), "l": ("leaf", 0.0),
    }
    (av,), stats = expectimax(TableModel(spec), "root", SearchConfig(max_outcomes_per_chance=2))
    assert av.method == "mixed" and av.stderr is not None
    assert stats.sampled_chance_nodes == 1 and stats.exact_chance_nodes >= 1
    assert abs(av.value - 0.5) <= 4 * av.stderr


# ---------------------------------------------------------------- limits

class InfiniteBinary:
    """Never-ending binary decision tree: state = tuple of moves, value = number of 'R' moves."""

    def __init__(self, root_actions=("L", "R")):
        self.root_actions = root_actions

    def decisions(self, s):
        return list(self.root_actions) if not s else ["L", "R"]

    def apply(self, s, a):
        return s + (a,)

    def is_chance(self, s):
        return False

    def outcomes(self, s, max_outcomes):
        raise AssertionError("no chance nodes")

    def sample(self, s, rng):
        raise AssertionError("no chance nodes")

    def is_terminal(self, s):
        return False

    def evaluate(self, s):
        return float(s.count("R"))

    def key(self, s):
        return s

    def action_key(self, a):
        return a


def test_depth_limit_respected():
    results, stats = expectimax(InfiniteBinary(), (), SearchConfig(max_depth=3))
    assert stats.max_depth_reached == 3
    assert all(av.depth_reached <= 3 for av in results)
    best = results[0]
    assert best.action_key == "R" and best.value == 3.0
    assert best.principal_line == ["R", "R", "R"]
    assert stats.nodes_expanded == 1 + 2 + 4 + 8  # root + full tree to depth 3
    assert not stats.budget_exhausted


def test_node_budget_respected_and_no_root_action_starved():
    model = InfiniteBinary(root_actions=("a", "b", "c"))
    results, stats = expectimax(model, (), SearchConfig(max_depth=30, max_nodes=100))
    assert stats.budget_exhausted and stats.budget_cutoffs > 0
    assert stats.nodes_expanded <= 100 + 1
    assert {av.action_key for av in results} == {"a", "b", "c"}
    for av in results:
        assert av.nodes >= 100 // 3
        assert av.truncated
        assert math.isfinite(av.value)


def test_tiny_budget_still_values_every_root_action():
    model = InfiniteBinary(root_actions=tuple("abcde"))
    results, stats = expectimax(model, (), SearchConfig(max_depth=5, max_nodes=2))
    assert len(results) == 5
    assert stats.budget_exhausted and stats.nodes_expanded <= 2 + 1
    # 2 nodes over 5 actions: allowances 0,0,0,1,1 -> three actions scored by evaluate only
    # (value 0), the last two expand one level and see an "R" child (value 1).
    assert [by_key(results)[k].nodes for k in "abcde"] == [0, 0, 0, 1, 1]
    assert [by_key(results)[k].value for k in "abcde"] == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert all(av.truncated for av in results)


# ---------------------------------------------------------------- transpositions

class SumGame:
    """Add 1 or 2 per step; state = running total, so +1,+2 and +2,+1 transpose. Value = total % 3."""

    def decisions(self, s):
        return ["+1", "+2"]

    def apply(self, s, a):
        return s + int(a)

    def is_chance(self, s):
        return False

    def outcomes(self, s, max_outcomes):
        return None

    def sample(self, s, rng):
        raise AssertionError

    def is_terminal(self, s):
        return s >= 7

    def evaluate(self, s):
        return float(s % 3) + (10.0 if s == 7 else 0.0)

    def key(self, s):
        return s

    def action_key(self, a):
        return a


def test_cache_hits_on_transpositions_without_changing_values():
    cached, s1 = expectimax(SumGame(), 0, SearchConfig(max_depth=5))
    plain, s2 = expectimax(SumGame(), 0, SearchConfig(max_depth=5, use_cache=False))
    assert s1.cache_hits > 0 and s2.cache_hits == 0
    assert s1.nodes_expanded < s2.nodes_expanded
    assert [(a.action_key, a.value) for a in cached] == [(a.action_key, a.value) for a in plain]
    assert cached[0].value >= 10.0  # reaching exactly 7 within 5 steps is possible


def test_sampled_subtrees_are_not_cached():
    spec = dice_spec(("x", "y"))
    spec["root"] = ("decision", {"x": "die_x", "y": "die_x"}, 0.0)  # both actions reach the same chance node
    cfg = SearchConfig(seed=1, **MC)
    results, stats = expectimax(TableModel(spec), "root", cfg)
    vals = by_key(results)
    assert vals["x"].value != vals["y"].value  # each root action sampled with its own RNG
    plain = by_key(expectimax(TableModel(spec), "root", SearchConfig(seed=1, use_cache=False, **MC))[0])
    assert {k: v.value for k, v in vals.items()} == {k: v.value for k, v in plain.items()}


# ---------------------------------------------------------------- serialisation / config

def test_results_are_json_serialisable():
    results, stats = expectimax(TableModel(dice_spec(("x", "y"))), "root", SearchConfig(**MC))
    payload = {"results": [av.to_dict() for av in results], "stats": stats.to_dict(),
               "config": SearchConfig(**MC).to_dict()}
    decoded = json.loads(json.dumps(payload, allow_nan=False))
    assert decoded["results"][0]["method"] == "monte_carlo"
    assert isinstance(results[0], ActionValue) and isinstance(stats, SearchStats)


def test_config_validation_and_round_trip():
    cfg = SearchConfig(max_depth=3, seed=5)
    assert SearchConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        SearchConfig(max_depth=0)
    with pytest.raises(ValueError):
        SearchConfig(samples_per_chance=1)
    with pytest.raises(ValueError):
        SearchConfig.from_dict({"bogus": 1})

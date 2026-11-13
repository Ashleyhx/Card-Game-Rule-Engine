"""Benchmark: lookahead planner vs the V1 heuristic policy on the bundled example snapshots.

For each snapshot with enough information it records the heuristic's first action
(``HeuristicPolicy.recommend``), the planner's first action (``plan_next_action``, money
objective, fixed seed and budget) and the planner's expected value for BOTH actions
(``compare_actions``), so the two choices are measured with the same evaluator.

The measure is the ``money`` objective: expected coins generated THIS turn, computed by
exact draw enumeration given the snapshot's zone knowledge. It is not the whole game
(it ignores what is bought, deck quality and future turns), so the numbers say which
first action generates more coins this turn and nothing more.

Run from the repo root (standard library only)::

    python benchmarks/planner_vs_heuristic.py [--json] [--examples NAME ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import card_game_engine  # noqa: F401
except ImportError:  # not installed: use the source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from card_game_engine.games.dominion.cli.commands import example_names, load_example
from card_game_engine.games.dominion.planning import PlanConfig, compare_actions, plan_next_action
from card_game_engine.games.dominion.policy import HeuristicPolicy
from card_game_engine.games.dominion.rules import DominionRuleEngine, state_from_dict

#: Fixed settings: money objective, end-of-turn horizon (overrides a snapshot's planning.horizon
#: so every row measures the same thing), default depth/budget, seed 0.
CONFIG = dict(objective="money", horizon="turn", max_depth=8, max_nodes=20000, seed=0)
INCONCLUSIVE = ("lower_bound", "partial")


def skip_reason(result) -> str | None:
    """Why a snapshot is left out (None = benchmarked)."""
    if result.information_level == "insufficient":
        return "draw pile size unknown (information level 'insufficient'): draws cannot be modelled"
    if result.blocked:
        return "state is blocked (pending draws / manual steps)"
    if result.recommended is None:
        return "no legal actions"
    return None


def verdict(same: bool, p, h) -> str:
    """Planner better / equal / worse by the measured value, or inconclusive when a bound decides it."""
    if same:
        return "equal"
    pv, hv = p.expected_value, h.expected_value
    p_bound, h_bound = p.value_provenance in INCONCLUSIVE, h.value_provenance in INCONCLUSIVE
    diff = round(pv - hv, 9)
    if not p_bound and not h_bound:
        return "better" if diff > 0 else "worse" if diff < 0 else "equal"
    if p.value_provenance == "lower_bound" and not h_bound and diff > 0:
        return "better"   # planner's true value >= its bound > heuristic's exact value
    if h.value_provenance == "lower_bound" and not p_bound and diff < 0:
        return "worse"    # heuristic's true value >= its bound > planner's exact value
    return "inconclusive"


def benchmark_one(name: str, engine: DominionRuleEngine) -> dict:
    state = state_from_dict(load_example(name), engine.registry, engine.ruleset)
    cfg = PlanConfig(**CONFIG)
    start = time.perf_counter()
    result = plan_next_action(state, cfg, engine)
    runtime = time.perf_counter() - start
    reason = skip_reason(result)
    if reason:
        return {"example": name, "skipped": reason}
    heuristic = HeuristicPolicy(engine).recommend(state).best.action
    planner_key = result.recommended.action_key
    same = heuristic.describe() == planner_key
    candidates = [result.recommended.action] + ([] if same else [heuristic])
    cands = {c.action_key: c for c in compare_actions(state, candidates, cfg, engine).candidates}
    p, h = cands[planner_key], cands[heuristic.describe()]
    return {
        "example": name,
        "information_level": result.information_level,
        "heuristic_action": heuristic.describe(),
        "planner_action": planner_key,
        "heuristic_value": h.expected_value,
        "heuristic_provenance": h.value_provenance,
        "planner_value": p.expected_value,
        "planner_provenance": p.value_provenance,
        "gap": p.expected_value - h.expected_value,
        "verdict": verdict(same, p, h),
        "planner_runtime_s": runtime,
        "planner_nodes": result.stats["nodes_expanded"],
        "budget_exhausted": result.stats["budget_exhausted"],
    }


def run_benchmark(names: list[str] | None = None) -> dict:
    engine = DominionRuleEngine()
    rows = [benchmark_one(n, engine) for n in (names or example_names())]
    done = [r for r in rows if "skipped" not in r]
    totals = {v: sum(r["verdict"] == v for r in done) for v in ("better", "equal", "worse", "inconclusive")}
    totals["benchmarked"] = len(done)
    totals["skipped"] = len(rows) - len(done)
    totals["mean_planner_runtime_s"] = sum(r["planner_runtime_s"] for r in done) / len(done) if done else None
    totals["total_planner_nodes"] = sum(r["planner_nodes"] for r in done)
    return {"config": CONFIG, "objective_note": "expected coins generated this turn (money objective); not the whole game",
            "rows": [r for r in rows if "skipped" not in r],
            "skipped": [{"example": r["example"], "reason": r["skipped"]} for r in rows if "skipped" in r],
            "totals": totals}


def _value(v: float, provenance: str) -> str:
    return (">=" if provenance == "lower_bound" else "") + f"{v:.2f}" + ("*" if provenance == "partial" else "")


def render(report: dict) -> str:
    head = (f"{'example':<28} {'heuristic first action':<41} {'planner first action':<41} "
            f"{'E[$] heur':>9} {'E[$] plan':>9} {'gap':>6}  {'verdict':<12} {'time':>7} {'nodes':>6}")
    lines = ["Planner vs heuristic - objective: expected coins generated this turn (money), exact draw enumeration",
             f"Config: {report['config']}", "", head, "-" * len(head)]
    for r in report["rows"]:
        lines.append(
            f"{r['example']:<28} {r['heuristic_action'][:41]:<41} {r['planner_action'][:41]:<41} "
            f"{_value(r['heuristic_value'], r['heuristic_provenance']):>9} "
            f"{_value(r['planner_value'], r['planner_provenance']):>9} {r['gap']:>+6.2f}  {r['verdict']:<12} "
            f"{r['planner_runtime_s']:>6.3f}s {r['planner_nodes']:>6}")
    t = report["totals"]
    lines += ["", ">= : lower bound (cards of unknown identity counted as $0); * : partial (player choice not modelled)."]
    if report["skipped"]:
        lines += ["", "Skipped:"] + [f"  {s['example']}: {s['reason']}" for s in report["skipped"]]
    mean = f"{t['mean_planner_runtime_s']:.3f}s" if t["mean_planner_runtime_s"] is not None else "n/a"
    lines += ["", f"Totals over {t['benchmarked']} snapshot(s): planner better {t['better']}, equal {t['equal']}, "
                  f"worse {t['worse']}, inconclusive (lower bound) {t['inconclusive']}; "
                  f"mean planner runtime {mean}, {t['total_planner_nodes']} nodes in total."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help="Machine-readable output")
    ap.add_argument("--examples", nargs="+", metavar="NAME", help="Only these bundled examples (default: all)")
    args = ap.parse_args(argv)
    report = run_benchmark(args.examples)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

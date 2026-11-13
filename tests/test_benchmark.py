"""The planner-vs-heuristic benchmark runs (tiny subset) and measures both choices with the same evaluator."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "planner_vs_heuristic.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("planner_vs_heuristic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_subset_json(bench, capsys):
    names = ["stranded_draw_vs_money", "opening_5_2", "unknown_deck_minimal"]
    assert bench.main(["--json", "--examples", *names]) == 0
    report = json.loads(capsys.readouterr().out)
    rows = {r["example"]: r for r in report["rows"]}
    stranded = rows["stranded_draw_vs_money"]
    assert stranded["heuristic_action"] == "Play Smithy" and stranded["planner_action"] == "Play Militia"
    assert stranded["heuristic_value"] == pytest.approx(3.0) and stranded["planner_value"] == pytest.approx(5.0)
    assert stranded["gap"] == pytest.approx(2.0) and stranded["verdict"] == "better"
    assert rows["opening_5_2"]["verdict"] == "equal"
    assert [s["example"] for s in report["skipped"]] == ["unknown_deck_minimal"]
    t = report["totals"]
    assert (t["benchmarked"], t["better"], t["equal"], t["worse"], t["skipped"]) == (2, 1, 1, 0, 1)
    assert t["mean_planner_runtime_s"] >= 0


def test_benchmark_text_table(bench, capsys):
    assert bench.main(["--examples", "stranded_draw_vs_money"]) == 0
    out = capsys.readouterr().out
    assert "Play Militia" in out and "Totals over 1 snapshot(s): planner better 1" in out


class _C:
    def __init__(self, value, provenance):
        self.expected_value, self.value_provenance = value, provenance


def test_verdict_respects_lower_bounds(bench):
    v = bench.verdict
    assert v(False, _C(5, "exact"), _C(3, "expected")) == "better"
    assert v(False, _C(4, "exact"), _C(2, "lower_bound")) == "inconclusive"  # heuristic may be worth more
    assert v(False, _C(2, "exact"), _C(4, "lower_bound")) == "worse"
    assert v(False, _C(4, "lower_bound"), _C(2, "exact")) == "better"
    assert v(False, _C(3, "lower_bound"), _C(3, "lower_bound")) == "inconclusive"
    assert v(True, _C(3, "lower_bound"), _C(3, "lower_bound")) == "equal"  # same action

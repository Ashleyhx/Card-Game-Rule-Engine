"""Session CLI (`cardgame dominion session ...`) and the Phase 2b integration fixes."""

import json
from fractions import Fraction

import pytest

from card_game_engine.cli import main as cardgame_main
from card_game_engine.games.dominion.cli.commands import load_example
from card_game_engine.games.dominion.cli.session_cli import (
    deduced_hand, fmt_p, interactive_session, load_session,
)
from card_game_engine.games.dominion.inference import to_game_state
from card_game_engine.games.dominion.planning import PlanConfig
from card_game_engine.games.dominion.rules import SnapshotError, action_to_dict, state_from_dict
from tests.dominion.conftest import FIRST_GAME


def run(capsys, *argv):
    code = cardgame_main(["dominion", "session", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def run_json(capsys, *argv):
    code, out, _ = run(capsys, *argv, "--json")
    return code, json.loads(out)


@pytest.fixture
def game(tmp_path, capsys):
    path = str(tmp_path / "game.json")
    code, out, _ = run(capsys, "start", path)
    assert code == 0 and "Initial deck loaded: Copper x7, Estate x3" in out
    return path


# ---------------------------------------------------------------- spec section 7 workflow

def test_spec_workflow(game, capsys):
    code, out, _ = run(capsys, "hand", game, "4 Copper, Estate")
    assert code == 0
    assert "Recorded #0: hand Copper x4, Estate" in out
    assert "Draw pile (deduced, 5 cards, order unknown): Copper x3, Estate x2" in out
    assert "known exactly - Copper x3, Estate x2" in out

    assert run(capsys, "buy", game, "Silver")[0] == 0
    code, out, _ = run(capsys, "end-turn", game)
    assert code == 0 and "Hand (deduced): Copper x3, Estate x2" in out

    # Turn 2: the hand is never entered - show presents the deduced hand, not an empty one.
    code, out, _ = run(capsys, "show", game)
    assert code == 0 and "Hand (deduced): Copper x3, Estate x2" in out and "Turn 2" in out
    code, data = run_json(capsys, "show", game)
    assert data["current_hand"] == {"status": "deduced", "cards": ["Copper"] * 3 + ["Estate"] * 2,
                                    "pending_draws": 5, "source": data["current_hand"]["source"]}

    assert run(capsys, "buy", game, "Silver")[0] == 0
    assert run(capsys, "end-turn", game)[0] == 0

    # Turn 3: reshuffle of 7 Copper, 3 Estate, 2 Silver; an exact distribution summing to 1.
    code, out, _ = run(capsys, "predict", game)
    assert code == 0 and "P(reshuffle while drawing): 1" in out and "Sum over all 12 hands: 1" in out
    code, pred = run_json(capsys, "predict", game, "--top", "100")
    assert pred["which"] == "start_of_turn" and pred["turn"] == 3 and pred["method"] == "exact"
    assert pred["p_reshuffle"]["p_exact"] == "1/1" and not pred["deterministic"]
    assert sum(Fraction(d["p_exact"]) for d in pred["distribution"]) == 1
    assert pred["outcomes"] == len(pred["distribution"]) == 12

    code, out, _ = run(capsys, "hand", game, "3 Copper, Estate, Silver")
    assert code == 0 and "Draw pile (deduced, 7 cards, order unknown): Copper x4, Estate x2, Silver" in out
    assert run(capsys, "end-turn", game)[0] == 0

    code, pred = run_json(capsys, "predict", game)
    assert pred["turn"] == 4 and pred["p_at_least_one"]["Silver"]["p_exact"] == "5/7"
    code, out, _ = run(capsys, "predict", game)
    assert "5/7 = 0.714" in out
    assert out.isascii()


def test_fmt_p():
    assert fmt_p(Fraction(5, 7)) == "5/7 = 0.714"
    assert fmt_p(Fraction(1)) == "1" and fmt_p(Fraction(0)) == "0"
    assert fmt_p(0.25) == "~0.250"


def test_mutating_json_returns_belief_summary(game, capsys):
    code, data = run_json(capsys, "hand", game, "4 Copper, Estate")
    assert code == 0 and data["ok"] and data["events"] == 1
    assert data["recorded"] == [{"kind": "observe_hand", "cards": ["Copper"] * 4 + ["Estate"]}]
    assert data["belief"]["inferred"]["draw_pile"]["certain"] == {"Copper": 3, "Estate": 2}
    assert data["next_hand"]["deterministic"] and data["next_hand"]["hand"] == ["Copper"] * 3 + ["Estate"] * 2
    assert data["current_hand"]["status"] == "observed"


# ---------------------------------------------------------------- consistency

def test_contradiction_exit_1_and_file_unchanged(game, capsys, tmp_path):
    run(capsys, "hand", game, "4 Copper, Estate")
    before = open(game, "rb").read()
    code, _, err = run(capsys, "end-turn", game)
    assert code == 0
    after_end = open(game, "rb").read()
    # Turn 2's hand is forced to be 3 Copper, 2 Estate: anything else contradicts the history.
    code, _, err = run(capsys, "hand", game, "4 Copper, Estate")
    assert code == 1 and "inconsistent" in err and "not changed" in err
    assert open(game, "rb").read() == after_end
    code, data = run_json(capsys, "hand", game, "5 Estate")
    assert code == 1 and data == {"ok": False, "error": "inconsistent", "message": data["message"]}
    assert "only 3 owned" in data["message"]
    code, _, err = run(capsys, "buy", game, "Province")  # $3 cannot buy a Province
    assert code == 1 and "impossible resource change" in err
    code, _, err = run(capsys, "buy", game, "Provnce")   # typo: rejected with a suggestion
    assert code == 1 and "Did you mean: Province?" in err
    assert open(game, "rb").read() == after_end
    assert before != after_end
    assert [p.name for p in tmp_path.iterdir()] == ["game.json"]  # atomic writes leave no temp files


def test_undo_correct_history_reset(game, capsys):
    run(capsys, "hand", game, "4 Copper, Estate")
    run(capsys, "buy", game, "Silver")
    code, rows = run_json(capsys, "history", game)
    assert code == 0 and [r["text"] for r in rows] == ["hand Copper x4, Estate", "buy Silver"]
    assert [r["turn"] for r in rows] == [1, 1]

    code, out, _ = run(capsys, "undo", game)
    assert code == 0 and "Undid #1: buy Silver" in out
    assert len(load_session(game).events) == 1

    code, out, _ = run(capsys, "correct", game, "0", "hand", "3 Copper, 2 Estate")
    assert code == 0 and "Corrected #0" in out and "known exactly - Copper x4, Estate" in out
    code, out, _ = run(capsys, "correct", game, "0", "hand 4 Copper, Estate")  # one quoted string also works
    assert code == 0 and "known exactly - Copper x3, Estate x2" in out

    run(capsys, "buy", game, "Silver")
    before = open(game, "rb").read()
    # $2 cannot have bought the Silver recorded later: the corrected history is inconsistent.
    code, _, err = run(capsys, "correct", game, "0", "hand", "2 Copper, 3 Estate")
    assert code == 1 and "Event #1" in err and open(game, "rb").read() == before
    code, _, err = run(capsys, "correct", game, "9", "end-turn")
    assert code == 1 and "No event #9" in err

    code, out, _ = run(capsys, "reset", game)
    assert code == 0 and "removed 2 event(s)" in out and load_session(game).events == ()
    code, _, err = run(capsys, "undo", game)
    assert code == 1 and "Nothing to undo" in err


def test_other_event_verbs(game, capsys):
    run(capsys, "hand", game, "4 Copper, Estate")
    assert run(capsys, "play", game, "Copper")[0] == 0
    assert run(capsys, "discard", game, "Estate")[0] == 0
    assert run(capsys, "gain", game, "Silver", "--to", "deck")[0] == 0
    assert run(capsys, "trash", game, "Copper", "--from", "in_play")[0] == 0
    assert run(capsys, "supply", game, "Province=7")[0] == 0
    code, out, _ = run(capsys, "draw", game, "Silver")  # the gained Silver is on top
    assert code == 0 and "Recorded #6: draw Silver" in out
    code, out, _ = run(capsys, "draw", game, "--count", "1")
    assert code == 0 and "1 card" in out
    code, rows = run_json(capsys, "history", game)
    assert [r["text"] for r in rows] == [
        "hand Copper x4, Estate", "play Copper", "discard Estate", "gain Silver to deck",
        "trash Copper from in_play", "supply Province=7", "draw Silver", "draw 1 unobserved"]
    data = load_session(game).to_dict()
    assert data["events"][3] == {"kind": "record_gain", "card": "Silver", "destination": "deck"}
    code, _, err = run(capsys, "draw", game)
    assert code == 1 and "--count" in err


# ---------------------------------------------------------------- start options

def test_start_options(tmp_path, capsys):
    path = str(tmp_path / "s.json")
    code, out, _ = run(capsys, "start", path, "--players", "3", "--kingdom", ", ".join(reversed(FIRST_GAME)))
    assert code == 0 and "3 players" in out
    code, _, err = run(capsys, "start", path)
    assert code == 2 and "--force" in err
    code, out, _ = run(capsys, "start", path, "--force")
    assert code == 0 and "2 players" in out


def test_start_from_snapshot(tmp_path, capsys):
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps(load_example("ordered_deck_known")))
    path = str(tmp_path / "mid.json")
    code, _, err = run(capsys, "start", path, "--from-snapshot", str(snap), "--players", "2")
    assert code == 2 and "--from-snapshot" in err
    code, out, _ = run(capsys, "start", path, "--from-snapshot", str(snap))
    assert code == 0 and "history before it is unknown" in out and "Turn 7" in out
    assert "top first: Gold, Estate, Silver, Copper, Estate" in out
    code, data = run_json(capsys, "show", path)
    assert any(f.startswith("started_from_snapshot") for f in data["belief"]["flags"])
    run(capsys, "play", path, "Village")
    code, out, _ = run(capsys, "show", path)
    assert "Hand (deduced): Copper x2, Estate, Gold, Smithy" in out  # the pending Village draw is the known Gold
    code, rows = run_json(capsys, "history", path)
    assert rows[0]["turn"] == 7


# ---------------------------------------------------------------- plan

def test_plan_from_session_returns_legal_action(game, capsys, engine):
    code, _, err = run(capsys, "plan", game)
    assert code == 1 and "Cannot plan" in err  # opening hand not determined yet
    run(capsys, "hand", game, "4 Copper, Estate")
    code, data = run_json(capsys, "plan", game, "--horizon", "hands:2", "--objective", "future_money")
    assert code == 0 and data["horizon"] == "hands:2"
    state = to_game_state(load_session(game))
    assert data["recommended"]["action"] in [action_to_dict(a) for a in engine.legal_actions(state)]


def test_plan_on_deduced_start_of_turn_hand(game, capsys):
    for argv in (("hand", game, "4 Copper, Estate"), ("buy", game, "Silver"), ("end-turn", game)):
        run(capsys, *argv)
    assert deduced_hand(load_session(game)) == ["Copper"] * 3 + ["Estate"] * 2
    before = open(game, "rb").read()
    code, data = run_json(capsys, "plan", game)
    assert code == 0 and any("Planning on the deduced hand" in n for n in data["session_notes"])
    assert open(game, "rb").read() == before  # planning never records anything


# ---------------------------------------------------------------- interactive

def test_interactive_scripted(tmp_path):
    path = tmp_path / "repl.json"
    script = iter(["help", "hand 4 Copper, Estate", "buy Silver", "end-turn", "show", "buy Provnce",
                   "buy Silver", "end-turn", "predict --top 2", "hand 3 Copper, Estate, Silver",
                   "plan --horizon hands:2 --objective future_money", "undo", "history", "bogus", "hand \"oops"])
    out: list[str] = []

    def fake_input(prompt):
        try:
            return next(script)
        except StopIteration:
            raise EOFError from None

    assert interactive_session(path, input_fn=fake_input, print_fn=out.append) == 0
    text = "\n".join(out)
    assert "New session" in text and "Verbs" in text
    assert "Hand (deduced): Copper x3, Estate x2" in text
    assert "Did you mean: Province?" in text
    assert "Sum over all 12 hands: 1" in text
    assert "RECOMMENDED" in text and "Undid #5" in text
    assert "invalid choice" in text and "No closing quotation" in text
    assert len(load_session(path).events) == 5  # saved after each successful mutation; undo removed one


def test_interactive_quit_and_existing_file(game):
    out: list[str] = []
    answers = iter(["show --json", "quit", "never read"])
    assert interactive_session(game, input_fn=lambda p: next(answers), print_fn=out.append) == 0
    assert any(line.startswith("{") and json.loads(line)["events"] == 0 for line in out)
    assert "never read" not in "\n".join(out)


# ---------------------------------------------------------------- fix: planning.horizon accepts hands:N

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


def test_deduced_hand_only_while_draws_pending(game, capsys):
    run(capsys, "hand", game, "4 Copper, Estate")
    run(capsys, "end-turn", game)
    code, out, _ = run(capsys, "buy", game, "Silver")  # turn 2's hand was deduced, then used
    assert code == 0 and "Hand (deduced, 2 cards): Estate x2" in out and "pending draw" not in out
    assert deduced_hand(load_session(game)) is None


@pytest.mark.parametrize("content", ['{"events": "abc"}', "[1, 2]", '{"events": [5]}'])
def test_malformed_session_file_is_an_error_not_a_crash(tmp_path, capsys, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    code, _, err = run(capsys, "show", str(path))
    assert code in (1, 2) and ("not a session" in err or "must be a JSON object" in err)
    assert path.read_text() == content

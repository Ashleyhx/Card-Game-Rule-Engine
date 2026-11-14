"""CLI and interactive-mode tests (no network, scripted input)."""

import json

import pytest

from card_game_engine.games.dominion.cli.interactive import Prompter, _resolve_followups, interactive_main
from card_game_engine.cli import main as cardgame_main
from card_game_engine.games.dominion.cli.commands import example_names, load_example
from card_game_engine.games.dominion.cli.parsing import parse_card_list, parse_deltas, parse_pile_overrides
from card_game_engine.games.dominion.rules import (
    Buy, DominionRuleEngine, EndTurn, PlayAction, PlayAllTreasures, load_registry, state_from_dict,
)
from tests.dominion.conftest import OTHER_KINGDOM


def run(capsys, *argv):
    code = cardgame_main(["dominion", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_examples_listed(capsys):
    code, out, _ = run(capsys, "examples")
    assert code == 0
    for name in example_names():
        assert name in out


@pytest.mark.parametrize("name", example_names())
def test_recommend_every_example_human_and_json(capsys, name):
    code, out, _ = run(capsys, "recommend", "--example", name)
    assert code == 0 and "RECOMMENDED" in out
    code, out, _ = run(capsys, "recommend", "--example", name, "--json")
    data = json.loads(out)
    assert code == 0 and data["recommendation"]["action"]["kind"]


def test_recommend_from_file(tmp_path, capsys):
    snap = {"num_players": 2, "kingdom": ["Cellar", "Market", "Merchant", "Militia", "Mine", "Moat",
                                          "Remodel", "Smithy", "Village", "Workshop"],
            "player": {"phase": "buy", "hand": ["Silver", "Silver", "Copper", "Copper"]}}
    f = tmp_path / "s.json"
    f.write_text(json.dumps(snap))
    code, out, _ = run(capsys, "recommend", str(f), "--json")
    data = json.loads(out)
    assert data["after_playing_treasures"]["coins"] == 6


def test_invalid_snapshot_exit_code(tmp_path, capsys):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"num_players": 9, "kingdom": [], "player": {}}))
    code, _, err = run(capsys, "recommend", str(f))
    assert code == 1 and "num_players" in err


def test_inconsistent_snapshot_rejected(tmp_path, capsys):
    snap = {"num_players": 2, "kingdom": ["Cellar", "Market", "Merchant", "Militia", "Mine", "Moat",
                                          "Remodel", "Smithy", "Village", "Workshop"],
            "player": {"phase": "action", "hand": [], "in_play": ["Gold"]}}
    f = tmp_path / "s.json"
    f.write_text(json.dumps(snap))
    code, out, _ = run(capsys, "recommend", str(f))
    assert code == 1 and "Treasures are in play" in out
    code, out, _ = run(capsys, "validate", str(f), "--json")
    assert code == 1 and json.loads(out)["valid"] is False


def test_legal_explain(capsys):
    code, out, _ = run(capsys, "legal", "-e", "opening_5_2", "--explain")
    assert code == 0 and "Play all Treasures" in out and "Costs $8; you have $0" in out


def test_cards_and_sources(capsys):
    code, out, _ = run(capsys, "cards", "throne room")
    assert code == 0 and "Throne Room" in out and "riograndegames.com" in out
    code, out, _ = run(capsys, "cards", "--kingdom", "--json")
    assert len(json.loads(out)) == 26
    code, out, _ = run(capsys, "sources")
    assert "Dominion2E.pdf" in out


def test_missing_input_is_usage_error(capsys):
    code, _, err = run(capsys, "recommend")
    assert code == 2 and "--example" in err


def test_parsers():
    reg = load_registry()
    assert parse_card_list("3 Copper, silver x2, throne room", reg) == \
        ["Copper"] * 3 + ["Silver"] * 2 + ["Throne Room"]
    assert parse_card_list("none", reg) == []
    with pytest.raises(ValueError, match="Smithy"):
        parse_card_list("Smithee", reg)
    assert parse_pile_overrides("Province=5, curse: 3", reg) == {"Province": 5, "Curse": 3}
    assert parse_deltas("+Silver x2, -Estate", reg) == {"Silver": 2, "Estate": -1}


def scripted(answers):
    it = iter(answers)
    out = []

    def _input(prompt):
        out.append(prompt)
        return next(it)

    return _input, out


def test_interactive_build_and_play_village(tmp_path):
    save = tmp_path / "snap.json"
    answers = [
        "2",                           # players
        "first",                       # kingdom preset
        "5",                           # turn number
        "n", "Province=7",             # supply changed
        "action",                      # phase
        "Village, Smithy, Copper x2, Estate",  # hand
        "",                            # in play (none)
        "changes", "+Village, +Smithy, +Silver",  # deck
        "6",                           # opponent VP
        "",                            # apply recommendation (Village)
        "Copper",                      # drew 1 card
        "q",                           # quit
    ]
    inp, prompts = scripted(answers)
    printed = []
    code = interactive_main(DominionRuleEngine(), save=str(save), input_fn=inp, print_fn=printed.append)
    text = "\n".join(printed)
    assert code == 0
    assert "RECOMMENDED: Play Village" in text
    assert "RECOMMENDED: Play Smithy" in text  # after drawing, Smithy is next
    snap = json.loads(save.read_text())
    assert snap["supply"]["Province"] == 7
    assert snap["player"]["owned_cards"]["Silver"] == 1


def test_interactive_buy_to_end_of_turn(tmp_path):
    answers = [
        "2", "first", "", "y", "buy",         # players, kingdom, turn unknown, supply full, phase
        "Gold, Gold, Silver, Estate", "",     # hand, in play (none -> no coin questions)
        "",                                   # bought this turn
        "skip", "",                           # deck, opp VP unknown
        "",                                   # play all treasures
        "",                                   # buy recommended card
        "",                                   # end turn
        "n",                                  # don't continue to the next turn
    ]
    inp, _ = scripted(answers)
    printed = []
    code = interactive_main(DominionRuleEngine(), input_fn=inp, print_fn=printed.append)
    text = "\n".join(printed)
    assert code == 0
    assert "Bought Province" in text and "Turn finished" in text


def test_quick_start_turn_1_then_turn_2_with_saved_history(tmp_path):
    save = tmp_path / "game.json"
    answers = [
        "3", "first", "1",          # players, kingdom, turn 1 -> quick start
        "3",                        # Coppers in hand (rest Estates)
        "", "",                     # opponents' VP (default 3)
        "",                         # play all Treasures ($3)
        "",                         # buy the recommended card
        "",                         # end turn
        "",                         # continue to next turn
        "Silver", "Province",       # what Opponent 1 / Opponent 2 gained
        "",                         # you gained/lost nothing on their turns
        "",                         # turn-2 hand: accept default (4 Copper, 1 Estate)
        "q",
    ]
    inp, prompts = scripted(answers)
    printed = []
    code = interactive_main(DominionRuleEngine(), save=str(save), input_fn=inp, print_fn=printed.append)
    text = "\n".join(printed)
    assert code == 0
    assert "Phase: BUY" in text and "you will have $3" in text and "Buy Gold" not in text
    assert any("Copper x4, Estate" in p for p in prompts)  # turn-2 hand inferred from turn 1
    snap = json.loads(save.read_text())
    assert snap["turn_number"] == 2 and snap["player"]["hand"] == ["Copper"] * 4 + ["Estate"]
    bought = snap["history"][0]["bought"]
    assert snap["history"][0]["turn"] == 1 and len(bought) == 1
    assert snap["player"]["owned_cards"][bought[0]] >= 1
    assert snap["supply"]["Province"] == 11 and [o["vp"] for o in snap["opponents"]] == [3, 9]
    assert "Bought so far: T1 " + bought[0] in text


def test_load_saved_finished_turn_continues(tmp_path):
    engine = DominionRuleEngine()
    s = state_from_dict(load_example("opening_5_2"))
    for a in (PlayAllTreasures(), Buy("Silver"), EndTurn()):
        s = engine.apply(s, a).state
    path = tmp_path / "s.json"
    path.write_text(json.dumps(s.to_dict()))
    inp, _ = scripted(["y", "", "", "Silver, Copper x3, Estate", "q"])  # continue, opponent, you, hand
    printed = []
    assert interactive_main(engine, load=str(path), input_fn=inp, print_fn=printed.append) == 0
    assert "RECOMMENDED: Play all Treasures" in "\n".join(printed)


def test_multi_turn_example_continues_to_next_turn(tmp_path):
    path = tmp_path / "game.json"
    path.write_text(json.dumps(load_example("saved_game_turn_3")))
    answers = [
        "", "Copper, Copper",          # play Moat, record the 2 drawn cards
        "", "", "", "",                # end Action phase, play Treasures, buy, end turn
        "",                            # continue to turn 4
        "Militia", "",                 # Opponent 1 gained Militia, Opponent 2 nothing
        "",                            # you gained nothing on their turns
        "Village, Copper x4",          # not in your deck -> asked again
        "Silver, Copper x3, Estate",
        "q",
    ]
    inp, _ = scripted(answers)
    printed = []
    assert interactive_main(DominionRuleEngine(), load=str(path), save=str(path),
                            input_fn=inp, print_fn=printed.append) == 0
    assert any("doesn't have that many: Village" in p for p in printed)
    snap = json.loads(path.read_text())
    assert snap["turn_number"] == 4 and snap["supply"]["Militia"] == 8
    assert [h["turn"] for h in snap["history"]] == [1, 2, 3]


# A real save from a multi-turn interactive game (finished turn 4, with history).
REAL_SAVE = {
    "schema_version": 1, "num_players": 3, "turn_number": 4,
    "kingdom": ["Cellar", "Market", "Merchant", "Militia", "Mine", "Moat", "Remodel", "Smithy", "Village", "Workshop"],
    "supply": {"Copper": 38, "Silver": 35, "Gold": 30, "Estate": 12, "Duchy": 12, "Province": 12, "Curse": 20,
               "Cellar": 10, "Market": 7, "Merchant": 10, "Militia": 10, "Mine": 10, "Moat": 8, "Remodel": 10,
               "Smithy": 10, "Village": 10, "Workshop": 10},
    "trash": [],
    "player": {"name": "You", "phase": "turn_over", "hand": [], "in_play": [], "actions": 0, "buys": 0, "coins": 0,
               "bought_this_turn": ["Silver"], "merchants_played": 0, "silver_played": False, "pending_draws": 0,
               "unresolved": [], "owned_cards": {"Copper": 8, "Estate": 3, "Moat": 1, "Market": 2, "Silver": 1},
               "deck_count": None, "discard_count": None},
    "opponents": [{"name": "Opponent 1", "vp": 3, "owned_cards": None},
                  {"name": "Opponent 2", "vp": 3, "owned_cards": None}],
    "history": [
        {"turn": 1, "hand": ["Copper", "Copper", "Estate", "Estate", "Estate"], "bought": ["Moat"]},
        {"turn": 2, "hand": ["Copper", "Copper", "Copper", "Copper", "Copper"], "bought": ["Market"]},
        {"turn": 3, "hand": ["Copper", "Copper", "Copper", "Market"], "bought": ["Market", "Copper"]},
        {"turn": 4, "hand": ["Copper", "Copper", "Copper"], "bought": ["Silver"]},
    ],
}


def test_real_save_resumes_into_the_next_turn(tmp_path):
    path = tmp_path / "game.json"
    path.write_text(json.dumps(REAL_SAVE))
    answers = ["", "", "", "",                 # continue; Opponent 1 / 2 gained nothing; you gained nothing
               "Market, Silver, Copper x3",    # turn-5 hand
               "q"]
    inp, _ = scripted(answers)
    printed = []
    assert interactive_main(DominionRuleEngine(), load=str(path), save=str(path),
                            input_fn=inp, print_fn=printed.append) == 0
    text = "\n".join(printed)
    assert "Turn: 5" in text and "RECOMMENDED: Play Market" in text
    assert "Bought so far: T1 Moat; T2 Market; T3 Copper, Market; T4 Silver" in text
    assert json.loads(path.read_text())["turn_number"] == 5


def test_game_ending_turn_stops_instead_of_continuing(tmp_path):
    path = tmp_path / "end.json"
    path.write_text(json.dumps(load_example("endgame_winning_province")))
    inp, prompts = scripted(["", "", ""])  # play Treasures, buy the last Province, end turn
    printed = []
    assert interactive_main(DominionRuleEngine(), load=str(path), input_fn=inp, print_fn=printed.append) == 0
    assert any(line.startswith("GAME OVER") for line in printed)
    assert not any("Continue to your next turn" in p for p in prompts)


def test_interactive_reprompts_on_bad_card_name():
    answers = ["2", "first", "", "y", "action", "Smithee", "Smithy", "", "skip", "", "q"]
    inp, _ = scripted(answers)
    printed = []
    interactive_main(DominionRuleEngine(), input_fn=inp, print_fn=printed.append)
    assert any("Did you mean" in p for p in printed)


def test_followup_reprompts_on_bad_card_name_in_hand():
    engine = DominionRuleEngine()
    s = state_from_dict({"num_players": 2, "kingdom": OTHER_KINGDOM,
                         "player": {"phase": "action", "hand": ["Chapel", "Copper", "Estate"]}})
    s = engine.apply(s, PlayAction("Chapel")).state  # Chapel leaves a manual step
    inp, _ = scripted(["Coppr", "Copper", "", "0"])  # typo, then fixed hand, no pile changes, $0
    printed = []
    s = _resolve_followups(Prompter(inp, printed.append), engine, s)
    assert any("Did you mean" in p for p in printed)
    assert s.turn.hand == ["Copper"] and not s.turn.unresolved


def test_interactive_eof_quits_cleanly():
    def _eof(_):
        raise EOFError

    assert interactive_main(DominionRuleEngine(), input_fn=_eof, print_fn=lambda *_: None) == 0


def test_runtime_is_fully_offline(monkeypatch, capsys):
    """Every example must work with networking disabled, and `recommend` / `plan --json` print valid JSON."""
    import socket

    def _no_network(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    for name in example_names():
        for command in (["recommend", "-e", name, "--json"], ["plan", "-e", name, "--json", "--nodes", "3000"]):
            assert cardgame_main(["dominion", *command]) == 0, (name, command)
            json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------- lookahead planner: plan / compare

def test_plan_human_output_has_every_section(capsys):
    code, out, _ = run(capsys, "plan", "-e", "stranded_draw_vs_money")
    assert code == 0
    assert ">> RECOMMENDED: Play Militia" in out and "E[$] = 5.00" in out
    assert "Play Smithy" in out and "E[$] = 3.00" in out  # alternative
    for section in ("Method: exact enumeration", "Information level: known_composition", "Threshold probabilities:",
                    "Top future outcomes", "Principal line: Play Militia", "Why:", "Search:", "nodes expanded",
                    "budget exhausted: no", "Limitations:"):
        assert section in out, section


def test_plan_json_is_the_result_dict(capsys, tmp_path):
    code, out, _ = run(capsys, "plan", "-e", "stranded_draw_vs_money", "--json", "--seed", "7", "--depth", "6",
                       "--nodes", "5000", "--samples", "8", "--top", "2")
    data = json.loads(out)
    assert code == 0 and data["recommended"]["action_key"] == "Play Militia"
    assert data["config"] == {"objective": "money", "horizon": None, "max_depth": 6, "max_nodes": 5000,
                              "max_outcomes_per_chance": 200, "samples_per_chance": 8, "seed": 7, "top_outcomes": 2}
    # --state PATH is an alias of the positional STATE
    f = tmp_path / "s.json"
    f.write_text(json.dumps(load_example("stranded_draw_vs_money")))
    code, out2, _ = run(capsys, "plan", "--state", str(f), "--json", "--seed", "7", "--depth", "6",
                        "--nodes", "5000", "--samples", "8", "--top", "2")
    assert code == 0 and json.loads(out2) == data


def test_plan_lower_bounds_and_objectives(capsys):
    code, out, _ = run(capsys, "plan", "-e", "partial_knowledge")
    assert code == 0 and "E[$] >= 7.80" in out and "P($>=5) >= " in out and "partially_known" in out
    code, out, _ = run(capsys, "plan", "-e", "reshuffle_next_hand", "--horizon", "next_hand",
                       "--objective", "next_hand_money", "--json")
    data = json.loads(out)
    assert code == 0 and data["horizon"] == "next_hand" and data["objective"] == "next_hand_money"
    assert data["recommended"]["action_key"] == "Buy Gold"
    assert data["recommended"]["expected_value"] == pytest.approx(5.0)  # 3 cards from {Copper 3, Silver 2, Gold}
    code, out, _ = run(capsys, "plan", "-e", "small_deck_village_smithy", "--objective", "threshold:6")
    assert code == 0 and "P($>=6)" in out


def test_plan_monte_carlo_shows_stderr(capsys, tmp_path):
    # 17 different single cards in the draw pile: Smithy has C(17,3) = 680 > 200 outcomes -> sampled.
    snap = load_example("small_deck_village_smithy")
    kingdom = snap["kingdom"]
    snap["player"]["hand"] = ["Smithy", "Copper"]
    snap["player"].pop("owned_cards")
    snap["player"]["draw_pile"] = {"composition": {c: 1 for c in kingdom + ["Copper", "Silver", "Gold", "Estate",
                                                                            "Duchy", "Province", "Curse"]}}
    f = tmp_path / "mc.json"
    f.write_text(json.dumps(snap))
    code, out, _ = run(capsys, "plan", str(f), "--samples", "8", "--depth", "3")
    assert code == 0 and "Monte Carlo" in out and "±" in out


def test_compare_every_legal_action_and_selected(capsys):
    code, out, _ = run(capsys, "compare", "-e", "small_deck_village_smithy")
    assert code == 0 and "Side by side" in out
    for label, value in (("Play Village", "5.20"), ("Play Smithy", "4.40"), ("End Action phase", "2.00")):
        assert label in out and value in out
    code, out, _ = run(capsys, "compare", "-e", "small_deck_village_smithy", "--json",
                       "--actions", "play smithy", "PLAY VILLAGE")
    data = json.loads(out)
    assert code == 0 and [c["action_key"] for c in data["candidates"]] == ["Play Village", "Play Smithy"]
    assert data["candidates"][0]["expected_value"] == pytest.approx(26 / 5)


def test_compare_unknown_label_lists_legal_actions(capsys):
    code, _, err = run(capsys, "compare", "-e", "small_deck_village_smithy", "--actions", "Play Gold")
    assert code == 2 and "Play Gold" in err and "'Play Village'" in err and "'Play Smithy'" in err


def test_plan_errors_reuse_exit_codes(tmp_path, capsys):
    code, _, err = run(capsys, "plan", "-e", "opening_5_2", "--objective", "greed")
    assert code == 2 and "Unknown objective" in err
    code, _, err = run(capsys, "plan", "-e", "opening_5_2", "--nodes", "0")
    assert code == 2 and "max_nodes" in err
    code, _, err = run(capsys, "plan")
    assert code == 2 and "--example" in err
    snap = {"num_players": 2, "kingdom": ["Cellar", "Market", "Merchant", "Militia", "Mine", "Moat",
                                          "Remodel", "Smithy", "Village", "Workshop"],
            "player": {"phase": "action", "hand": [], "in_play": ["Gold"]}}
    f = tmp_path / "s.json"
    f.write_text(json.dumps(snap))
    code, out, _ = run(capsys, "compare", str(f))
    assert code == 1 and "Treasures are in play" in out
    code, out, _ = run(capsys, "plan", str(f), "--json")
    assert code == 1 and json.loads(out)["valid"] is False
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"num_players": 9, "kingdom": [], "player": {}}))
    code, _, err = run(capsys, "plan", str(bad))
    assert code == 1 and "num_players" in err


def test_python_m_entry_point_runs_plan():
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[2] / "src")
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run([sys.executable, "-m", "card_game_engine", "dominion", "plan", "-e",
                           "stranded_draw_vs_money", "--json"], capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["recommended"]["action_key"] == "Play Militia"


def _turn_one_snapshot(tmp_path, **player):
    snap = {"num_players": 2, "kingdom": ["Cellar", "Market", "Merchant", "Militia", "Mine", "Moat",
                                          "Remodel", "Smithy", "Village", "Workshop"],
            "player": {"phase": "buy", "owned_cards": {"Copper": 7, "Estate": 3}, "discard_count": 0, **player}}
    f = tmp_path / "turn1.json"
    f.write_text(json.dumps(snap))
    return str(f)


def test_plan_horizon_hands_and_future_money(capsys, tmp_path):
    f = _turn_one_snapshot(tmp_path, hand=["Estate"], in_play=["Copper"] * 4, coins=4)
    code, out, _ = run(capsys, "compare", f, "--horizon", "hands:2", "--objective", "future_money", "--json",
                       "--actions", "Buy Silver", "End turn (Clean-up)")
    data = json.loads(out)
    assert code == 0 and data["horizon"] == "hands:2" and data["objective"] == "future_money"
    silver = data["candidates"][0]
    assert silver["action_key"] == "Buy Silver" and silver["expected_value"] == pytest.approx(78 / 11)
    assert [h["expected_coins"] for h in silver["projected_hands"]] == [pytest.approx(3.0), pytest.approx(45 / 11)]
    code, out, _ = run(capsys, "plan", f, "--horizon", "hands:2", "--objective", "future_money")
    assert code == 0 and ">> RECOMMENDED: Buy Silver" in out and "Horizon: hands:2" in out
    assert "Projected hands" in out and "hand 2: E[$] = 4.09, P($>=5) = 0.35" in out
    code, _, err = run(capsys, "plan", f, "--horizon", "hands:9")
    assert code == 2 and "hands:<N>" in err


def test_plan_and_compare_print_tie(capsys, tmp_path):
    f = _turn_one_snapshot(tmp_path, hand=["Copper"] * 4 + ["Estate"])
    code, out, _ = run(capsys, "plan", f, "--horizon", "next_hand", "--objective", "next_hand_money")
    assert code == 0 and ">> TIE under objective next_hand_money" in out
    assert "NOT better" in out and ">> RECOMMENDED" not in out
    code, out, _ = run(capsys, "plan", f, "--horizon", "next_hand", "--objective", "next_hand_money", "--json")
    data = json.loads(out)
    assert data["tie"] is True and data["recommended"] is not None and len(data["tied_actions"]) >= 2
    code, out, _ = run(capsys, "compare", f, "--objective", "next_hand_money")
    assert code == 0 and ">> TIE under objective next_hand_money" in out
    # compare-all collapses single Play-Treasure actions into Play all Treasures
    code, out, _ = run(capsys, "compare", f, "--json")
    keys = [c["action_key"] for c in json.loads(out)["candidates"]]
    assert "Play all Treasures" in keys and "Play Copper" not in keys
    code, out, _ = run(capsys, "compare", f, "--json", "--actions", "Play Copper")
    assert code == 0 and [c["action_key"] for c in json.loads(out)["candidates"]] == ["Play Copper"]

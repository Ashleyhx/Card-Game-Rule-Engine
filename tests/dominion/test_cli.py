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
    """Every example must work with networking disabled."""
    import socket

    def _no_network(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    for name in example_names():
        assert cardgame_main(["dominion", "recommend", "-e", name, "--json"]) == 0
    capsys.readouterr()

"""Game-agnostic core: registry, interface conformance, top-level CLI."""

import json

import pytest

from card_game_engine.cli import main
from card_game_engine.core import GameAction, Policy, RuleEngine, Transition, available_games, get_game
from card_game_engine.games.dominion.cli.commands import load_example


def test_registry_lists_dominion():
    assert "dominion" in available_games()
    game = get_game("Dominion")
    assert game.id == "dominion" and "Second Edition" in game.ruleset


def test_unknown_game():
    with pytest.raises(KeyError, match="Available"):
        get_game("poker")


def test_dominion_implements_core_interfaces():
    game = get_game("dominion")
    engine = game.create_engine()
    policy = game.create_policy(engine)
    assert isinstance(engine, RuleEngine)
    assert isinstance(policy, Policy)
    state = game.parse_state(load_example("opening_5_2"))
    actions = engine.legal_actions(state)
    assert actions and all(isinstance(a, GameAction) for a in actions)
    assert isinstance(engine.apply(state, actions[0]), Transition)


def test_game_agnostic_loop_uses_only_core_contracts():
    """What a future agent/planner does: ask the policy, apply via the engine, until blocked or done."""
    game = get_game("dominion")
    engine = game.create_engine()
    policy = game.create_policy(engine)
    state = game.parse_state(load_example("endgame_winning_province"))
    taken = []
    for _ in range(10):
        if engine.blocking_issues(state) or not engine.legal_actions(state):
            break
        best = policy.recommend(state).ranked[0].action
        assert best in engine.legal_actions(state)
        taken.append(best.describe())
        state = engine.apply(state, best).state
    assert taken == ["Play all Treasures", "Buy Province", "End turn (Clean-up)"]


def test_cli_games(capsys):
    assert main(["games"]) == 0
    assert "dominion" in capsys.readouterr().out
    assert main(["games", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == "dominion"


def test_cli_requires_game(capsys):
    with pytest.raises(SystemExit):
        main([])
    capsys.readouterr()

"""Game-agnostic session entry point (core.initialize_session)."""

import dataclasses

import pytest

from card_game_engine.core import GameDefinition, get_game, initialize_session
from card_game_engine.core import session as core_session


def test_initialize_dominion_session():
    s = initialize_session("dominion")
    assert s.domain == "dominion" and s.events == ()
    from card_game_engine.games.dominion.inference import infer_belief_state

    b = infer_belief_state(s)
    assert b.pending_draws == 5 and b.owned == {"Copper": 7, "Estate": 3}
    s3 = initialize_session("Dominion", {"num_players": 3})
    assert s3.config["num_players"] == 3 and infer_belief_state(s3).game.num_players == 3


def test_unknown_domain():
    with pytest.raises(KeyError, match="Available"):
        initialize_session("poker")


def test_game_without_session_support(monkeypatch):
    plain = dataclasses.replace(get_game("dominion"), initialize_session=None)
    monkeypatch.setattr(core_session, "get_game", lambda domain: plain)
    with pytest.raises(NotImplementedError, match="does not support"):
        initialize_session("dominion")


def test_game_definition_field_is_optional():
    g = get_game("dominion")
    legacy = GameDefinition(g.id, g.name, g.ruleset, g.create_engine, g.create_policy, g.parse_state, g.register_cli)
    assert legacy.initialize_session is None and g.initialize_session is not None


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="unknown session config"):
        initialize_session("dominion", {"seed": 1})

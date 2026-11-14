"""Dominion Second Edition (Base Set) game plugin.

* ``rules``  - verified card data + deterministic rule engine (no strategy)
* ``policy`` - heuristic decision policy that ranks the engine's legal actions
* ``inference`` - observation-tracking session: belief over hidden zones, cross-turn inference
* ``cli``    - ``cardgame dominion ...`` commands and the interactive assistant
"""

from __future__ import annotations

from ...core.interfaces import GameDefinition


def _create_engine():
    from .rules import DominionRuleEngine

    return DominionRuleEngine()


def _create_policy(engine):
    from .policy import HeuristicPolicy

    return HeuristicPolicy(engine)


def _parse_state(data: dict):
    from .rules import state_from_dict

    return state_from_dict(data)


def _register_cli(parser):
    from .cli.commands import register_cli

    register_cli(parser)


def _initialize_session(config: dict):
    from .inference import initialize_session

    return initialize_session(config)


GAME = GameDefinition(
    id="dominion",
    name="Dominion",
    ruleset="Second Edition, Base Set (2-4 players)",
    create_engine=_create_engine,
    create_policy=_create_policy,
    parse_state=_parse_state,
    register_cli=_register_cli,
    description="Deck-building card game by Donald X. Vaccarino (Rio Grande Games). Rules researched 2026-11-13.",
    initialize_session=_initialize_session,
)

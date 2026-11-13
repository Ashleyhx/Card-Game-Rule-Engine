"""Registry of available game plugins.

To add a game: create ``card_game_engine/games/<id>/`` exposing a ``GAME``
:class:`GameDefinition`, then add its module path to ``_GAME_MODULES``.
"""

from __future__ import annotations

import importlib

from .interfaces import GameDefinition

_GAME_MODULES: dict[str, str] = {
    "dominion": "card_game_engine.games.dominion",
}


def available_games() -> list[str]:
    return sorted(_GAME_MODULES)


def get_game(game_id: str) -> GameDefinition:
    key = game_id.strip().lower()
    if key not in _GAME_MODULES:
        raise KeyError(f"Unknown game '{game_id}'. Available: {', '.join(available_games())}")
    game = importlib.import_module(_GAME_MODULES[key]).GAME
    if game.id != key:
        raise RuntimeError(f"Game module for '{key}' declares id '{game.id}'.")
    return game


def all_games() -> list[GameDefinition]:
    return [get_game(g) for g in available_games()]

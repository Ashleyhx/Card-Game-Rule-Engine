"""Game-agnostic entry point for observation-tracking sessions."""

from __future__ import annotations

from typing import Any

from .registry import get_game


def initialize_session(domain: str, config: dict | None = None) -> Any:
    """Start a tracking session for ``domain`` using that game's own starting setup.

    Raises ``KeyError`` for an unknown game and ``NotImplementedError`` if the game has no session support.
    """
    game = get_game(domain)
    if game.initialize_session is None:
        raise NotImplementedError(f"Game '{game.id}' does not support tracking sessions.")
    return game.initialize_session(dict(config or {}))

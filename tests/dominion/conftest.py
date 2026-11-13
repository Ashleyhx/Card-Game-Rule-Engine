import pytest

from card_game_engine.games.dominion.rules import DominionRuleEngine, state_from_dict

FIRST_GAME = ["Cellar", "Market", "Merchant", "Militia", "Mine", "Moat", "Remodel", "Smithy", "Village", "Workshop"]
OTHER_KINGDOM = ["Chapel", "Festival", "Gardens", "Laboratory", "Library", "Moneylender",
                 "Poacher", "Throne Room", "Witch", "Bandit"]


@pytest.fixture(scope="session")
def engine():
    return DominionRuleEngine()


@pytest.fixture
def make_state():
    """Build a GameState from keyword overrides of a minimal snapshot."""

    def _make(kingdom=None, num_players=2, supply=None, opponents=None, **player):
        snap = {
            "schema_version": 1,
            "num_players": num_players,
            "kingdom": kingdom or FIRST_GAME,
            "player": {"phase": "action", "hand": [], **player},
        }
        if supply is not None:
            snap["supply"] = supply
        if opponents is not None:
            snap["opponents"] = opponents
        return state_from_dict(snap)

    return _make

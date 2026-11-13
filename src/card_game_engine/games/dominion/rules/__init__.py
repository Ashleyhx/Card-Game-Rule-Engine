"""Rule Engine: verified card data, typed state, legal actions, transitions.

This package has no dependency on any policy; future planners/agents can use it directly.
"""

from .actions import (
    Action,
    Buy,
    EndActionPhase,
    EndTurn,
    PlayAction,
    PlayAllTreasures,
    PlayTreasure,
    action_from_dict,
    action_to_dict,
)
from .cards import CardDef, CardRegistry, UnknownCardError, load_registry
from .engine import GameEndStatus, IllegalActionError, DominionRuleEngine, Transition
from .ruleset import Ruleset, load_ruleset
from .state import GameState, OpponentInfo, Phase, SnapshotError, TurnState, new_game_turn_state, state_from_dict
from .validation import Issue, has_errors, validate_state
from .zones import (
    UNKNOWN_CARD,
    DiscardKnowledge,
    DrawPileKnowledge,
    InformationLevel,
    add_to_discard,
    cleanup_zones,
    derive_zone_knowledge,
    put_on_top,
    reshuffle,
)

__all__ = [
    "Action", "Buy", "EndActionPhase", "EndTurn", "PlayAction", "PlayAllTreasures", "PlayTreasure",
    "action_from_dict", "action_to_dict",
    "CardDef", "CardRegistry", "UnknownCardError", "load_registry",
    "GameEndStatus", "IllegalActionError", "DominionRuleEngine", "Transition",
    "Ruleset", "load_ruleset",
    "GameState", "OpponentInfo", "Phase", "SnapshotError", "TurnState", "new_game_turn_state", "state_from_dict",
    "Issue", "has_errors", "validate_state",
    "UNKNOWN_CARD", "DiscardKnowledge", "DrawPileKnowledge", "InformationLevel", "add_to_discard",
    "cleanup_zones", "derive_zone_knowledge", "put_on_top", "reshuffle",
]

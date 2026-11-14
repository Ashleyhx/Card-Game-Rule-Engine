"""History-aware state inference for one real Dominion game: events, belief worlds, session API.

Depends on ``rules`` and ``planning``; neither imports this package.
"""

from .belief import BeliefState, Contradiction, World, collapse, predictive
from .events import (
    EVENT_TYPES,
    EndTurnObservation,
    Event,
    IllegalEvent,
    InconsistentObservation,
    ObserveHand,
    RecordDiscard,
    RecordDraw,
    RecordGain,
    RecordPlay,
    RecordPurchase,
    RecordSupply,
    RecordTrash,
    SessionError,
    event_from_dict,
)
from .session import (
    HandNotDetermined,
    HandPrediction,
    Session,
    correct_event,
    infer_belief_state,
    initialize_session,
    plan_next_action,
    predict_next_hand,
    record_event,
    record_observation,
    remove_event,
    reset_session,
    to_game_state,
)

__all__ = [
    "BeliefState", "Contradiction", "World", "collapse", "predictive",
    "EVENT_TYPES", "EndTurnObservation", "Event", "IllegalEvent", "InconsistentObservation", "ObserveHand",
    "RecordDiscard", "RecordDraw", "RecordGain", "RecordPlay", "RecordPurchase", "RecordSupply", "RecordTrash",
    "SessionError", "event_from_dict",
    "HandNotDetermined", "HandPrediction", "Session", "correct_event", "infer_belief_state", "initialize_session",
    "plan_next_action", "predict_next_hand", "record_event", "record_observation", "remove_event", "reset_session",
    "to_game_state",
]

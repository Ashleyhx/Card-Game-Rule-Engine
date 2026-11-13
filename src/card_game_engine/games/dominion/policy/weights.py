"""Strategy weights used by the Dominion heuristic policy.

These are *opinions*, not rules. They are hand-tuned approximations informed by
widely shared community strategy (Big Money buy rules and the Penultimate Province
Rule from DominionStrategy.com - see docs/RULES_RESEARCH.md). Every number here is
deliberately exposed so it can be inspected, tuned, or replaced by a future
search/simulation-based planner without touching the rule engine.
"""

from __future__ import annotations

# Rough standalone strength of each Kingdom card in a typical Base-set game, 0-10.
KINGDOM_RATING: dict[str, float] = {
    "Witch": 9.0,
    "Laboratory": 7.0,
    "Militia": 6.5,
    "Smithy": 6.5,
    "Sentry": 6.0,
    "Library": 6.0,
    "Council Room": 6.0,
    "Market": 6.0,
    "Artisan": 6.0,
    "Chapel": 6.0,
    "Festival": 5.5,
    "Bandit": 5.0,
    "Moneylender": 5.0,
    "Poacher": 5.0,
    "Remodel": 5.0,
    "Throne Room": 5.0,
    "Mine": 4.5,
    "Village": 4.5,
    "Merchant": 4.5,
    "Vassal": 4.5,
    "Harbinger": 4.0,
    "Moat": 4.0,
    "Cellar": 3.5,
    "Workshop": 3.5,
    "Bureaucrat": 3.5,
    "Gardens": 3.0,
}

# Buy value of basic cards by game stage (same scale as kingdom ratings).
# Stages: early -> mid -> dance (Duchy dancing) -> late -> endgame.
BASIC_BUY_VALUE: dict[str, dict[str, float]] = {
    "Province": {"early": 6.0, "mid": 10.0, "dance": 11.0, "late": 12.0, "endgame": 14.0},
    "Gold":     {"early": 9.0, "mid": 8.5,  "dance": 7.0,  "late": 4.5,  "endgame": 1.0},
    "Duchy":    {"early": 0.5, "mid": 2.0,  "dance": 5.5,  "late": 6.5,  "endgame": 8.0},
    "Silver":   {"early": 5.0, "mid": 4.0,  "dance": 3.0,  "late": 2.0,  "endgame": 0.5},
    "Estate":   {"early": -3.0, "mid": -2.0, "dance": -1.0, "late": 0.5, "endgame": 3.0},
    "Copper":   {"early": -1.5, "mid": -1.5, "dance": -1.0, "late": -0.5, "endgame": -0.5},
    "Curse":    {"early": -12.0, "mid": -12.0, "dance": -12.0, "late": -12.0, "endgame": -12.0},
}

# Kingdom card value multiplier by stage (actions lose value as the game ends).
KINGDOM_STAGE_MULTIPLIER = {"early": 1.0, "mid": 0.85, "dance": 0.55, "late": 0.35, "endgame": 0.1}

# Action-phase play values.
VALUE_PER_CARD_DRAWN = 1.5
VALUE_PER_COIN = 1.0
VALUE_PER_BUY = 0.6
VALUE_PER_USEFUL_ACTION = 2.0
VALUE_PER_SPARE_ACTION = 0.3
SEQUENCING_BONUS = 3.0  # play non-terminals before terminals
ATTACK_VALUE = {"Witch": 3.0, "Militia": 1.5, "Bandit": 1.0, "Bureaucrat": 0.5}

# Treasure plays are never harmful in the Base set; keep them on top.
PLAY_TREASURES_SCORE = 50.0

# Endgame adjustments when VP totals are known.
WINNING_MOVE_BONUS = 100.0
LOSING_MOVE_PENALTY = -100.0
PENULTIMATE_PROVINCE_PENALTY = -8.0

# Confidence thresholds on the score gap between the top two choices.
HIGH_CONFIDENCE_GAP = 3.0
MEDIUM_CONFIDENCE_GAP = 1.0

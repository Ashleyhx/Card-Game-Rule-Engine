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
    "Moat": 3.0,  # +2 Cards terminal: notably weaker than Smithy; its defence is scored separately
    "Cellar": 3.5,
    "Workshop": 3.5,
    "Bureaucrat": 3.5,
    "Gardens": 3.0,
}

# Buy value of basic cards by game stage (same scale as kingdom ratings).
# Stages: early -> mid -> dance (Duchy dancing) -> late -> endgame.
# With a known deck, the money a card adds is scored by the next-shuffle odds term below, so Silver
# and Gold keep only the rest here (previous all-in value minus TYPICAL_MONEY_VALUE: no double count).
BASIC_BUY_VALUE: dict[str, dict[str, float]] = {
    "Province": {"early": 6.0, "mid": 10.0, "dance": 11.0, "late": 12.0, "endgame": 14.0},
    "Gold":     {"early": 5.0, "mid": 4.5,  "dance": 6.0,  "late": 4.0,  "endgame": 1.0},
    "Duchy":    {"early": 0.5, "mid": 2.0,  "dance": 5.5,  "late": 6.5,  "endgame": 8.0},
    "Silver":   {"early": 3.5, "mid": 3.5,  "dance": 2.5,  "late": 1.8,  "endgame": 0.5},
    "Estate":   {"early": -3.0, "mid": -2.0, "dance": -1.0, "late": 0.5, "endgame": 3.0},
    "Copper":   {"early": -1.5, "mid": -1.5, "dance": -1.0, "late": -0.5, "endgame": -0.5},
    "Curse":    {"early": -12.0, "mid": -12.0, "dance": -12.0, "late": -12.0, "endgame": -12.0},
}
# Where no odds are computed (deck unknown; the planner's static `purchase` objective) a typical
# money value is added instead, so Silver/Gold keep their previous all-in values there.
TYPICAL_MONEY_VALUE: dict[str, dict[str, float]] = {
    "Gold":   {"early": 4.0, "mid": 4.0, "dance": 1.0, "late": 0.5, "endgame": 0.0},
    "Silver": {"early": 1.5, "mid": 0.5, "dance": 0.5, "late": 0.2, "endgame": 0.0},
}


def typical_basic_value(card: str, stage: str) -> float:
    """Value of a basic card without deck-specific odds: BASIC_BUY_VALUE + TYPICAL_MONEY_VALUE."""
    return BASIC_BUY_VALUE[card][stage] + TYPICAL_MONEY_VALUE.get(card, {}).get(stage, 0.0)


# Next-shuffle odds term (policy/odds.py): NEXT_SHUFFLE_WEIGHT x sum over the stage's price points of
# weight x (P(5-card hand reaches $N | buy the card) - P(... | buy nothing)). Price points follow the
# Big Money buy rules (DominionStrategy "Money strategies": Province $8, Gold $6-7, Silver $3-5):
# early you want $5 hands ($5 Actions) and some $6 (Gold), mid $6/$8, later mostly $8.
NEXT_SHUFFLE_WEIGHT = 10.0
NEXT_SHUFFLE_PRICE_WEIGHTS: dict[str, dict[int, float]] = {
    "early": {5: 1.0, 6: 0.5},
    "mid": {6: 1.0, 8: 1.0},
    "dance": {6: 0.25, 8: 1.0},
    "late": {8: 0.75},
    "endgame": {8: 0.25},  # few shuffles left: deck quality barely matters
}
# Kingdom ratings already include a typical economic contribution, so Kingdom cards get only this
# share of the odds term (enough to separate deck fit: a second terminal, draw without money...).
KINGDOM_NEXT_SHUFFLE_SHARE = 0.5

# Moat's defence, per attack in the kingdom (the strongest one counts), x (1 + 0.5 per extra opponent).
# Junking attacks matter most (DominionStrategy "Moat"); Militia alone is a small reason to buy Moat.
MOAT_DEFENCE = {"Witch": 2.0, "Bandit": 0.8, "Bureaucrat": 0.6, "Militia": 0.4}
MOAT_DEFENCE_PER_EXTRA_OPPONENT = 0.5

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

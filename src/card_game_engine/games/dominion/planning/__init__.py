"""Probabilistic lookahead planning for Dominion (built on the rule engine; no card effects here)."""

from .draws import BOUNDARY_UNKNOWN_DRAW_PILE, DrawOutcome, count_draw_outcomes, enumerate_draws, sample_draw
from .evaluation import (
    METRIC_PROVENANCE,
    OBJECTIVE_NAMES,
    FutureMoneyObjective,
    MoneyObjective,
    NextHandMoneyObjective,
    Objective,
    PurchaseObjective,
    ThresholdObjective,
    get_objective,
    leaf_metrics,
    value_is_lower_bound,
)
from .planner import (
    MAX_PROJECTED_HANDS,
    TIE_TOLERANCE,
    CandidateEvaluation,
    ComparisonResult,
    DominionPlanningModel,
    OutcomeSummary,
    PlanConfig,
    PlanNode,
    PlanResult,
    compare_actions,
    evaluate_action_sequences,
    first_action_candidates,
    plan_next_action,
    projected_hand_count,
)
from .transitions import (
    LIBRARY_STEP,
    OPPONENT_ONLY_STEPS,
    PlanState,
    card_totals,
    from_game_state,
    pending_draws,
    simulate_action,
    simulate_draw,
    simulate_end_turn,
    simulate_projected_cleanup,
    simulate_purchase,
    simulate_treasure,
)

__all__ = [
    "BOUNDARY_UNKNOWN_DRAW_PILE", "DrawOutcome", "count_draw_outcomes", "enumerate_draws", "sample_draw",
    "LIBRARY_STEP", "OPPONENT_ONLY_STEPS", "PlanState", "card_totals", "from_game_state", "pending_draws",
    "simulate_action", "simulate_draw", "simulate_end_turn", "simulate_projected_cleanup", "simulate_purchase",
    "simulate_treasure", "METRIC_PROVENANCE", "OBJECTIVE_NAMES", "FutureMoneyObjective", "MoneyObjective",
    "NextHandMoneyObjective", "Objective", "PurchaseObjective", "ThresholdObjective", "get_objective", "leaf_metrics",
    "value_is_lower_bound", "MAX_PROJECTED_HANDS", "TIE_TOLERANCE", "CandidateEvaluation",
    "ComparisonResult", "DominionPlanningModel", "OutcomeSummary", "PlanConfig", "PlanNode", "PlanResult",
    "compare_actions", "evaluate_action_sequences", "first_action_candidates", "plan_next_action",
    "projected_hand_count",
]

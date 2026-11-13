"""Dominion heuristic policy: ranks the Rule Engine's legal actions with transparent heuristics.

The policy never decides legality itself - it only scores what the engine says is
legal, so rules and strategy stay separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..rules import (
    Action, EndTurn, GameState, PlayAllTreasures, PlayTreasure, DominionRuleEngine, action_to_dict,
)
from . import weights as W
from .context import GameContext, build_context
from .scoring import HeuristicScorer

ALWAYS_UNCERTAIN = [
    "Scores are heuristic estimates (no search or simulation); treat close calls as judgement calls.",
    "Hidden information is not modelled: your draw order and opponents' hands are unknown to the policy.",
]


@dataclass
class RankedAction:
    rank: int
    action: Action
    score: float
    reasons: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "action": action_to_dict(self.action),
            "score": round(self.score, 2),
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


@dataclass
class Recommendation:
    phase: str
    stage: str
    stage_reason: str
    ranked: list[RankedAction]
    plan: list[str]
    confidence: str
    confidence_reason: str
    uncertainties: list[str]
    blocking: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    # When the best move is "Play all Treasures", the ranking of buys afterwards (deterministic).
    projected_coins: int | None = None
    projected: list[RankedAction] = field(default_factory=list)

    @property
    def best(self) -> RankedAction | None:
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "stage": self.stage,
            "stage_reason": self.stage_reason,
            "recommendation": self.best.to_dict() if self.best else None,
            "alternatives": [r.to_dict() for r in self.ranked[1:]],
            "plan": self.plan,
            "confidence": {"level": self.confidence, "reason": self.confidence_reason},
            "uncertainties": self.uncertainties,
            "blocking": self.blocking,
            "context": self.context,
            "after_playing_treasures": (
                {"coins": self.projected_coins, "ranked": [r.to_dict() for r in self.projected]}
                if self.projected else None
            ),
        }


class HeuristicPolicy:
    def __init__(self, engine: DominionRuleEngine | None = None):
        self.engine = engine or DominionRuleEngine()
        self.scorer = HeuristicScorer(self.engine)

    def rank(self, state: GameState, ctx: GameContext | None = None) -> list[RankedAction]:
        ctx = ctx or build_context(state, self.engine)
        legal = self.engine.legal_actions(state)
        if any(isinstance(a, PlayAllTreasures) for a in legal):
            legal = [a for a in legal if not isinstance(a, PlayTreasure)]  # equivalent, avoid clutter
        scored = []
        for a in legal:
            sc = self.scorer.score(state, ctx, a)
            scored.append((sc.total, a, sc))
        # Stable, deterministic ordering: score desc, then label.
        scored.sort(key=lambda x: (-x[0], x[1].describe()))
        return [
            RankedAction(rank=i + 1, action=a, score=total, reasons=sc.reasons, warnings=sc.warnings)
            for i, (total, a, sc) in enumerate(scored)
        ]

    def plan(self, state: GameState, max_steps: int = 12) -> list[str]:
        """Greedy plan for the rest of the current phase, stopping at hidden information."""
        steps: list[str] = []
        current = state
        for _ in range(max_steps):
            blocking = self.engine.blocking_issues(current)
            if blocking:
                steps.append("Then: " + blocking[0] + " Re-run the recommendation with your new hand.")
                break
            ranked = self.rank(current)
            if not ranked:
                break
            choice = ranked[0].action
            steps.append(choice.describe())
            if isinstance(choice, EndTurn):
                break
            current = self.engine.apply(current, choice).state
        return steps

    def recommend(self, state: GameState, top_n: int | None = None) -> Recommendation:
        ctx = build_context(state, self.engine)
        blocking = self.engine.blocking_issues(state)
        ranked = [] if blocking else self.rank(state, ctx)
        if top_n is not None:
            ranked = ranked[:top_n]

        uncertainties = list(ALWAYS_UNCERTAIN) + ctx.unknowns + [
            "Assumption: " + a for a in state.assumptions if "owned_cards" not in a  # already in ctx.unknowns
        ]
        projected: list[RankedAction] = []
        projected_coins = None
        if ranked and isinstance(ranked[0].action, PlayAllTreasures):
            after = self.engine.apply(state, ranked[0].action).state
            projected_coins = after.turn.coins
            projected = self.rank(after)
            if top_n is not None:
                projected = projected[:top_n]
            confidence, why = self._confidence(projected, ctx)
            why = "Play Treasures first (no downside). For the buy: " + why
        else:
            confidence, why = self._confidence(ranked, ctx)
        plan = [] if blocking else self.plan(state)
        context = {
            "provinces_left": ctx.provinces_left,
            "provinces_start": ctx.provinces_start,
            "empty_piles": ctx.empty_piles,
            "low_piles": ctx.low_piles,
            "your_vp": ctx.my_vp,
            "opponent_vp": ctx.opponent_vps,
            "deck_size": ctx.deck.total if ctx.deck else None,
            "actions": state.turn.actions,
            "buys": state.turn.buys,
            "coins": state.turn.coins,
        }
        return Recommendation(
            phase=state.turn.phase.value,
            stage=ctx.stage,
            stage_reason=ctx.stage_reason,
            ranked=ranked,
            plan=plan,
            confidence=confidence,
            confidence_reason=why,
            uncertainties=uncertainties,
            blocking=blocking,
            context=context,
            projected_coins=projected_coins,
            projected=projected,
        )

    @staticmethod
    def _confidence(ranked: list[RankedAction], ctx: GameContext) -> tuple[str, str]:
        if not ranked:
            return "n/a", "No legal actions to rank."
        if len(ranked) == 1:
            return "high", "Only one legal action."
        top, second = ranked[0], ranked[1]
        if isinstance(top.action, PlayAllTreasures):
            return "high", "Playing Treasures before buying has no downside in the Base set."
        gap = top.score - second.score
        if gap >= W.HIGH_CONFIDENCE_GAP:
            level = "high"
        elif gap >= W.MEDIUM_CONFIDENCE_GAP:
            level = "medium"
        else:
            level = "low"
        reason = f"Score gap to the next option ({second.action.describe()}) is {gap:.1f}."
        downgrade = (ctx.deck is None) or (not ctx.vp_known and ctx.stage in ("late", "endgame"))
        if downgrade and level != "low" and gap < 50:
            level = {"high": "medium", "medium": "low"}[level]
            reason += " Lowered because key information (deck or scores) is missing."
        if top.warnings:
            reason += " See warnings on the top choice."
        return level, reason

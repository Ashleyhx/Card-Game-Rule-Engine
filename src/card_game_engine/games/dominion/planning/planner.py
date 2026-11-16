"""Probabilistic lookahead planner for one Dominion turn (optionally projecting future hands).

:class:`DominionPlanningModel` exposes planning states to the game-agnostic
``core.search.expectimax``:

* decisions are exactly ``engine.legal_actions`` with two documented, value-preserving
  reductions in the Buy phase: while Treasures are in hand and nothing has been bought,
  Buy actions are not expanded before Play-all-Treasures (you cannot play Treasures
  after buying, and playing them never hurts), and individual Play-Treasure actions are
  collapsed into Play-all-Treasures;
* a state with pending draws is a chance node over the exact draw outcomes
  (``enumerate_draws``), or sampled (``sample_draw``) when there are too many;
* leaves: the turn ended (horizon ``turn``), the next hand was drawn (horizon
  ``next_hand`` = ``hands:1``) or the N-th projected hand was drawn (horizon ``hands:N``),
  a planning boundary (manual choice / unknown draw pile), or no legal actions. Depth and
  node budgets cut the search off earlier; such leaves are flagged.
* ``hands:N``: after the current turn, N future start-of-turn hands are projected with no
  decisions in future turns: each projected hand is a chance node (exact grouped draw
  enumeration, or sampling), then goes to the discard pile at its Clean-up
  (``simulate_projected_cleanup``) before the next one is drawn, reshuffling only when the
  rules require it. Per projected hand, E[$] and P(hand >= $5) are reported (plus P(hand has >= K
  target cards) with the ``target:<T>>=<K>`` objective).

For each root candidate a second pass follows the planner's own decisions (argmax with
the same tie-breaking as the search) and records the distribution of leaf outcomes, so
outcome probabilities and threshold probabilities are consistent with the expected
value. The pass is exact when every chance node it meets is enumerated, otherwise it is
a Monte Carlo estimate (labelled ``monte_carlo``); it has its own node budget and is
skipped (with a limitation note) if that runs out.

Ties: when the two best candidates have equal values under the objective (within
``TIE_TOLERANCE``) the result says so (``tie`` / ``tied_actions``); the first listed action
is then only the documented tie-break, not a better action.

Honesty labels: when a candidate can reach unknown cards ("?" in hand, or an
``unknown_draw_pile`` boundary) its value and threshold probabilities are labelled
``lower_bound`` (unknown cards count $0), and when it can reach a ``manual:<card>``
boundary (a player sub-choice that is not modelled) they are labelled ``partial``. A
comparison against such a value is reported as not conclusive.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field, fields, replace
from fractions import Fraction
from typing import Any, Hashable

from ....core.search import ActionValue, SearchConfig, SearchStats, expectimax
from ..rules.actions import Action, Buy, EndTurn, PlayAllTreasures, PlayTreasure, action_from_dict, action_to_dict
from ..rules.engine import DominionRuleEngine
from ..rules.state import MAX_PROJECTED_HANDS, GameState, projected_hand_count
from ..rules.zones import derive_zone_knowledge
from .draws import DrawOutcome, count_draw_outcomes, enumerate_draws, sample_draw
from .evaluation import (
    METRIC_PROVENANCE, TIE_BREAK_WEIGHT, FutureMoneyObjective, MoneyObjective, NextHandMoneyObjective, Objective,
    TargetObjective, get_objective, leaf_label, leaf_metrics, tie_break, value_is_lower_bound,
)
from .transitions import (
    PlanState, from_game_state, pending_draws, simulate_action, simulate_draw, simulate_end_turn,
    simulate_projected_cleanup,
)

THRESHOLDS = (5, 6, 8)
HAND_THRESHOLD = 5            # per projected hand: P(hand >= $5)
TIE_TOLERANCE = 1e-9
FUTURE_OBJECTIVES = (NextHandMoneyObjective, FutureMoneyObjective)  # need a horizon that projects hands
OPPONENTS_NOT_MODELLED = ("Opponent effects are not modelled: attacks only change opponents' hands (cleared), "
                          "and opponents' turns are not simulated.")
MONEY_BUY_NOTE = ("The 'money' objective does not judge which card to buy (coins generated do not depend on it; the "
                  "purchase is only a tie-break), so the Buy in the line is not buy advice: use the 'purchase' objective "
                  "(--objective purchase, heuristic) or `cardgame dominion recommend` for that.")
TARGET_BUY_NOTE = ("The 'target:' objective does not judge which card to buy (it counts target cards in hand and in "
                   "play, or in the first projected hand, which a bought card reaches only through a reshuffle; "
                   "otherwise the purchase is only a tie-break), so the Buy in the line is not buy advice: use the "
                   "'purchase' objective (--objective purchase, heuristic) or `cardgame dominion recommend` for that.")
TIE_HINT = "try another objective or a longer horizon"
TARGET_TIE_HINT = ("the 'target:' value counts only the first projected hand, so a longer horizon does not change it: "
                   "compare the per-hand target probabilities P(target) of the projected hands, or try another objective")
INCONCLUSIVE = ("lower_bound", "partial")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlanConfig:
    """Planner settings. ``horizon=None`` means: the snapshot's ``planning.horizon``, else ``turn``
    (``next_hand`` for the ``next_hand_money`` and ``future_money`` objectives)."""

    objective: str = "money"
    horizon: str | None = None
    max_depth: int = 8
    max_nodes: int = 20000
    max_outcomes_per_chance: int = 200
    samples_per_chance: int = 32
    seed: int = 0
    top_outcomes: int = 5

    def __post_init__(self) -> None:
        get_objective(self.objective)  # raises ValueError for an unknown name
        if self.horizon is not None:
            projected_hand_count(self.horizon)  # raises ValueError for an unknown horizon
        if self.top_outcomes < 1:
            raise ValueError("top_outcomes must be >= 1")
        self.search_config()  # validates depth, nodes, outcomes and samples

    def search_config(self) -> SearchConfig:
        return SearchConfig(max_depth=self.max_depth, max_nodes=self.max_nodes,
                            max_outcomes_per_chance=self.max_outcomes_per_chance,
                            samples_per_chance=self.samples_per_chance, seed=self.seed)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanConfig":
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown PlanConfig field(s): {', '.join(unknown)}")
        return cls(**data)


# ---------------------------------------------------------------------------
# Planning model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanNode:
    """Search state: the current plan state plus, once the turn has ended, the state just before Clean-up."""

    ps: PlanState
    turn: PlanState | None = None

    @property
    def turn_view(self) -> PlanState:
        return self.turn or self.ps

    def key(self) -> tuple:
        return (self.ps.key(), None if self.turn is None else self.turn.key())


class DominionPlanningModel:
    """Implements ``core.search.PlanningModel`` over :class:`PlanNode`. Card effects come from the engine."""

    def __init__(self, engine: DominionRuleEngine, objective: Objective, horizon: str, root: PlanNode,
                 root_actions: list[Action] | None = None):
        self.engine = engine
        self.objective = objective
        self.horizon = horizon
        self.hands = projected_hand_count(horizon)
        self.root = root
        self.root_actions = root_actions  # restricts the root's decisions (compare_actions)
        self._metrics: dict[Hashable, dict[str, Any]] = {}

    # decisions -------------------------------------------------------------
    def decisions(self, s: PlanNode) -> list[Action]:
        if self.is_terminal(s) or self.is_chance(s):
            return []
        if self.root_actions is not None and s is self.root:
            return list(self.root_actions)
        legal = self.engine.legal_actions(s.ps.game)
        if any(isinstance(a, PlayAllTreasures) for a in legal):  # Treasures in hand, nothing bought yet
            legal = [a for a in legal if not isinstance(a, (PlayTreasure, Buy))]
        return legal

    def apply(self, s: PlanNode, a: Action) -> PlanNode:
        if isinstance(a, EndTurn):
            return PlanNode(simulate_end_turn(s.ps, self.engine), turn=s.ps)
        return PlanNode(simulate_action(s.ps, a, self.engine), s.turn)

    def action_key(self, a: Action) -> str:
        return a.describe()

    # chance ----------------------------------------------------------------
    def is_chance(self, s: PlanNode) -> bool:
        return not self.is_terminal(s) and pending_draws(s.ps) > 0

    def _child(self, s: PlanNode, o: DrawOutcome) -> tuple[PlanNode, str]:
        index = len(s.ps.projected) + 1  # which projected hand is being drawn (once the turn ended)
        prefix = "draw " if not s.ps.turn_ended else "next hand: " if index == 1 else f"hand {index}: "
        label = prefix + (", ".join(o.cards) or "nothing")
        if o.reshuffled:
            label += " (after reshuffle)"
        if o.boundary:
            label += f" [{o.boundary}: {o.owed} card(s) owed]"
        ps = simulate_draw(s.ps, o, self.engine)
        if ps.turn_ended and ps.boundary is None and not pending_draws(ps) and index < self.hands:
            ps = simulate_projected_cleanup(ps, self.engine)  # no decisions in future turns
        return PlanNode(ps, s.turn), label

    def exact_outcomes(self, s: PlanNode, max_outcomes: int) -> list[tuple[Fraction, PlanNode, str]] | None:
        ps, n = s.ps, pending_draws(s.ps)
        if count_draw_outcomes(ps.draw, ps.discard, n) > max_outcomes:
            return None
        return [(o.p_exact, *self._child(s, o)) for o in enumerate_draws(ps.draw, ps.discard, n)]

    def outcomes(self, s: PlanNode, max_outcomes: int) -> list[tuple[float, PlanNode, str]] | None:
        exact = self.exact_outcomes(s, max_outcomes)
        return None if exact is None else [(float(p), child, label) for p, child, label in exact]

    def sample(self, s: PlanNode, rng: random.Random) -> tuple[PlanNode, str]:
        return self._child(s, sample_draw(s.ps.draw, s.ps.discard, pending_draws(s.ps), rng))

    # leaves ----------------------------------------------------------------
    def is_terminal(self, s: PlanNode) -> bool:
        ps = s.ps
        if ps.boundary is not None:
            return True
        return ps.turn_ended and (self.hands == 0 or pending_draws(ps) == 0)

    def metrics(self, s: PlanNode) -> dict[str, Any]:
        k = s.key()
        m = self._metrics.get(k)
        if m is None:
            target = self.objective.target if isinstance(self.objective, TargetObjective) else None
            m = leaf_metrics(s.turn_view, s.ps, self.engine, self.root.ps, target)
            m["objective_value"] = self.objective.utility(m)
            self._metrics[k] = m
        return m

    def evaluate(self, s: PlanNode) -> float:
        m = self.metrics(s)
        return m["objective_value"] + TIE_BREAK_WEIGHT * tie_break(m)

    def key(self, s: PlanNode) -> Hashable:
        return s.key()


# ---------------------------------------------------------------------------
# Outcome distribution pass (follows the planner's own argmax decisions)
# ---------------------------------------------------------------------------

class _OutOfBudget(Exception):
    pass


@dataclass
class _Dist:
    value: float                  # ranking value, computed exactly like core.search
    utility: Fraction             # expected objective value (no tie-break)
    groups: dict[str, list]       # label -> [probability (Fraction), representative metrics]
    sampled: bool = False
    has_chance: bool = False


class _DistributionPass:
    def __init__(self, model: DominionPlanningModel, cfg: PlanConfig, budget: int, rng: random.Random):
        self.model, self.cfg, self.budget, self.rng = model, cfg, budget, rng
        self.used = 0
        self.memo: dict[tuple, _Dist] = {}

    def run(self, s: PlanNode, remaining: int) -> _Dist:
        mkey = (self.model.key(s), remaining)
        hit = self.memo.get(mkey)
        if hit is not None:
            return hit
        self.used += 1
        if self.used > self.budget:
            raise _OutOfBudget
        model = self.model
        if model.is_terminal(s):
            d = self.leaf(s)
        elif model.is_chance(s):
            d = self.chance(s, remaining)
        elif remaining <= 0:
            d = self.leaf(s)
        else:
            d = self.decision(s, remaining)
        if not d.sampled:
            self.memo[mkey] = d
        return d

    def leaf(self, s: PlanNode) -> _Dist:
        m = self.model.metrics(s)
        return _Dist(self.model.evaluate(s), Fraction(m["objective_value"]), {leaf_label(m): [Fraction(1), m]})

    def decision(self, s: PlanNode, remaining: int) -> _Dist:
        decisions = {self.model.action_key(a): a for a in reversed(self.model.decisions(s))}
        if not decisions:
            return self.leaf(s)
        best, best_rank = None, None
        for k in sorted(decisions):
            d = self.run(self.model.apply(s, decisions[k]), remaining - 1)
            rank = (-round(d.value, 12), k)  # same ranking as core.search
            if best_rank is None or rank < best_rank:
                best, best_rank = d, rank
        return best

    def chance(self, s: PlanNode, remaining: int) -> _Dist:
        exact = self.model.exact_outcomes(s, self.cfg.max_outcomes_per_chance)
        if exact is not None:
            weighted = [(p, self.run(child, remaining)) for p, child, _ in exact]
            probs = [float(p) for p, _ in weighted]
            total = math.fsum(probs)
            value = math.fsum((pf / total) * d.value for pf, (_, d) in zip(probs, weighted))
            sampled = False
        else:
            n = self.cfg.samples_per_chance
            weighted = [(Fraction(1, n), self.run(self.model.sample(s, self.rng)[0], remaining)) for _ in range(n)]
            value = math.fsum(d.value for _, d in weighted) / n
            sampled = True
        groups: dict[str, list] = {}
        for p, d in weighted:
            for label, (q, m) in d.groups.items():
                g = groups.setdefault(label, [Fraction(0), m])
                g[0] += p * q
        utility = sum((p * d.utility for p, d in weighted), Fraction(0))
        return _Dist(value, utility, groups, sampled or any(d.sampled for _, d in weighted), True)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class OutcomeSummary:
    """One group of leaves with the same label ("important possible future state")."""

    probability: float
    label: str
    metrics: dict[str, Any]       # metrics of a representative leaf of the group
    provenance: str               # of the probability: "exact" or "monte_carlo"
    metric_provenance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateEvaluation:
    action: dict[str, Any]
    action_key: str
    legal: bool = True
    illegal_reasons: list[str] = field(default_factory=list)
    expected_value: float | None = None
    value_provenance: str | None = None   # "exact" | "expected" | "monte_carlo" | "heuristic" | "lower_bound" | "partial"
    stderr: float | None = None
    method: str | None = None             # search method: "exact" | "monte_carlo" | "mixed"
    truncated: bool = False
    outcome_distribution: list[OutcomeSummary] = field(default_factory=list)
    other_probability: float = 0.0        # probability mass not shown in outcome_distribution
    distribution_method: str | None = None  # "exact" | "monte_carlo" | None (not computed)
    threshold_probabilities: dict[int, float] = field(default_factory=dict)
    threshold_provenance: str | None = None  # "exact" | "monte_carlo" | "lower_bound" | "partial" | None
    # One dict per projected hand (horizon next_hand / hands:N): hand, expected_coins, p_at_least_5, provenance
    # (as threshold_provenance), p_lower_bound, and expected_coins_exact / p_at_least_5_exact (fraction strings,
    # only when the distribution is exact); with the target objective also target ("Treasure>=3"), p_target
    # and p_target_exact (P(the hand has >= K target cards)).
    projected_hands: list[dict[str, Any]] = field(default_factory=list)
    boundaries: dict[str, float] = field(default_factory=dict)
    flags: dict[str, float] = field(default_factory=dict)
    principal_line: list[str] = field(default_factory=list)
    nodes: int = 0
    depth_reached: int = 0
    explanation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome_distribution"] = [o.to_dict() for o in self.outcome_distribution]
        d["threshold_probabilities"] = {str(k): v for k, v in self.threshold_probabilities.items()}
        return d


@dataclass
class PlanResult:
    recommended: CandidateEvaluation | None
    alternatives: list[CandidateEvaluation]
    objective: str
    objective_description: str
    value_name: str
    horizon: str
    method: str
    information_level: str
    information_notes: list[str]
    stats: dict[str, Any]
    config: dict[str, Any]
    limitations: list[str]
    explanation: list[str]
    blocked: list[str] = field(default_factory=list)
    tie: bool = False              # best and second-best have equal value: ``recommended`` is only the tie-break
    tied_actions: list[str] = field(default_factory=list)  # labels of the candidates tied with the best

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["recommended"] = self.recommended.to_dict() if self.recommended else None
        d["alternatives"] = [c.to_dict() for c in self.alternatives]
        return d


@dataclass
class ComparisonResult:
    candidates: list[CandidateEvaluation]
    objective: str
    value_name: str
    horizon: str
    information_level: str
    difference_notes: list[str]
    stats: dict[str, Any]
    limitations: list[str]
    tie: bool = False              # the first two legal candidates have equal value under the objective
    tied_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d


# ---------------------------------------------------------------------------
# Building results
# ---------------------------------------------------------------------------

@dataclass
class _Setup:
    engine: DominionRuleEngine
    cfg: PlanConfig
    objective: Objective
    horizon: str
    root: PlanNode
    level: str
    notes: list[str]


def _setup(state: GameState, config: PlanConfig | None, engine: DominionRuleEngine | None) -> _Setup:
    engine = engine or DominionRuleEngine()
    cfg = config or PlanConfig()
    objective = get_objective(cfg.objective, state, engine)
    default = "next_hand" if isinstance(objective, FUTURE_OBJECTIVES) else "turn"
    horizon = cfg.horizon or state.planning_horizon or default
    if isinstance(objective, FUTURE_OBJECTIVES) and projected_hand_count(horizon) == 0:
        raise ValueError(f"The {objective.name} objective needs horizon 'next_hand' or 'hands:N' "
                         f"(1 <= N <= {MAX_PROJECTED_HANDS}).")
    if isinstance(objective, TargetObjective):
        objective = replace(objective, hands=projected_hand_count(horizon))  # utility: this turn or the next hand
    draw, _, notes = derive_zone_knowledge(state, engine.registry)
    root = PlanNode(from_game_state(state, engine))
    return _Setup(engine, cfg, objective, horizon, root, draw.level.value, notes)


def _fmt(v: float | None, stderr: float | None = None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}" + (f" ± {stderr:.2f}" if stderr else "")


def _value_text(c: CandidateEvaluation, name: str = "") -> str:
    """``name=4.00``, or ``name >= 4.00`` for a lower bound (``name`` may be empty)."""
    if c.value_provenance == "lower_bound":
        return f"{name} >= {_fmt(c.expected_value, c.stderr)}".lstrip()
    return f"{name}={_fmt(c.expected_value, c.stderr)}" if name else _fmt(c.expected_value, c.stderr)


def _thresholds_text(c: CandidateEvaluation) -> str:
    op = " >= " if c.threshold_provenance == "lower_bound" else "="
    return ", ".join(f"P($>={t}){op}{p:.2f}" for t, p in c.threshold_probabilities.items())


def _hands_text(c: CandidateEvaluation) -> str:
    parts = []
    for h in c.projected_hands:
        op = " >= " if h["provenance"] == "lower_bound" else "="
        target = f", P({h['target']}){op}{h['p_target']:.2f}" if "p_target" in h else ""
        parts.append(f"hand {h['hand']} E[$]{op}{h['expected_coins']:.2f}, "
                     f"P($>={HAND_THRESHOLD}){op}{h['p_at_least_5']:.2f}{target} ({h['provenance']})")
    return "; ".join(parts)


def _hand_stats(groups: list, n_hands: int, prov: str, flags: dict[str, float],
                objective: Objective | None = None) -> list[dict[str, Any]]:
    """E[$] and P(hand >= $5) per projected hand from the grouped leaf distribution (and, for a
    ``TargetObjective``, P(hand has >= K target cards)).

    A leaf that did not reach hand i (boundary or cut-off) contributes $0 to it, which makes
    that hand's numbers lower bounds (``p_lower_bound`` > 0), like owed or unknown cards.
    """
    target = objective if isinstance(objective, TargetObjective) else None
    out = []
    for i in range(n_hands):
        expected = p_five = p_lower = p_target = Fraction(0)
        for _, (p, m) in groups:
            coins = m.get("hand_coins", [])
            if i < len(coins):
                expected += p * coins[i]
                p_five += p if coins[i] >= HAND_THRESHOLD else 0
                p_lower += p if m["hand_coins_is_lower_bound"][i] else 0
                if target is not None:
                    p_target += p if m["hand_target_counts"][i] >= target.k else 0
            else:
                p_lower += p
        if flags.get("p_manual_boundary"):
            provenance = "partial"
        elif p_lower:
            provenance = "lower_bound"
        else:
            provenance = prov
        exact = prov == "exact"
        out.append({
            "hand": i + 1, "expected_coins": float(expected), "p_at_least_5": float(p_five),
            "provenance": provenance, "p_lower_bound": float(p_lower),
            "expected_coins_exact": str(expected) if exact else None,
            "p_at_least_5_exact": str(p_five) if exact else None,
        })
        if target is not None:
            out[-1].update(target=f"{target.target.name}>={target.k}", p_target=float(p_target),
                           p_target_exact=str(p_target) if exact else None)
    return out


def _line_has_buy(c: CandidateEvaluation) -> bool:
    return any(step.startswith("Buy ") for step in c.principal_line)


def _unknown_counted(objective: Objective | None) -> str:
    """How the objective scores unknown ("?") or unreached cards, for lower-bound wording."""
    return "as non-matching" if isinstance(objective, TargetObjective) else "as $0"


def _inconclusive_note(best: CandidateEvaluation, other: CandidateEvaluation, value_name: str,
                       objective: Objective | None = None) -> str | None:
    """Why ``best`` ranking above ``other`` is not a conclusive comparison (None if it is)."""
    b, o = best.action["label"], other.action["label"]
    if other.value_provenance == "lower_bound":
        return (f"Not conclusive: {o}'s {value_name} is only a lower bound (unknown cards counted "
                f"{_unknown_counted(objective)}), "
                f"so it may be as good as or better than {b}.")
    if other.value_provenance == "partial":
        return (f"Not conclusive: {o}'s {value_name} is partial (a player choice is not modelled and was evaluated "
                f"as if it did nothing), so it may be better than shown.")
    if best.value_provenance == "partial":
        return (f"Not conclusive: {b}'s {value_name} is partial (a player choice is not modelled and was evaluated "
                f"as if it did nothing).")
    return None


def _evaluate_candidate(model: DominionPlanningModel, su: _Setup, av: ActionValue, n_candidates: int
                        ) -> CandidateEvaluation:
    cfg, objective = su.cfg, su.objective
    budget = max(cfg.max_nodes // max(n_candidates, 1), 2 * av.nodes + 16)
    dist_pass = _DistributionPass(model, cfg, budget, random.Random(f"{cfg.seed}:distribution:{av.action_key}"))
    try:
        d: _Dist | None = dist_pass.run(model.apply(model.root, av.action), cfg.max_depth - 1)
    except _OutOfBudget:
        d = None
    exact_search = av.method == "exact" and not av.truncated
    use_pass = d is not None and not d.sampled and exact_search
    c = CandidateEvaluation(
        action=action_to_dict(av.action), action_key=av.action_key,
        # Without the exact pass the search value is used, rounded to 1e-6 to drop the tie-break term.
        expected_value=float(d.utility) if use_pass else round(av.value, 6),
        stderr=av.stderr, method=av.method, truncated=av.truncated, principal_line=list(av.principal_line),
        nodes=av.nodes, depth_reached=av.depth_reached,
    )
    if d is not None:
        groups = sorted(d.groups.items(), key=lambda kv: (-kv[1][0], kv[0]))
        prov = "monte_carlo" if d.sampled else "exact"
        c.distribution_method = prov
        shown = groups[:cfg.top_outcomes]
        c.outcome_distribution = [
            OutcomeSummary(float(p), label, m, prov,
                           {k: METRIC_PROVENANCE.get(k, objective.provenance) for k in m})  # objective_value
            for label, (p, m) in shown
        ]
        c.other_probability = float(Fraction(1) - sum((p for _, (p, _) in shown), Fraction(0)))

        def prob(pred) -> float:
            return float(sum((p for _, (p, m) in groups if pred(m)), Fraction(0)))

        c.threshold_probabilities = {t: prob(lambda m, t=t: m["coins_generated"] >= t) for t in THRESHOLDS}
        for reason in sorted({m["boundary"] for _, (_, m) in groups if m["boundary"]}):
            c.boundaries[reason] = prob(lambda m, r=reason: m["boundary"] == r)
        c.flags = {
            "p_unknown_cards_in_hand": prob(lambda m: m["unknown_cards_in_hand"] > 0),
            "p_stopped_mid_turn": prob(lambda m: m["stopped"] == "mid_turn"),
            "p_next_hand_lower_bound": prob(lambda m: m.get("next_hand_coins_is_lower_bound", False)),
            "p_coins_lower_bound": prob(lambda m: m["coins_is_lower_bound"]),
            "p_value_lower_bound": prob(lambda m: value_is_lower_bound(objective, m)),
            "p_manual_boundary": prob(lambda m: (m["boundary"] or "").startswith("manual:")),
        }
        if c.flags["p_manual_boundary"]:
            c.threshold_provenance = "partial"
        elif c.flags["p_coins_lower_bound"]:
            c.threshold_provenance = "lower_bound"
        else:
            c.threshold_provenance = prov
        c.projected_hands = _hand_stats(groups, model.hands, prov, c.flags, objective)
    if objective.provenance == "heuristic":
        c.value_provenance = "heuristic"
    elif c.flags.get("p_manual_boundary"):
        c.value_provenance = "partial"
    elif c.flags.get("p_value_lower_bound") or (av.truncated and (isinstance(objective, FUTURE_OBJECTIVES) or (
            isinstance(objective, TargetObjective) and objective.hands))):
        # A budget cut-off before the projection ends scores the hands it did not reach as $0.
        c.value_provenance = "lower_bound"
    elif av.method != "exact":
        c.value_provenance = "monte_carlo"
    else:
        c.value_provenance = "expected" if d is None or d.has_chance else "exact"
    c.explanation = _candidate_explanation(c, objective)
    return c


_THRESHOLD_LABELS = {
    "lower_bound": "lower bounds: unknown cards counted as $0",
    "partial": "partial: a player choice is not modelled",
}


def _candidate_explanation(c: CandidateEvaluation, objective: Objective) -> list[str]:
    out = [f"{c.action['label']}: {_value_text(c, objective.value_name)} ({c.value_provenance})."]
    if c.value_provenance == "lower_bound" and c.flags.get("p_value_lower_bound"):
        out.append(f"This value is a lower bound: with probability {c.flags['p_value_lower_bound']:.2f} the line "
                   f"reaches cards of unknown identity, which are counted {_unknown_counted(objective)}.")
    if c.value_provenance == "lower_bound" and c.truncated:
        unreached = "match nothing" if isinstance(objective, TargetObjective) else "count $0"
        out.append("This value is a lower bound: the node budget cut off part of the search, and projected hands "
                   f"it did not reach {unreached} (raise --nodes or shorten the horizon).")
    elif c.value_provenance == "partial":
        out.append(f"This value is partial: with probability {c.flags['p_manual_boundary']:.2f} the line reaches a "
                   "player choice that is not modelled; it is evaluated as if that choice did nothing.")
    if c.distribution_method:
        label = _THRESHOLD_LABELS.get(c.threshold_provenance, c.threshold_provenance)
        out.append(f"Threshold probabilities ({label}): {_thresholds_text(c)}.")
        if c.projected_hands:
            out.append(f"Projected hands (no decisions in future turns): {_hands_text(c)}.")
        top = c.outcome_distribution[0]
        out.append(f"Most likely outcome: {top.label} (p={top.probability:.2f}).")
    else:
        out.append("Outcome distribution not computed (its node budget ran out).")
    if c.principal_line:
        out.append("Expected line: " + " -> ".join(c.principal_line) + ".")
    for reason, p in c.boundaries.items():
        out.append(f"Planning stops at boundary '{reason}' with probability {p:.2f}.")
    return out


def _search(su: _Setup, root_actions: list[Action] | None) -> tuple[list[CandidateEvaluation], SearchStats]:
    model = DominionPlanningModel(su.engine, su.objective, su.horizon, su.root, root_actions)
    values, stats = expectimax(model, su.root, su.cfg.search_config())
    return [_evaluate_candidate(model, su, av, len(values)) for av in values], stats


def _tied(cands: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    """Candidates whose value equals the best one's (within TIE_TOLERANCE), if the second-best does; else [].

    Equal values that are only lower bounds or partial are not a tie (the true values may differ); the
    "Not conclusive" notes report those instead.
    """
    valued = [c for c in cands if c.legal and c.expected_value is not None]
    if len(valued) < 2 or abs(valued[0].expected_value - valued[1].expected_value) > TIE_TOLERANCE:
        return []
    tied = [c for c in valued if abs(valued[0].expected_value - c.expected_value) <= TIE_TOLERANCE]
    return [] if any(c.value_provenance in INCONCLUSIVE for c in tied) else tied


def _tie_note(tied: list[CandidateEvaluation], objective: Objective, horizon: str) -> str:
    labels = ", ".join(c.action["label"] for c in tied)
    hint = TARGET_TIE_HINT if isinstance(objective, TargetObjective) and objective.hands else TIE_HINT
    return (f"TIE: no difference under objective {objective.name} (horizon {horizon}) between {labels} "
            f"({_value_text(tied[0], objective.value_name)}); the decision is beyond this objective/horizon "
            f"({hint}). {tied[0].action['label']} is listed first only by the "
            "documented tie-break (more coins generated, then more cards drawn, more coins spent, more Buys kept), "
            "not because it is better.")


def overall_method(cands: list[CandidateEvaluation]) -> str:
    """The single evaluation method of these candidates, "mixed" if they differ, "none" if none was evaluated."""
    methods = {c.method for c in cands if c.method}
    if not methods:
        return "none"
    return methods.pop() if len(methods) == 1 else "mixed"


def _limitations(su: _Setup, cands: list[CandidateEvaluation], stats: SearchStats) -> list[str]:
    out: list[str] = []
    if su.objective.provenance == "heuristic":
        out.append(f"Objective '{su.objective.name}' is heuristic: values are strategy scores from policy weights, "
                   "not coins or probabilities.")
    else:
        out.append(f"Objective '{su.objective.name}' is exact given the zone knowledge; expected values and "
                   "probabilities come from exact draw enumeration unless marked monte_carlo.")
    out.append(f"Information level: {su.level}. " + " ".join(su.notes))
    if isinstance(su.objective, MoneyObjective) and any(_line_has_buy(c) for c in cands):
        out.append(MONEY_BUY_NOTE)
    if isinstance(su.objective, TargetObjective) and any(_line_has_buy(c) for c in cands):
        out.append(TARGET_BUY_NOTE)
    for c in cands:
        for reason, p in c.boundaries.items():
            if reason.startswith("manual:"):
                effect = "the choice is not modelled, so the value is partial (evaluated as if the choice did nothing)"
            elif c.flags.get("p_value_lower_bound") or su.objective.provenance == "heuristic":
                they = "are counted as non-matching" if isinstance(su.objective, TargetObjective) else "count $0"
                effect = f"cards still owed are unknown, so the value is a lower bound (they {they})"
            else:
                effect = f"cards still owed are unknown, which does not affect this objective's {su.objective.value_name}"
            out.append(f"{c.action['label']}: hidden information / manual choice boundary '{reason}' reached with "
                       f"probability {p:.2f}; {effect}.")
        if c.flags.get("p_unknown_cards_in_hand"):
            out.append(f"{c.action['label']}: unknown cards ('?') are drawn with probability "
                       f"{c.flags['p_unknown_cards_in_hand']:.2f}; they are not played or counted (lower bound).")
        if c.flags.get("p_next_hand_lower_bound"):
            out.append(f"{c.action['label']}: next-hand coins are a lower bound with probability "
                       f"{c.flags['p_next_hand_lower_bound']:.2f} (unknown cards count $0).")
        if c.flags.get("p_stopped_mid_turn"):
            out.append(f"{c.action['label']}: the search stopped mid-turn (depth/budget limit) with probability "
                       f"{c.flags['p_stopped_mid_turn']:.2f}; unplayed Treasures in hand are counted.")
        if c.distribution_method is None and c.legal:
            out.append(f"{c.action['label']}: outcome distribution not computed (node budget).")
    if stats.budget_exhausted:
        truncated = [c.action["label"] for c in cands if c.truncated]
        out.append(f"Node budget ({su.cfg.max_nodes}) exhausted: {stats.budget_cutoffs} state(s) were scored without "
                   f"further lookahead ({', '.join(truncated) or 'no root action affected'}). Values are best under budget.")
    if stats.sampled_chance_nodes:
        out.append(f"{stats.sampled_chance_nodes} chance node(s) had more than {su.cfg.max_outcomes_per_chance} "
                   f"outcomes and were sampled ({su.cfg.samples_per_chance} samples, seed {su.cfg.seed}); "
                   "those values are Monte Carlo estimates ± stderr.")
    out.append(OPPONENTS_NOT_MODELLED)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_next_action(state: GameState, config: PlanConfig | None = None,
                     engine: DominionRuleEngine | None = None) -> PlanResult:
    """Recommend the next action by probabilistic lookahead (best under the configured budget)."""
    su = _setup(state, config, engine)
    blocked = su.engine.blocking_issues(state)
    if blocked:
        cands, stats = [], SearchStats()
        explanation = ["No recommendation: the state is blocked until you resolve: " + " ".join(blocked)]
    else:
        cands, stats = _search(su, None)
        explanation = _plan_explanation(su, cands)
    legal = su.engine.legal_actions(state)
    recommended = cands[0] if cands else None
    tied = _tied(cands)
    if recommended is not None and action_from_dict(recommended.action) not in legal:
        raise RuntimeError(f"planner recommended an illegal action: {recommended.action}")
    return PlanResult(
        recommended=recommended, alternatives=cands[1:], objective=su.objective.name,
        objective_description=su.objective.describe(), value_name=su.objective.value_name, horizon=su.horizon,
        method=overall_method(cands), information_level=su.level, information_notes=list(su.notes),
        stats=stats.to_dict(), config=su.cfg.to_dict(), limitations=_limitations(su, cands, stats),
        explanation=explanation, blocked=blocked, tie=bool(tied), tied_actions=[c.action["label"] for c in tied],
    )


def _plan_explanation(su: _Setup, cands: list[CandidateEvaluation]) -> list[str]:
    if not cands:
        return ["No recommendation: there are no legal actions (the turn is over)."]
    best, vn = cands[0], su.objective.value_name
    out = []
    tied = _tied(cands)
    if tied:
        out.append(_tie_note(tied, su.objective, su.horizon))
    if len(cands) > 1:
        nxt = cands[1]
        diff = (best.expected_value or 0.0) - (nxt.expected_value or 0.0)
        inconclusive = {best.value_provenance, nxt.value_provenance} & set(INCONCLUSIVE)
        shown = " on the shown values" if inconclusive else ""
        if not tied:
            out.append(f"Recommended {best.action['label']}: {_value_text(best, vn)} vs next best "
                       f"{nxt.action['label']}: {_value_text(nxt)} (difference {diff:+.2f}{shown}).")
        if best.threshold_probabilities and nxt.threshold_probabilities:
            out.append(f"{best.action['label']}: {_thresholds_text(best)}; {nxt.action['label']}: {_thresholds_text(nxt)}.")
        if best.projected_hands and nxt.projected_hands:
            out.append(f"{best.action['label']}: {_hands_text(best)}. {nxt.action['label']}: {_hands_text(nxt)}.")
        out.extend(note for c in cands[1:] if (note := _inconclusive_note(best, c, vn, su.objective)))
    else:
        out.append(f"Recommended {best.action['label']} (the only candidate): {_value_text(best, vn)}.")
    out.extend(best.explanation[1:])
    if best.method != "exact":
        out.append(f"Values are {best.method} estimates (sampling); the stderr is shown as ±.")
    return out


def first_action_candidates(state: GameState, engine: DominionRuleEngine | None = None) -> list[Action]:
    """Every legal first action, with individual Play-Treasure actions collapsed into Play-all-Treasures
    (as the planner's own decisions do; no Buy pruning)."""
    legal = list((engine or DominionRuleEngine()).legal_actions(state))
    if any(isinstance(a, PlayAllTreasures) for a in legal):
        legal = [a for a in legal if not isinstance(a, PlayTreasure)]
    return legal


def evaluate_action_sequences(state: GameState, config: PlanConfig | None = None,
                              engine: DominionRuleEngine | None = None) -> list[CandidateEvaluation]:
    """Evaluate every legal first action (``first_action_candidates``), best first."""
    return compare_actions(state, None, config, engine).candidates


def compare_actions(state: GameState, candidates: list[Action | dict[str, Any]] | None = None,
                    config: PlanConfig | None = None, engine: DominionRuleEngine | None = None) -> ComparisonResult:
    """Evaluate the given first actions; illegal ones are reported with the engine's reasons, not evaluated.

    ``candidates=None`` compares every legal first action (``first_action_candidates``); explicit
    candidates may name a single Treasure.
    """
    su = _setup(state, config, engine)
    if candidates is None:
        candidates = first_action_candidates(state, su.engine)
    legal: list[Action] = []
    illegal: list[CandidateEvaluation] = []
    for raw in candidates:
        try:
            action = action_from_dict(raw) if isinstance(raw, dict) else raw
            reasons = su.engine.check(state, action)
        except (TypeError, ValueError) as e:
            illegal.append(CandidateEvaluation(action=dict(raw) if isinstance(raw, dict) else {"label": str(raw)},
                                               action_key=str(raw), legal=False, illegal_reasons=[str(e)]))
            continue
        if reasons:
            illegal.append(CandidateEvaluation(action=action_to_dict(action), action_key=action.describe(),
                                               legal=False, illegal_reasons=reasons,
                                               explanation=[f"{action.describe()} is illegal: " + " ".join(reasons)]))
        else:
            legal.append(action)
    cands, stats = _search(su, legal) if legal else ([], SearchStats())
    notes = []
    tied = _tied(cands)
    if tied:
        notes.append(_tie_note(tied, su.objective, su.horizon))
    if cands:
        best, vn = cands[0], su.objective.value_name
        for c in cands[1:]:
            diff = (best.expected_value or 0.0) - (c.expected_value or 0.0)
            if any(c is t for t in tied):
                note = (f"{best.action['label']} ({_value_text(best, vn)}) vs {c.action['label']} "
                        f"({_value_text(c)}): no difference under objective {su.objective.name}")
            else:
                note = (f"{best.action['label']} ({_value_text(best, vn)}) vs {c.action['label']} "
                        f"({_value_text(c)}): difference {diff:+.2f}")
            if best.threshold_probabilities and c.threshold_probabilities:
                note += f"; {_thresholds_text(best)} vs {_thresholds_text(c)}"
            if best.projected_hands and c.projected_hands:
                note += f"; {_hands_text(best)} vs {_hands_text(c)}"
            notes.append(note + ".")
            inconclusive = _inconclusive_note(best, c, vn, su.objective)
            if inconclusive:
                notes.append(inconclusive)
    notes.extend(f"{c.action.get('label', c.action_key)}: not evaluated (illegal: {' '.join(c.illegal_reasons)})"
                 for c in illegal)
    return ComparisonResult(
        candidates=cands + illegal, objective=su.objective.name, value_name=su.objective.value_name,
        horizon=su.horizon, information_level=su.level, difference_notes=notes, stats=stats.to_dict(),
        limitations=_limitations(su, cands, stats), tie=bool(tied), tied_actions=[c.action["label"] for c in tied],
    )

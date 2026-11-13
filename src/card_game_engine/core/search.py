"""Game-agnostic probabilistic lookahead: depth/budget-limited expectimax with Monte Carlo fallback.

The search knows nothing about any particular game. A game plugin supplies a
:class:`PlanningModel` that exposes three kinds of state:

* **decision nodes** -- the planning player picks one of ``decisions(s)``; the search
  takes the maximum over the children's values;
* **chance nodes** -- ``is_chance(s)`` is true; the model either enumerates the exact
  outcome distribution (``outcomes``) or, when that would be too large, draws samples
  (``sample``). Exact nodes take the probability-weighted sum; sampled nodes take a
  Monte Carlo average and propagate a standard-error estimate;
* **leaves** -- terminal states, states with no decisions (e.g. a planning boundary),
  states at the depth limit and states reached after the node budget ran out. Leaves
  are scored with ``evaluate(s)``.

The result is the *best value found under the given depth and node budget* according
to the model's ``evaluate`` function. It is not a proof of optimality: depth cut-offs,
budget cut-offs and sampling all limit what the search can see.

Determinism: root actions are processed in ``action_key`` order and each root action
gets its own RNG seeded with the string ``f"{seed}:{action_key}"`` (string seeding is
stable across processes, independent of ``PYTHONHASHSEED``). A root action's estimate
therefore does not depend on which other root actions exist or on the order the model
lists them in, provided the node budget is not the limiting factor.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Hashable, Protocol, TypeVar, runtime_checkable

S = TypeVar("S")  # planning state
A = TypeVar("A")  # decision / action

#: Tolerance for ``sum(p) == 1`` when validating exact outcome distributions.
PROBABILITY_TOLERANCE = 1e-9

#: Values are rounded to this many decimals before ranking, so floating-point noise
#: (e.g. ``0.1 + 0.2`` vs ``0.3``) does not defeat the deterministic ``action_key`` tie-break.
_RANK_DECIMALS = 12

__all__ = [
    "PROBABILITY_TOLERANCE", "ActionValue", "PlanningModel", "SearchConfig", "SearchStats", "expectimax",
]


@runtime_checkable
class PlanningModel(Protocol[S, A]):
    """What a game must provide to be searched by :func:`expectimax`.

    All methods must be pure: they never mutate their inputs.
    """

    def decisions(self, s: S) -> list[A]:
        """Legal decisions at ``s`` (``[]`` if terminal, a chance node, or a planning boundary)."""
        ...

    def apply(self, s: S, a: A) -> S:
        """State after taking decision ``a`` in ``s``."""
        ...

    def is_chance(self, s: S) -> bool:
        """True if the next transition from ``s`` is random rather than chosen."""
        ...

    def outcomes(self, s: S, max_outcomes: int) -> list[tuple[float, S, str]] | None:
        """Exact outcome distribution ``[(probability, child, label), ...]`` (probabilities sum
        to 1), or ``None`` when there would be more than ``max_outcomes`` outcomes."""
        ...

    def sample(self, s: S, rng: random.Random) -> tuple[S, str]:
        """One random outcome ``(child, label)`` drawn with ``rng`` (the only randomness source)."""
        ...

    def is_terminal(self, s: S) -> bool:
        """True if nothing further can happen from ``s``."""
        ...

    def evaluate(self, s: S) -> float:
        """Scalar utility of ``s`` from the planning player's point of view (higher is better)."""
        ...

    def key(self, s: S) -> Hashable:
        """Canonical hashable key: equal keys must mean equal future values (transpositions)."""
        ...

    def action_key(self, a: A) -> str:
        """Stable, unique string for a decision (used for ordering, tie-breaks, RNG seeds and lines)."""
        ...


@dataclass
class SearchConfig:
    """Search limits.

    ``max_depth`` counts decision steps (the root action is step 1; chance nodes do not
    consume depth). ``max_nodes`` bounds the number of nodes expanded across all root
    actions. A chance node is enumerated exactly when the model reports at most
    ``max_outcomes_per_chance`` outcomes, otherwise ``samples_per_chance`` samples are
    averaged. ``use_cache`` toggles the transposition cache (values are identical
    either way; it only saves work). ``collect_principal_line`` toggles recording of
    each root action's principal line.
    """

    max_depth: int = 6
    max_nodes: int = 20000
    max_outcomes_per_chance: int = 200
    samples_per_chance: int = 32
    seed: int = 0
    use_cache: bool = True
    collect_principal_line: bool = True

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        if self.max_outcomes_per_chance < 1:
            raise ValueError("max_outcomes_per_chance must be >= 1")
        if self.samples_per_chance < 2:
            raise ValueError("samples_per_chance must be >= 2 (a standard error needs at least 2 samples)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown SearchConfig field(s): {', '.join(unknown)}")
        return cls(**data)


@dataclass
class SearchStats:
    """Counters describing one :func:`expectimax` run.

    ``nodes_expanded`` counts every state processed (internal or leaf), excluding cache
    hits and budget cut-offs; the root counts as one. ``budget_cutoffs`` counts states
    scored with ``evaluate`` only because the node budget ran out.
    """

    nodes_expanded: int = 0
    chance_nodes: int = 0
    exact_chance_nodes: int = 0
    sampled_chance_nodes: int = 0
    max_depth_reached: int = 0
    budget_exhausted: bool = False
    budget_cutoffs: int = 0
    cache_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionValue:
    """Value of one root action.

    * ``value`` -- expected value under the model's ``evaluate``, best-under-budget.
    * ``stderr`` -- ``None`` when ``method == "exact"``; otherwise the propagated Monte Carlo
      standard-error estimate of ``value``.
    * ``method`` -- ``"exact"`` (no sampling anywhere in the explored subtree),
      ``"monte_carlo"`` (every chance node in the subtree was sampled) or ``"mixed"``
      (both exact and sampled chance nodes).
    * ``truncated`` -- the node budget cut off part of this action's subtree.
    * ``principal_line`` -- the root action key, then along the line: the chosen decision's
      key at decision nodes and the most probable (exact) / most frequently sampled (Monte
      Carlo) outcome label at chance nodes.
    """

    action: Any
    action_key: str
    value: float
    stderr: float | None
    method: str
    nodes: int
    depth_reached: int
    truncated: bool = False
    principal_line: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": _jsonable_action(self.action, self.action_key),
            "action_key": self.action_key,
            "value": self.value,
            "stderr": self.stderr,
            "method": self.method,
            "nodes": self.nodes,
            "depth_reached": self.depth_reached,
            "truncated": self.truncated,
            "principal_line": list(self.principal_line),
        }


def _jsonable_action(action: Any, action_key: str) -> Any:
    to_dict = getattr(action, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if action is None or isinstance(action, (str, int, float, bool)):
        return action
    return action_key


class _Result:
    """Value of a subtree plus provenance flags (internal)."""

    __slots__ = ("value", "var", "has_exact_chance", "has_sampled", "truncated", "rel_depth", "line")

    def __init__(self, value: float, var: float = 0.0, has_exact_chance: bool = False, has_sampled: bool = False,
                 truncated: bool = False, rel_depth: int = 0, line: tuple[str, ...] = ()) -> None:
        self.value = value
        self.var = var                      # variance of the estimate (0 for exact subtrees)
        self.has_exact_chance = has_exact_chance
        self.has_sampled = has_sampled
        self.truncated = truncated          # a budget cut-off happened somewhere below
        self.rel_depth = rel_depth          # max decision steps taken below this node
        self.line = line

    @property
    def method(self) -> str:
        if not self.has_sampled:
            return "exact"
        return "mixed" if self.has_exact_chance else "monte_carlo"


def _rank_key(value: float, action_key: str) -> tuple[float, str]:
    return (-round(value, _RANK_DECIMALS), action_key)


class _Searcher:
    def __init__(self, model: PlanningModel, config: SearchConfig) -> None:
        self.model = model
        self.cfg = config
        self.stats = SearchStats()
        self.cache: dict[tuple[Hashable, int], _Result] = {}
        self.rng = random.Random()
        self.used = 0
        self.allowance = 0
        self.lines = config.collect_principal_line

    def sorted_decisions(self, s: Any) -> list[tuple[str, Any]]:
        seen: dict[str, Any] = {}
        for a in self.model.decisions(s):
            k = self.model.action_key(a)
            seen.setdefault(k, a)  # identical keys are the same decision by contract
        return sorted(seen.items(), key=lambda kv: kv[0])

    def leaf(self, s: Any) -> _Result:
        return _Result(float(self.model.evaluate(s)))

    def visit(self, s: Any, remaining: int, depth: int) -> _Result:
        model, stats = self.model, self.stats
        ckey = None
        if self.cfg.use_cache:
            ckey = (model.key(s), remaining)
            hit = self.cache.get(ckey)
            if hit is not None:
                stats.cache_hits += 1
                stats.max_depth_reached = max(stats.max_depth_reached, depth + hit.rel_depth)
                return hit
        stats.max_depth_reached = max(stats.max_depth_reached, depth)
        if self.used >= self.allowance:
            stats.budget_exhausted = True
            stats.budget_cutoffs += 1
            r = self.leaf(s)
            r.truncated = True
            return r
        self.used += 1
        stats.nodes_expanded += 1

        if model.is_terminal(s):
            r = self.leaf(s)
        elif model.is_chance(s):
            r = self.chance(s, remaining, depth)
        elif remaining <= 0:
            r = self.leaf(s)
        else:
            decisions = self.sorted_decisions(s)
            r = self.decision(s, decisions, remaining, depth) if decisions else self.leaf(s)

        if ckey is not None and not r.has_sampled and not r.truncated:
            self.cache[ckey] = r
        return r

    def decision(self, s: Any, decisions: list[tuple[str, Any]], remaining: int, depth: int) -> _Result:
        best: _Result | None = None
        best_rank: tuple[float, str] | None = None
        best_key = ""
        has_exact = has_sampled = truncated = False
        rel_depth = 0
        for k, a in decisions:
            r = self.visit(self.model.apply(s, a), remaining - 1, depth + 1)
            has_exact |= r.has_exact_chance
            has_sampled |= r.has_sampled
            truncated |= r.truncated
            rel_depth = max(rel_depth, 1 + r.rel_depth)
            rank = _rank_key(r.value, k)
            if best_rank is None or rank < best_rank:
                best, best_rank, best_key = r, rank, k
        assert best is not None
        line = (best_key,) + best.line if self.lines else ()
        # The max's stderr is approximated by the chosen child's stderr (selection bias ignored).
        return _Result(best.value, best.var, has_exact, has_sampled, truncated, rel_depth, line)

    def chance(self, s: Any, remaining: int, depth: int) -> _Result:
        self.stats.chance_nodes += 1
        outs = self.model.outcomes(s, self.cfg.max_outcomes_per_chance)
        if outs is not None:
            return self.exact_chance(outs, remaining, depth)
        return self.sampled_chance(s, remaining, depth)

    def exact_chance(self, outs: list[tuple[float, Any, str]], remaining: int, depth: int) -> _Result:
        self.stats.exact_chance_nodes += 1
        probs = []
        for item in outs:
            p = float(item[0])
            if not math.isfinite(p) or p < 0.0:
                raise ValueError(f"invalid outcome probability {item[0]!r} (label {item[2]!r})")
            probs.append(p)
        total = math.fsum(probs)
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError(f"outcome probabilities sum to {total!r}, expected 1 (tolerance {PROBABILITY_TOLERANCE})")
        value_terms: list[float] = []
        var_terms: list[float] = []
        has_sampled = truncated = False
        rel_depth = 0
        best_rank: tuple[float, str] | None = None
        best_line: tuple[str, ...] = ()
        for p, (_, child, label) in zip(probs, outs):
            if p == 0.0:
                continue
            w = p / total
            r = self.visit(child, remaining, depth)
            value_terms.append(w * r.value)
            var_terms.append(w * w * r.var)
            has_sampled |= r.has_sampled
            truncated |= r.truncated
            rel_depth = max(rel_depth, r.rel_depth)
            if self.lines:
                rank = (-p, str(label))
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_line = ((str(label),) if label else ()) + r.line
        return _Result(math.fsum(value_terms), math.fsum(var_terms), True, has_sampled, truncated, rel_depth,
                       best_line)

    def sampled_chance(self, s: Any, remaining: int, depth: int) -> _Result:
        self.stats.sampled_chance_nodes += 1
        n = self.cfg.samples_per_chance
        values: list[float] = []
        child_vars: list[float] = []
        has_exact = truncated = False
        rel_depth = 0
        label_counts: Counter[str] = Counter()
        first_line: dict[str, tuple[str, ...]] = {}
        for _ in range(n):
            child, label = self.model.sample(s, self.rng)
            r = self.visit(child, remaining, depth)
            values.append(r.value)
            child_vars.append(r.var)
            has_exact |= r.has_exact_chance
            truncated |= r.truncated
            rel_depth = max(rel_depth, r.rel_depth)
            if self.lines:
                label = str(label)
                label_counts[label] += 1
                first_line.setdefault(label, r.line)
        mean = math.fsum(values) / n
        sample_var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
        # Var(mean) is estimated by s^2 / n, which already includes the children's own
        # estimation noise in expectation; the second term guards the degenerate case
        # where every sample happened to land on equal values.
        var = max(sample_var / n, math.fsum(child_vars) / (n * n))
        line: tuple[str, ...] = ()
        if self.lines and label_counts:
            label = min(label_counts, key=lambda lb: (-label_counts[lb], lb))
            line = ((label,) if label else ()) + first_line[label]
        return _Result(mean, var, has_exact, True, truncated, rel_depth, line)


def expectimax(model: PlanningModel, root: Any, config: SearchConfig | None = None
               ) -> tuple[list[ActionValue], SearchStats]:
    """Value every legal root decision by depth- and budget-limited expectimax.

    Returns the root actions sorted best-first (ties broken by ``action_key``) and the
    run's :class:`SearchStats`. If ``root`` is terminal, a chance node, or has no
    decisions, returns ``([], stats)``.

    Budget fairness: root actions are processed in ``action_key`` order; each gets an
    allowance of ``remaining_budget // remaining_actions`` node expansions, so every root
    action receives at least ``max_nodes // n_root_actions`` and none is starved. Every
    root action is valued even when its allowance is zero (its child is then scored with
    ``evaluate`` and marked truncated).

    Values are best-under-budget estimates according to ``model.evaluate``; they are not
    guaranteed optimal.
    """
    cfg = config or SearchConfig()
    searcher = _Searcher(model, cfg)
    stats = searcher.stats
    if model.is_terminal(root) or model.is_chance(root):
        return [], stats
    actions = searcher.sorted_decisions(root)
    if not actions:
        return [], stats
    stats.nodes_expanded += 1  # the root itself

    results: list[ActionValue] = []
    remaining_budget = cfg.max_nodes
    for i, (k, a) in enumerate(actions):
        searcher.allowance = remaining_budget // (len(actions) - i)
        searcher.used = 0
        searcher.rng = random.Random(f"{cfg.seed}:{k}")
        r = searcher.visit(model.apply(root, a), cfg.max_depth - 1, 1)
        remaining_budget -= searcher.used
        method = r.method
        results.append(ActionValue(
            action=a,
            action_key=k,
            value=r.value,
            stderr=None if method == "exact" else math.sqrt(r.var),
            method=method,
            nodes=searcher.used,
            depth_reached=1 + r.rel_depth,
            truncated=r.truncated,
            principal_line=[k, *r.line] if cfg.collect_principal_line else [],
        ))
    results.sort(key=lambda av: _rank_key(av.value, av.action_key))
    return results, stats

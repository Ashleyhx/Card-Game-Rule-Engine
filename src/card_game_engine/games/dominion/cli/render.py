"""Human-readable rendering (plain ASCII so it works in any terminal)."""

from __future__ import annotations

from collections import Counter

from ..planning import CandidateEvaluation, ComparisonResult, PlanConfig, PlanResult, overall_method
from ..policy import RankedAction, Recommendation
from ..rules import DominionRuleEngine, GameState, Issue

RULE = "-" * 72


def fmt_cards(cards) -> str:
    if not cards:
        return "(none)"
    counts = Counter(cards)
    return ", ".join(f"{n} x{c}" if c > 1 else n for n, c in sorted(counts.items()))


def render_state(state: GameState, engine: DominionRuleEngine) -> list[str]:
    t = state.turn
    start = engine.ruleset.starting_province_count(state.num_players)
    lines = [
        f"Players: {state.num_players}   Turn: {state.turn_number or '?'}   Phase: {t.phase.value.upper()}",
        f"Hand:     {fmt_cards(t.hand)}",
        f"In play:  {fmt_cards(t.in_play)}",
    ]
    if t.bought:
        lines.append(f"Bought:   {fmt_cards(t.bought)}")
    lines.append(
        f"Actions {t.actions} | Buys {t.buys} | ${t.coins}    "
        f"Provinces {state.supply.get('Province', '?')}/{start}    "
        f"Empty piles: {', '.join(state.empty_piles()) or '-'}"
    )
    if state.owned_cards is not None:
        lines.append(f"Your VP: {engine.victory_points(state.owned_cards)} ({sum(state.owned_cards.values())} cards)")
    opp = [f"{o.name} {o.vp if o.vp is not None else '?'}" for o in state.opponents]
    if opp:
        lines.append("Opponent VP: " + ", ".join(opp))
    if state.history:
        lines.append("Bought so far: " + "; ".join(
            f"T{h['turn'] or '?'} {fmt_cards(h['bought']) if h['bought'] else '-'}" for h in state.history))
    return lines


def _ranked_lines(r: RankedAction, verbose: bool, indent: str = "  ") -> list[str]:
    out = [f"{indent}{r.rank:>2}. {r.action.describe():<44} score {r.score:7.2f}"]
    if verbose:
        for reason in r.reasons:
            out.append(f"{indent}      {reason}")
    for w in r.warnings:
        out.append(f"{indent}      ! {w}")
    return out


def render_advice(advice: Recommendation, state: GameState, engine: DominionRuleEngine, top: int = 5, verbose: bool = True) -> str:
    lines = ["Card Game Rule Engine - Dominion 2E Base Set (offline)", RULE]
    lines += render_state(state, engine)
    lines.append(f"Stage: {advice.stage} ({advice.stage_reason})")
    lines.append(RULE)

    if advice.blocking:
        lines.append("Cannot recommend yet - resolve these first:")
        lines += [f"  - {b}" for b in advice.blocking]
        return "\n".join(lines)

    best = advice.best
    if best is None:
        lines.append("No moves left this turn.")
        return "\n".join(lines)
    lines.append(f">> RECOMMENDED: {best.action.describe()}    [confidence: {advice.confidence}]")
    for reason in best.reasons:
        lines.append(f"     {reason}")
    for w in best.warnings:
        lines.append(f"     ! {w}")

    if advice.projected:
        lines.append("")
        lines.append(f"After playing your Treasures you will have ${advice.projected_coins}. Best buys:")
        for r in advice.projected[:top]:
            lines += _ranked_lines(r, verbose)
    elif len(advice.ranked) > 1:
        lines.append("")
        lines.append("Alternatives:")
        for r in advice.ranked[1:top]:
            lines += _ranked_lines(r, verbose)

    lines.append("")
    lines.append("Plan for this phase: " + " -> ".join(advice.plan))
    lines.append(f"Confidence: {advice.confidence} - {advice.confidence_reason}")
    lines.append("Uncertainty & assumptions:")
    lines += [f"  - {u}" for u in advice.uncertainties]
    return "\n".join(lines)


# ---------------------------------------------------------------- lookahead planner

METHOD_NAMES = {
    "exact": "exact enumeration of draws",
    "monte_carlo": "Monte Carlo sampling (estimates ± stderr)",
    "mixed": "mixed: exact enumeration + Monte Carlo sampling",
    "none": "no search (nothing to evaluate)",
}


def plan_value(c: CandidateEvaluation) -> str:
    """``= 4.00``, ``= 4.00 ± 0.12`` when sampled, ``>= 4.00`` for a lower bound."""
    if c.expected_value is None:
        return "n/a"
    v = f"{c.expected_value:.2f}"
    if c.stderr:
        v += f" ± {c.stderr:.2f}"
    return (">= " if c.value_provenance == "lower_bound" else "= ") + v


def _tags(c: CandidateEvaluation) -> str:
    return f"[{c.value_provenance}; search {c.method}]"


def _thresholds(c: CandidateEvaluation) -> str:
    if not c.threshold_probabilities:
        return "not computed"
    op = " >= " if c.threshold_provenance == "lower_bound" else "="
    text = ", ".join(f"P($>={t}){op}{p:.2f}" for t, p in c.threshold_probabilities.items())
    return f"{text}  ({c.threshold_provenance})"


def _projected_hands(c: CandidateEvaluation, indent: str) -> list[str]:
    lines = [f"{indent}Projected hands (no decisions in future turns):"] if c.projected_hands else []
    for h in c.projected_hands:
        op = ">= " if h["provenance"] == "lower_bound" else "= "
        lines.append(f"{indent}  hand {h['hand']}: E[$] {op}{h['expected_coins']:.2f}, "
                     f"P($>=5) {op}{h['p_at_least_5']:.2f}  ({h['provenance']})")
    return lines


def _tie_lines(tied: list[str], objective: str, value_name: str, best: CandidateEvaluation, horizon: str) -> list[str]:
    return [f">> TIE under objective {objective} (horizon {horizon}): {', '.join(tied)}    "
            f"{value_name} {plan_value(best)}",
            "   No difference under this objective/horizon - the decision is beyond it (try another objective or a "
            "longer horizon).",
            f"   Listed first by tie-break only (NOT better): {best.action['label']}    {_tags(best)}"]


def _candidate_details(c: CandidateEvaluation, indent: str = "  ") -> list[str]:
    lines = [f"{indent}Threshold probabilities: {_thresholds(c)}"] + _projected_hands(c, indent)
    if c.outcome_distribution:
        lines.append(f"{indent}Top future outcomes ({c.distribution_method}):")
        lines += [f"{indent}  p={o.probability:.2f}  {o.label}" for o in c.outcome_distribution]
        if c.other_probability > 1e-9:
            lines.append(f"{indent}  p={c.other_probability:.2f}  (other outcomes)")
    else:
        lines.append(f"{indent}Top future outcomes: not computed (node budget)")
    if c.principal_line:
        lines.append(f"{indent}Principal line: " + " -> ".join(c.principal_line))
    return lines


def _candidate_row(rank: int, c: CandidateEvaluation, value_name: str) -> str:
    return f"  {rank:>2}. {c.action['label']:<44} {value_name} {plan_value(c):<17} {_tags(c)}"


def _search_lines(stats: dict, cfg: PlanConfig) -> list[str]:
    return [
        "Search:",
        f"  nodes expanded {stats['nodes_expanded']} (budget {cfg.max_nodes}), max depth reached "
        f"{stats['max_depth_reached']} (limit {cfg.max_depth}), budget exhausted: "
        f"{'yes' if stats['budget_exhausted'] else 'no'}",
        f"  chance nodes {stats['chance_nodes']} (exact {stats['exact_chance_nodes']}, sampled "
        f"{stats['sampled_chance_nodes']}), cache hits {stats['cache_hits']}, seed {cfg.seed}",
    ]


def _plan_header(state: GameState, engine: DominionRuleEngine, objective: str, description: str, horizon: str,
                 method: str, level: str, notes: list[str]) -> list[str]:
    lines = ["Card Game Rule Engine - Dominion 2E Base Set (offline) - lookahead planner", RULE]
    lines += render_state(state, engine)
    lines.append(RULE)
    lines.append(f"Objective: {objective} - {description}")
    lines.append(f"Horizon: {horizon}    Method: {METHOD_NAMES.get(method, method)}")
    lines.append(f"Information level: {level}")
    lines += [f"  - {n}" for n in notes]
    lines.append(RULE)
    return lines


def render_plan(result: PlanResult, state: GameState, engine: DominionRuleEngine, cfg: PlanConfig) -> str:
    lines = _plan_header(state, engine, result.objective, result.objective_description, result.horizon,
                         result.method, result.information_level, result.information_notes)
    best = result.recommended
    if best is None:
        lines += result.explanation
        return "\n".join(lines)
    if result.tie:
        lines += _tie_lines(result.tied_actions, result.objective, result.value_name, best, result.horizon)
    else:
        lines.append(f">> RECOMMENDED: {best.action['label']}    {result.value_name} {plan_value(best)} {_tags(best)}")
    lines += _candidate_details(best, "     ")
    if result.alternatives:
        lines += ["", "Alternatives:"]
        lines += [_candidate_row(i, c, result.value_name)
                  for i, c in enumerate(result.alternatives[:cfg.top_outcomes], start=2)]
        hidden = len(result.alternatives) - cfg.top_outcomes
        if hidden > 0:
            lines.append(f"  ... {hidden} more (use --top or 'compare')")
    lines += ["", "Why:"] + [f"  - {e}" for e in result.explanation]
    lines += [""] + _search_lines(result.stats, cfg)
    lines += ["Limitations:"] + [f"  - {x}" for x in result.limitations]
    return "\n".join(lines)


def render_comparison(result: ComparisonResult, state: GameState, engine: DominionRuleEngine, cfg: PlanConfig,
                      description: str) -> str:
    legal = [c for c in result.candidates if c.legal]
    illegal = [c for c in result.candidates if not c.legal]
    lines = _plan_header(state, engine, result.objective, description, result.horizon, overall_method(legal),
                         result.information_level, [])
    if not legal:
        lines.append("No legal action to evaluate.")
    elif result.tie:
        lines += _tie_lines(result.tied_actions, result.objective, result.value_name, legal[0], result.horizon)
    else:
        lines.append(f">> RECOMMENDED: {legal[0].action['label']}    {result.value_name} {plan_value(legal[0])} "
                     f"{_tags(legal[0])}")
    if legal:
        lines += ["", "Side by side (best first):"]
        for i, c in enumerate(legal, start=1):
            lines.append(_candidate_row(i, c, result.value_name))
            lines += _candidate_details(c, "        ")
    if illegal:
        lines += ["", "Not evaluated (illegal):"]
        lines += [f"  - {c.action.get('label', c.action_key)}: {' '.join(c.illegal_reasons)}" for c in illegal]
    if result.difference_notes:
        lines += ["", "Why:"] + [f"  - {n}" for n in result.difference_notes]
    lines += [""] + _search_lines(result.stats, cfg)
    lines += ["Limitations:"] + [f"  - {x}" for x in result.limitations]
    return "\n".join(lines)


def render_issues(issues: list[Issue]) -> str:
    if not issues:
        return "No issues found."
    tag = {"error": "ERROR", "warning": "WARN ", "info": "info "}
    return "\n".join(f"[{tag[i.severity]}] {i.message}" for i in issues)

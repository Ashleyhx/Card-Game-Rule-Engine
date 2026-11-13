"""Human-readable rendering (plain ASCII so it works in any terminal)."""

from __future__ import annotations

from collections import Counter

from ..policy import Recommendation, RankedAction
from ..rules import GameState, Issue, DominionRuleEngine

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


def render_issues(issues: list[Issue]) -> str:
    if not issues:
        return "No issues found."
    tag = {"error": "ERROR", "warning": "WARN ", "info": "info "}
    return "\n".join(f"[{tag[i.severity]}] {i.message}" for i in issues)

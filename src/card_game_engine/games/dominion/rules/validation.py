"""Rule-level consistency checks on a parsed snapshot."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .engine import DominionRuleEngine
from .state import GameState, Phase


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warning" | "info"
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "message": self.message}


def validate_state(state: GameState, engine: DominionRuleEngine | None = None) -> list[Issue]:
    engine = engine or DominionRuleEngine()
    reg, rules = engine.registry, engine.ruleset
    issues: list[Issue] = []
    t = state.turn
    allowed = set(state.supply)

    # Supply bounds
    for pile, n in state.supply.items():
        start = rules.starting_pile_size(pile, state.num_players, reg)
        if n > start:
            issues.append(Issue("error", f"Supply: {pile} has {n} cards but starts with only {start}."))

    # Cards must belong to this game
    for label, cards in (("hand", t.hand), ("in play", t.in_play), ("bought this turn", t.bought)):
        for c in sorted(set(cards)):
            if c not in allowed:
                issues.append(Issue("error", f"{c} (in {label}) is not part of this game's Supply."))
    if state.owned_cards:
        for c in sorted(state.owned_cards):
            if c not in allowed:
                issues.append(Issue("error", f"{c} (in owned_cards) is not part of this game's Supply."))

    # Hand and play must be a subset of owned cards
    if state.owned_cards is not None:
        visible = Counter(t.hand) + Counter(t.in_play)
        for c, n in visible.items():
            if state.owned_cards.get(c, 0) < n:
                issues.append(Issue(
                    "error",
                    f"owned_cards lists {state.owned_cards.get(c, 0)} {c} but {n} are in your hand/play area.",
                ))

    # Phase consistency
    treasures_in_play = [c for c in t.in_play if reg.get(c).is_treasure]
    if t.phase == Phase.ACTION:
        if treasures_in_play:
            issues.append(Issue("error", "Treasures are in play but it is the Action phase (Treasures are played in the Buy phase)."))
        if t.bought:
            issues.append(Issue("error", "Cards were bought this turn but it is still the Action phase."))

    # Coin sanity (skipped after Clean-up, which has already discarded everything).
    if t.phase != Phase.TURN_OVER:
        treasure_value = sum(reg.get(c).treasure_value for c in treasures_in_play)
        if t.silver_played and "Silver" in treasures_in_play:
            treasure_value += t.merchants_played
        spent = sum(reg.get(c).cost for c in t.bought)
        expected = treasure_value - spent
        if not any(reg.get(c).is_action for c in t.in_play) and t.coins > expected:
            issues.append(Issue(
                "error",
                f"Coins (${t.coins}) are more than your cards in play produce (${expected}). "
                "'coins' counts only cards already played; Treasures still in your hand are added when you play them.",
            ))
        elif t.coins < expected:
            issues.append(Issue(
                "warning",
                f"Coins (${t.coins}) are lower than your Treasures in play produce (${treasure_value}) minus spending (${spent}). Double-check 'coins'.",
            ))

    if t.phase == Phase.ACTION and t.actions == 0 and any(reg.get(c).is_action for c in t.hand):
        issues.append(Issue("info", "You have Action cards in hand but 0 Actions left; only 'End Action phase' is legal."))

    if t.pending_draws or t.unresolved:
        issues.append(Issue("warning", "There are pending draws/manual steps; resolve them to get recommendations."))

    end = engine.game_end_status(state)
    if end.game_over:
        issues.append(Issue("info", "End condition already met - the game ends after this turn. " + " ".join(end.reasons)))

    for a in state.assumptions:
        issues.append(Issue("info", "Assumption: " + a))
    return issues


def has_errors(issues: list[Issue]) -> bool:
    return any(i.severity == "error" for i in issues)

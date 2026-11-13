"""Objectives (strategy profiles) and leaf metrics for the lookahead planner.

A leaf is described by two planning states: ``turn`` - the state at the end of (or,
for a cut-off leaf, during) the current turn, before Clean-up - and ``final`` - the
last state reached (after Clean-up and the next-hand draw for the ``next_hand``
horizon; the same object as ``turn`` otherwise). :func:`leaf_metrics` turns them into
plain numbers; an :class:`Objective` turns the metrics into one utility.

Provenance of every number is explicit (``METRIC_PROVENANCE``):

* ``exact``      - follows from the rules and the leaf state alone;
* ``expected``   - probability-weighted over an exact enumeration of draws;
* ``monte_carlo``- a sampled estimate;
* ``heuristic``  - a strategy opinion (policy weights), never a probability.

Candidate values can additionally be labelled ``lower_bound`` (unknown cards counted as
$0, see :func:`value_is_lower_bound`) or ``partial`` (a player sub-choice that is not
modelled was evaluated as if it did nothing); see ``planner.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from ..policy import weights as W
from ..policy.context import build_context
from ..rules.actions import PlayAllTreasures
from ..rules.engine import DominionRuleEngine
from ..rules.state import GameState, Phase
from ..rules.zones import UNKNOWN_CARD
from .draws import BOUNDARY_UNKNOWN_DRAW_PILE
from .transitions import PlanState

OBJECTIVE_NAMES = ("money", "threshold:<T>", "purchase", "next_hand_money")

#: Weight of the secondary tie-break added to the utility during search. The tie-break
#: is in [0, 1), so it moves a value by less than 1e-6 and never outweighs a real
#: difference (see :func:`tie_break`).
TIE_BREAK_WEIGHT = 1e-6

METRIC_PROVENANCE: dict[str, str] = {
    "coins": "exact",
    "spent": "exact",
    "coins_generated": "exact",
    "unplayed_treasure_coins": "exact",
    "buys_left": "exact",
    "actions_left": "exact",
    "cards_drawn_this_turn": "exact",
    "action_cards_stranded": "exact",
    "bought": "exact",
    "unknown_cards_in_hand": "exact",
    "coins_is_lower_bound": "exact",
    "boundary": "exact",
    "stopped": "exact",
    "next_hand": "exact",
    "next_hand_coins": "exact",
    "next_hand_unknown": "exact",
    "next_hand_coins_is_lower_bound": "exact",
}


def _zone_card_count(ps: PlanState) -> int:
    t = ps.game.turn
    return len(t.hand) + len(t.in_play) + ps.unknown_in_hand + len(ps.game.trash)


def _playable_treasure_coins(ps: PlanState, engine: DominionRuleEngine) -> int:
    """Coins the Treasures still in hand would add if played now (computed by the engine).

    Only for a leaf cut off mid-turn with nothing bought: playing Treasures is then always
    possible later in the turn, so counting them is a lower bound, not a guess.
    """
    t = ps.game.turn
    if t.bought or t.phase == Phase.TURN_OVER or not any(engine.card(c).is_treasure for c in t.hand):
        return 0
    game = replace(ps.game, turn=replace(t, phase=Phase.BUY, pending_draws=0, unresolved=[]))
    return engine.apply(game, PlayAllTreasures()).state.turn.coins - t.coins


def leaf_metrics(turn: PlanState, final: PlanState, engine: DominionRuleEngine, root: PlanState) -> dict[str, Any]:
    """Plain, exactly computed facts about one leaf (see ``METRIC_PROVENANCE``).

    ``cards_drawn_this_turn`` counts cards drawn since the snapshot ``root``.
    ``coins_is_lower_bound``: this turn's coins may be higher than computed, because unknown
    cards ("?") are in hand or drawing stopped mid-turn at an unknown draw pile (both count $0).
    """
    t = turn.game.turn
    spent = sum(engine.card(c).cost for c in t.bought)
    ended = final.turn_ended
    extra = 0 if ended else _playable_treasure_coins(turn, engine)
    actions_in_hand = sum(1 for c in t.hand if engine.card(c).is_action)
    if final.boundary:
        stopped = "boundary"
    elif final.next_hand is not None and final.game.turn.pending_draws == 0:
        stopped = "next_hand"
    elif ended:
        stopped = "turn_end"
    else:
        stopped = "mid_turn"
    m: dict[str, Any] = {
        "coins": t.coins,
        "spent": spent,
        "coins_generated": t.coins + spent + extra,
        "unplayed_treasure_coins": extra,
        "buys_left": t.buys,
        "actions_left": t.actions,
        "cards_drawn_this_turn": _zone_card_count(turn) - _zone_card_count(root),
        "action_cards_stranded": actions_in_hand if t.actions == 0 else 0,
        "bought": sorted(t.bought),
        "unknown_cards_in_hand": turn.unknown_in_hand,
        "coins_is_lower_bound": turn.unknown_in_hand > 0
        or (final.boundary == BOUNDARY_UNKNOWN_DRAW_PILE and not final.turn_ended),
        "boundary": final.boundary,
        "stopped": stopped,
    }
    if final.next_hand is not None:
        unknown = sum(1 for c in final.next_hand if c == UNKNOWN_CARD)
        m["next_hand"] = list(final.next_hand)
        m["next_hand_coins"] = sum(engine.card(c).treasure_value for c in final.next_hand if c != UNKNOWN_CARD)
        m["next_hand_unknown"] = unknown
        m["next_hand_coins_is_lower_bound"] = unknown > 0 or final.game.turn.pending_draws > 0
    return m


def leaf_label(m: dict[str, Any]) -> str:
    """Readable grouping label for an outcome, e.g. ``"coins=$6, bought Gold"`` (``$6+`` = lower bound)."""
    parts = [f"coins=${m['coins_generated']}{'+' if m['coins_is_lower_bound'] else ''}"]
    parts.append("bought " + ", ".join(m["bought"]) if m["bought"] else "bought nothing")
    if m["unknown_cards_in_hand"]:
        parts.append(f"{m['unknown_cards_in_hand']} unknown card(s) in hand")
    if "next_hand_coins" in m:
        lb = "+" if m["next_hand_coins_is_lower_bound"] else ""
        parts.append(f"next hand ${m['next_hand_coins']}{lb}")
    if m["boundary"]:
        parts.append(f"boundary {m['boundary']}")
    elif m["stopped"] == "mid_turn":
        parts.append("stopped mid-turn (depth/budget limit)")
    return ", ".join(parts)


def tie_break(m: dict[str, Any]) -> float:
    """Secondary preference in [0, 1), in priority order: more coins generated, more cards drawn
    (unknown cards count $0 but may be Treasures), more coins spent, more Buys kept."""
    n = (min(m["coins_generated"], 99) * 10_000 + min(max(m["cards_drawn_this_turn"], 0), 9) * 1_000
         + min(m["spent"], 99) * 10 + min(m["buys_left"], 9))
    return n / 1_000_000


class Objective(Protocol):
    name: str
    provenance: str      # "exact" or "heuristic" for the utility of a single leaf
    value_name: str      # how the expected utility is shown, e.g. "E[$]"

    def describe(self) -> str: ...

    def utility(self, metrics: dict[str, Any]) -> float: ...


@dataclass(frozen=True)
class MoneyObjective:
    name: str = "money"
    provenance: str = "exact"
    value_name: str = "E[$]"

    def describe(self) -> str:
        return ("Coins generated this turn (coins left + cost of cards bought this turn); exact, no heuristics. "
                "It does not judge which card to buy: among buys of equal cost the line picks arbitrarily "
                "(use 'purchase' for buy quality).")

    def utility(self, metrics: dict[str, Any]) -> float:
        return metrics["coins_generated"]


@dataclass(frozen=True)
class ThresholdObjective:
    target: int
    provenance: str = "exact"

    @property
    def name(self) -> str:
        return f"threshold:{self.target}"

    @property
    def value_name(self) -> str:
        return f"P($>={self.target})"

    def describe(self) -> str:
        return (f"1 if at least ${self.target} is generated this turn, else 0: the expected value is the "
                f"probability of reaching ${self.target}.")

    def utility(self, metrics: dict[str, Any]) -> float:
        return 1.0 if metrics["coins_generated"] >= self.target else 0.0


@dataclass(frozen=True)
class PurchaseObjective:
    stage: str
    stage_reason: str = ""
    name: str = "purchase"
    provenance: str = "heuristic"
    value_name: str = "E[buy value]"

    def describe(self) -> str:
        return (f"Heuristic value of the cards bought this turn (policy weights, {self.stage} stage: "
                f"{self.stage_reason or 'n/a'}); a strategy score, not coins or a probability.")

    def card_value(self, card: str) -> float:
        if card in W.BASIC_BUY_VALUE:
            return W.BASIC_BUY_VALUE[card][self.stage]
        return W.KINGDOM_RATING.get(card, 4.0) * W.KINGDOM_STAGE_MULTIPLIER[self.stage]

    def utility(self, metrics: dict[str, Any]) -> float:
        return sum(self.card_value(c) for c in metrics["bought"])


@dataclass(frozen=True)
class NextHandMoneyObjective:
    name: str = "next_hand_money"
    provenance: str = "exact"
    value_name: str = "E[next-hand $]"

    def describe(self) -> str:
        return ("Coins from Treasures in the hand drawn after Clean-up; unknown cards ('?') count $0, "
                "so the value is a lower bound when unknown cards can be drawn.")

    def utility(self, metrics: dict[str, Any]) -> float:
        return metrics.get("next_hand_coins", 0)


def value_is_lower_bound(objective: Objective, metrics: dict[str, Any]) -> bool:
    """True when a leaf's objective value only bounds the true value from below.

    Coins, ``threshold:<T>`` and next-hand coins never decrease when an unknown card turns
    out to be known (Treasures add coins, Actions may be left unplayed), so counting unknown
    cards as $0 gives a lower bound. Heuristic objectives are never labelled bounds.
    """
    if objective.provenance == "heuristic":
        return False
    if isinstance(objective, NextHandMoneyObjective):
        return metrics["coins_is_lower_bound"] or metrics.get("next_hand_coins_is_lower_bound", False)
    return metrics["coins_is_lower_bound"]


def get_objective(name: str, state: GameState | None = None, engine: DominionRuleEngine | None = None) -> Objective:
    """Parse an objective name. ``purchase`` computes the game stage once from ``state``."""
    key = (name or "").strip().lower()
    if key == "money":
        return MoneyObjective()
    if key == "next_hand_money":
        return NextHandMoneyObjective()
    if key == "purchase":
        if state is None:
            return PurchaseObjective("mid", "no state given; mid stage assumed")
        ctx = build_context(state, engine or DominionRuleEngine())
        return PurchaseObjective(ctx.stage, ctx.stage_reason)
    if key.startswith("threshold:"):
        raw = key.split(":", 1)[1]
        if raw.isdigit() and int(raw) >= 1:
            return ThresholdObjective(int(raw))
        raise ValueError(f"Bad threshold objective {name!r}: expected threshold:<T> with an integer T >= 1 (e.g. threshold:6).")
    raise ValueError(f"Unknown objective {name!r}; choose one of: {', '.join(OBJECTIVE_NAMES)} (e.g. threshold:6).")


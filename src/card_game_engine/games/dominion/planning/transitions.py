"""Pure planning transitions: rule-engine moves plus explicit zone knowledge.

Card effects come only from :class:`DominionRuleEngine` (``apply`` / ``resolve_pending``);
this module adds what the engine deliberately leaves open: which cards are drawn
(a :class:`DrawOutcome` chosen by a chance node) and where known cards move between
the draw pile, discard pile and hand. Inputs are never mutated. For the ``hands:N``
horizon, ``simulate_projected_cleanup`` moves a projected future hand (no decisions in
future turns) to the discard pile so that the following hand can be drawn.

Manual steps left by the engine are handled as follows:

* ``OPPONENT_ONLY_STEPS`` (attacks / "each other player draws") do not change your
  turn and are cleared. Simplification: Witch's Curses are not removed from the Supply.
* Library: the "set aside Actions" choice is resolved as "keep everything" (always legal).
* Bureaucrat's Silver goes onto the draw pile; Bandit's Gold into the discard pile.
* Any other manual step makes the state a leaf: ``boundary = "manual:<card>"``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from ..rules.actions import Action, Buy, EndTurn, PlayAction, PlayAllTreasures, PlayTreasure
from ..rules.engine import DominionRuleEngine
from ..rules.state import GameState
from ..rules.zones import (
    UNKNOWN_CARD,
    DiscardKnowledge,
    DrawPileKnowledge,
    add_to_discard,
    cleanup_zones,
    derive_zone_knowledge,
    put_on_top,
)
from .draws import DrawOutcome

OPPONENT_ONLY_STEPS: tuple[str, ...] = (
    "Militia attack:",
    "Witch attack:",
    "Bandit attack:",
    "Bureaucrat attack:",
    "Council Room: each other player draws",
)
LIBRARY_STEP = "Library:"


@dataclass(frozen=True)
class PlanState:
    game: GameState                  # never mutated; new states come from the engine / dataclasses.replace
    draw: DrawPileKnowledge
    discard: DiscardKnowledge
    turn_ended: bool = False
    next_hand: tuple[str, ...] | None = None  # Stage B: hand drawn after Clean-up (the latest projected hand)
    projected: tuple[tuple[str, ...], ...] = ()  # earlier projected hands, already cleaned up (horizon hands:N)
    boundary: str | None = None      # why planning cannot continue from here
    unknown_in_hand: int = 0         # drawn "?" cards (the engine only tracks known cards)

    def key(self) -> tuple:
        """Canonical hashable key for memoization."""
        t = self.game.turn
        return (
            t.phase.value, tuple(sorted(t.hand)), self.unknown_in_hand, tuple(sorted(t.in_play)),
            t.actions, t.buys, t.coins, tuple(sorted(t.bought)), t.merchants_played, t.silver_played,
            t.pending_draws, tuple(sorted(t.unresolved)), tuple(sorted(self.game.supply.items())),
            tuple(sorted(self.game.trash)),
            (self.draw.top, self.draw.rest, self.draw.unknown, self.draw.size_known),
            (self.discard.cards, self.discard.unknown, self.discard.size_known),
            self.turn_ended, self.next_hand, self.projected, self.boundary,
        )

    @property
    def projected_hands(self) -> tuple[tuple[str, ...], ...]:
        """Every projected future hand so far, earliest first (the last one may still be owed cards)."""
        return self.projected + (() if self.next_hand is None else (self.next_hand,))


def _sync(ps: PlanState) -> PlanState:
    """Keep the game's zone fields consistent with the plan's zone knowledge."""
    draw, discard = ps.draw, ps.discard
    game = replace(
        ps.game,
        draw_pile=draw if draw.size_known else None,
        deck_count=draw.size if draw.size_known else None,
        discard_pile=discard if discard.size_known else None,
        discard_count=discard.size if discard.size_known else None,
    )
    return replace(ps, game=game)


def _step_card(step: str) -> str:
    return step.split(":", 1)[0].split(" (", 1)[0]


def _settle(ps: PlanState) -> PlanState:
    """Clear manual steps that need no decision; flag a boundary for the rest."""
    steps = ps.game.turn.unresolved
    remaining = [s for s in steps if not s.startswith(OPPONENT_ONLY_STEPS + (LIBRARY_STEP,))]
    game = ps.game
    if remaining != steps:
        game = replace(game, turn=replace(game.turn, unresolved=remaining))
    boundary = f"manual:{_step_card(remaining[0])}" if remaining else ps.boundary
    return _sync(replace(ps, game=game, boundary=boundary))


def from_game_state(state: GameState, engine: DominionRuleEngine) -> PlanState:
    """Root planning state: zone knowledge derived by ``derive_zone_knowledge`` (no guessing)."""
    draw, discard, _ = derive_zone_knowledge(state, engine.registry)
    return _settle(PlanState(state.copy(), draw, discard))


def _place_gains(
    before: GameState, after: GameState, action: PlayAction, ps: PlanState, engine: DominionRuleEngine
) -> tuple[DrawPileKnowledge, DiscardKnowledge]:
    """Put cards the engine gained (Bureaucrat's Silver, Bandit's Gold) into the right zone."""
    gain = engine.card(action.throne_target or action.card).gain
    if gain is None:
        return ps.draw, ps.discard
    gained = [gain.card] * (before.supply.get(gain.card, 0) - after.supply.get(gain.card, 0))
    if gain.destination == "deck":
        return put_on_top(ps.draw, gained), ps.discard
    return ps.draw, add_to_discard(ps.discard, gained)


def simulate_action(ps: PlanState, action: Action, engine: DominionRuleEngine) -> PlanState:
    """Apply any decision via the engine (raises IllegalActionError); the result may have pending draws."""
    if isinstance(action, EndTurn):
        return simulate_end_turn(ps, engine)
    after = engine.apply(ps.game, action).state
    draw, discard = ps.draw, ps.discard
    if isinstance(action, Buy):
        discard = add_to_discard(discard, [engine.card(action.card).name])
    elif isinstance(action, PlayAction):
        draw, discard = _place_gains(ps.game, after, action, ps, engine)
        if ps.unknown_in_hand and engine.card(action.throne_target or action.card).special == "library":
            # The engine sizes "draw to 7" from the known hand only; "?" cards in hand count too.
            target = 7 - (len(after.turn.hand) + ps.unknown_in_hand)
            after.turn.pending_draws = max(ps.game.turn.pending_draws, target, 0)  # ``after`` is a fresh copy
    return _settle(replace(ps, game=after, draw=draw, discard=discard))


def simulate_treasure(ps: PlanState, card: str | None, engine: DominionRuleEngine) -> PlanState:
    """Play one Treasure, or all Treasures in hand when ``card`` is None."""
    return simulate_action(ps, PlayAllTreasures() if card is None else PlayTreasure(card), engine)


def simulate_purchase(ps: PlanState, card: str, engine: DominionRuleEngine) -> PlanState:
    """Buy ``card``; the engine pays and updates the Supply, the card joins the known discard pile."""
    return simulate_action(ps, Buy(card), engine)


def pending_draws(ps: PlanState) -> int:
    return ps.game.turn.pending_draws


def simulate_draw(ps: PlanState, outcome: DrawOutcome, engine: DominionRuleEngine) -> PlanState:
    """Apply a draw outcome for the pending draws (this turn's hand, or ``next_hand`` once the turn ended)."""
    pending = pending_draws(ps)
    if pending == 0 or len(outcome.cards) + outcome.owed > pending:
        raise ValueError(f"Outcome draws {len(outcome.cards) + outcome.owed} card(s) but {pending} draw(s) are pending.")
    known = [c for c in outcome.cards if c != UNKNOWN_CARD]
    unknown = len(outcome.cards) - len(known)
    common = dict(draw=outcome.draw_after, discard=outcome.discard_after, boundary=outcome.boundary or ps.boundary)
    if ps.turn_ended:
        game = replace(ps.game, turn=replace(ps.game.turn, pending_draws=outcome.owed))
        next_hand = tuple(sorted((ps.next_hand or ()) + outcome.cards))
        return _sync(replace(ps, game=game, next_hand=next_hand, **common))
    game = engine.resolve_pending(ps.game, drawn=known, clear_unresolved=False)
    game.turn.pending_draws = outcome.owed  # ``game`` is a fresh copy
    return _sync(replace(ps, game=game, unknown_in_hand=ps.unknown_in_hand + unknown, **common))


def simulate_end_turn(ps: PlanState, engine: DominionRuleEngine) -> PlanState:
    """EndTurn via the engine; hand and play go to the discard pile and the next hand is pending."""
    after = engine.apply(ps.game, EndTurn()).state
    after.turn.pending_draws = engine.ruleset.hand_size  # ``after`` is a fresh copy
    t = ps.game.turn
    discard = cleanup_zones(ps.discard, t.hand, t.in_play, ps.unknown_in_hand)
    return _sync(replace(ps, game=after, discard=discard, turn_ended=True, unknown_in_hand=0))


def simulate_projected_cleanup(ps: PlanState, engine: DominionRuleEngine) -> PlanState:
    """Clean-up of a projected future turn in which nothing is decided (horizon ``hands:N``).

    The fully drawn projected hand goes to the discard pile ("?" cards as unknown cards) and
    the following hand becomes pending; drawing it (and reshuffling, if the rules require it)
    is a separate chance step (``simulate_draw``).
    """
    if not ps.turn_ended or ps.next_hand is None or pending_draws(ps):
        raise ValueError("A projected Clean-up needs a fully drawn projected hand after the turn ended.")
    known = [c for c in ps.next_hand if c != UNKNOWN_CARD]
    discard = cleanup_zones(ps.discard, known, (), len(ps.next_hand) - len(known))
    game = replace(ps.game, turn=replace(ps.game.turn, pending_draws=engine.ruleset.hand_size))
    return _sync(replace(ps, game=game, discard=discard, projected=ps.projected_hands, next_hand=None))


def card_totals(ps: PlanState) -> Counter:
    """Every card accounted for across hand, play, known zones, next hand and "?" placeholders.

    Earlier projected hands (``projected``) are only a record: their cards are in the discard pile.
    """
    t = ps.game.turn
    totals = Counter(t.hand) + Counter(t.in_play) + Counter(ps.next_hand or ())
    totals[UNKNOWN_CARD] += ps.unknown_in_hand
    if ps.draw.size_known:
        totals += Counter(ps.draw.top) + ps.draw.rest_counter
        totals[UNKNOWN_CARD] += ps.draw.unknown
    if ps.discard.size_known:
        totals += ps.discard.counter
        totals[UNKNOWN_CARD] += ps.discard.unknown
    return +totals

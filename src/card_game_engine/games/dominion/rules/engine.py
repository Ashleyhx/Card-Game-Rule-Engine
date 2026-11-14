"""Deterministic rule engine: legal-action enumeration, constraint checks, transitions.

Design principles
-----------------
* Pure functions over :class:`GameState`: ``apply`` returns a *new* state.
* Never invent hidden information. "+N Cards" becomes ``pending_draws``; effects that
  need player sub-choices or hidden knowledge become ``unresolved`` manual steps.
* While draws or manual steps are pending, no further actions are legal (a card must be
  fully resolved before the next one is played).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ....core.interfaces import IllegalActionError, Transition
from .actions import Action, Buy, EndActionPhase, EndTurn, PlayAction, PlayAllTreasures, PlayTreasure
from .cards import CardDef, CardRegistry, UnknownCardError, load_registry
from .ruleset import Ruleset, load_ruleset
from .state import GameState, Phase
from .zones import add_to_discard, put_on_top

__all__ = ["DominionRuleEngine", "GameEndStatus", "IllegalActionError", "Transition"]


@dataclass(frozen=True)
class GameEndStatus:
    game_over: bool
    reasons: tuple[str, ...]
    empty_piles: tuple[str, ...]


class DominionRuleEngine:
    """Dominion 2E Base Set rules; implements :class:`card_game_engine.core.RuleEngine`."""

    def __init__(self, registry: CardRegistry | None = None, ruleset: Ruleset | None = None):
        self.registry = registry or load_registry()
        self.ruleset = ruleset or load_ruleset()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def card(self, name: str) -> CardDef:
        return self.registry.get(name)

    def blocking_issues(self, state: GameState) -> list[str]:
        """Reasons the current decision point cannot proceed (pending hidden info / manual steps)."""
        issues: list[str] = []
        t = state.turn
        if t.pending_draws > 0:
            issues.append(f"Draw {t.pending_draws} card(s) and add them to your hand before continuing.")
        for step in t.unresolved:
            issues.append(f"Resolve manually: {step}")
        return issues

    def check(self, state: GameState, action: Action) -> list[str]:
        """Return a list of rule violations; an empty list means the action is legal."""
        blocking = self.blocking_issues(state)
        if blocking:
            return blocking
        t = state.turn
        if t.phase == Phase.TURN_OVER:
            return ["The turn is over."]

        if isinstance(action, PlayAction):
            return self._check_play_action(state, action)
        if isinstance(action, EndActionPhase):
            return [] if t.phase == Phase.ACTION else ["Not in the Action phase."]
        if isinstance(action, PlayTreasure):
            return self._check_play_treasure(state, action.card)
        if isinstance(action, PlayAllTreasures):
            reasons = []
            if t.phase != Phase.BUY:
                reasons.append("Treasures can only be played in the Buy phase.")
            if not any(self._is_treasure(c) for c in t.hand):
                reasons.append("No Treasures in hand.")
            if t.bought and self.ruleset.no_treasures_after_buying:
                reasons.append("You cannot play Treasures after buying a card this turn.")
            return reasons
        if isinstance(action, Buy):
            return self._check_buy(state, action.card)
        if isinstance(action, EndTurn):
            return [] if t.phase == Phase.BUY else ["End the Action phase first (the Buy phase always follows)."]
        return [f"Unknown action type {type(action).__name__}"]

    def legal_actions(self, state: GameState) -> list[Action]:
        if self.blocking_issues(state) or state.turn.phase == Phase.TURN_OVER:
            return []
        t = state.turn
        candidates: list[Action] = []
        if t.phase == Phase.ACTION:
            for name in sorted(set(t.hand)):
                cdef = self.card(name)
                if not cdef.is_action:
                    continue
                if cdef.special == "throne_room":
                    hand_after = Counter(t.hand)
                    hand_after[name] -= 1
                    targets = sorted(c for c, n in hand_after.items() if n > 0 and self.card(c).is_action)
                    for target in targets:
                        candidates.append(PlayAction(name, throne_target=target))
                    candidates.append(PlayAction(name))
                elif cdef.special == "moneylender":
                    if "Copper" in t.hand:
                        candidates.append(PlayAction(name, trash_copper=True))
                    candidates.append(PlayAction(name, trash_copper=False))
                else:
                    candidates.append(PlayAction(name))
            candidates.append(EndActionPhase())
        elif t.phase == Phase.BUY:
            treasures = sorted({c for c in t.hand if self._is_treasure(c)})
            if treasures:
                candidates.append(PlayAllTreasures())
                candidates.extend(PlayTreasure(c) for c in treasures)
            for pile in state.supply:
                candidates.append(Buy(pile))
            candidates.append(EndTurn())
        return [a for a in candidates if not self.check(state, a)]

    def illegal_explanations(self, state: GameState) -> list[tuple[Action, list[str]]]:
        """For each Supply pile you *cannot* buy (in the Buy phase), explain why."""
        out: list[tuple[Action, list[str]]] = []
        if state.turn.phase == Phase.BUY:
            for pile in state.supply:
                reasons = self.check(state, Buy(pile))
                if reasons:
                    out.append((Buy(pile), reasons))
        return out

    # ------------------------------------------------------------------
    # Scoring / end of game
    # ------------------------------------------------------------------
    def victory_points(self, owned: dict[str, int]) -> int:
        total_cards = sum(owned.values())
        vp = 0
        for name, n in owned.items():
            cdef = self.card(name)
            if cdef.special == "gardens":
                vp += n * (total_cards // 10)
            elif cdef.vp is not None:
                vp += n * cdef.vp
        return vp

    def game_end_status(self, state: GameState) -> GameEndStatus:
        """Would the game end if the current turn ended now?"""
        empty = tuple(state.empty_piles())
        reasons: list[str] = []
        if state.supply.get("Province", 1) <= 0:
            reasons.append("The Province pile is empty.")
        if len(empty) >= self.ruleset.empty_piles_to_end:
            reasons.append(f"{len(empty)} Supply piles are empty ({', '.join(empty)}).")
        return GameEndStatus(bool(reasons), tuple(reasons), empty)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def apply(self, state: GameState, action: Action) -> Transition:
        reasons = self.check(state, action)
        if reasons:
            raise IllegalActionError(action, reasons)
        s = state.copy()
        notes: list[str] = []
        if isinstance(action, PlayAction):
            self._apply_play_action(s, action, notes)
        elif isinstance(action, EndActionPhase):
            s.turn.phase = Phase.BUY
            notes.append("Entered the Buy phase.")
        elif isinstance(action, PlayTreasure):
            self._play_treasure(s, action.card, notes)
        elif isinstance(action, PlayAllTreasures):
            # Silver first so that Merchant bonuses are visible in notes; order does not change the total.
            treasures = sorted((c for c in s.turn.hand if self._is_treasure(c)),
                               key=lambda c: (c != "Silver", c))
            for c in treasures:
                self._play_treasure(s, c, notes)
        elif isinstance(action, Buy):
            self._apply_buy(s, action.card, notes)
        elif isinstance(action, EndTurn):
            self._apply_cleanup(s, notes)
        return Transition(s, notes)

    def resolve_pending(
        self,
        state: GameState,
        drawn: list[str] | None = None,
        hand: list[str] | None = None,
        clear_unresolved: bool = True,
    ) -> GameState:
        """Record what the player drew / how they resolved manual steps.

        ``drawn`` cards are added to hand (and must not exceed ``pending_draws``),
        or ``hand`` replaces the hand entirely (used after manual effects such as Chapel).
        """
        s = state.copy()
        t = s.turn
        if drawn is not None:
            canon = [self.registry.resolve(c) for c in drawn]
            if len(canon) > t.pending_draws:
                raise ValueError(f"You drew {len(canon)} cards but only {t.pending_draws} draw(s) are pending.")
            t.hand = sorted(t.hand + canon)
            t.pending_draws = 0  # fewer cards = deck and discard ran out
            if canon and (s.draw_pile is not None or s.discard_pile is not None):
                # Which pile the cards came from (and whether a reshuffle happened) isn't observable
                # here, so known zone contents can no longer be trusted: drop them rather than guess.
                s.draw_pile = s.discard_pile = None
        if hand is not None:
            t.hand = sorted(self.registry.resolve(c) for c in hand)
            t.pending_draws = 0
        if clear_unresolved:
            t.unresolved = []
        return s

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _is_treasure(self, name: str) -> bool:
        return self.card(name).is_treasure

    def _check_play_action(self, state: GameState, action: PlayAction) -> list[str]:
        t = state.turn
        reasons: list[str] = []
        try:
            cdef = self.card(action.card)
        except UnknownCardError as e:
            return [str(e)]
        if t.phase != Phase.ACTION:
            reasons.append("Action cards can only be played in the Action phase.")
        if not cdef.is_action:
            reasons.append(f"{cdef.name} is not an Action card.")
        if cdef.name not in t.hand:
            reasons.append(f"{cdef.name} is not in your hand.")
        if t.actions < 1:
            reasons.append("No Actions remaining.")
        if action.throne_target is not None:
            if cdef.special != "throne_room":
                reasons.append("Only Throne Room takes a target.")
            else:
                try:
                    target = self.card(action.throne_target)
                except UnknownCardError as e:
                    return reasons + [str(e)]
                hand_after = Counter(t.hand)
                hand_after[cdef.name] -= 1
                if hand_after[target.name] < 1:
                    reasons.append(f"No {target.name} in hand for Throne Room to play.")
                if not target.is_action:
                    reasons.append(f"Throne Room can only play an Action card ({target.name} is not).")
        if action.trash_copper is not None and cdef.special != "moneylender":
            reasons.append("Only Moneylender has a trash-Copper option.")
        if action.trash_copper and "Copper" not in t.hand:
            reasons.append("No Copper in hand to trash.")
        return reasons

    def _check_play_treasure(self, state: GameState, card: str) -> list[str]:
        t = state.turn
        reasons: list[str] = []
        try:
            cdef = self.card(card)
        except UnknownCardError as e:
            return [str(e)]
        if t.phase != Phase.BUY:
            reasons.append("Treasures can only be played in the Buy phase.")
        if not cdef.is_treasure:
            reasons.append(f"{cdef.name} is not a Treasure.")
        if cdef.name not in t.hand:
            reasons.append(f"{cdef.name} is not in your hand.")
        if t.bought and self.ruleset.no_treasures_after_buying:
            reasons.append("You cannot play Treasures after buying a card this turn.")
        return reasons

    def _check_buy(self, state: GameState, card: str) -> list[str]:
        t = state.turn
        reasons: list[str] = []
        try:
            cdef = self.card(card)
        except UnknownCardError as e:
            return [str(e)]
        if t.phase != Phase.BUY:
            reasons.append("Cards can only be bought in the Buy phase.")
        if t.buys < 1:
            reasons.append("No Buys remaining.")
        if cdef.name not in state.supply:
            reasons.append(f"{cdef.name} is not in this game's Supply.")
        elif state.supply[cdef.name] <= 0:
            reasons.append(f"The {cdef.name} pile is empty.")
        if cdef.cost > t.coins:
            reasons.append(f"Costs ${cdef.cost}; you have ${t.coins}.")
        return reasons

    def _owned_add(self, s: GameState, card: str, n: int = 1) -> None:
        if s.owned_cards is not None:
            s.owned_cards[card] = s.owned_cards.get(card, 0) + n
            if s.owned_cards[card] <= 0:
                del s.owned_cards[card]

    def _resolve_effects(self, s: GameState, cdef: CardDef, notes: list[str], label: str, via_throne: bool) -> None:
        t = s.turn
        e = cdef.effects
        t.actions += e.actions
        t.buys += e.buys
        t.coins += e.coins
        if e.cards:
            t.pending_draws += e.cards
        bonuses = []
        if e.cards:
            bonuses.append(f"+{e.cards} Card{'s' if e.cards != 1 else ''}")
        if e.actions:
            bonuses.append(f"+{e.actions} Action{'s' if e.actions != 1 else ''}")
        if e.buys:
            bonuses.append(f"+{e.buys} Buy{'s' if e.buys != 1 else ''}")
        if e.coins:
            bonuses.append(f"+${e.coins}")
        if bonuses:
            notes.append(f"{label}: " + ", ".join(bonuses) + ".")

        # Moneylender's trash choice is handled by the caller (part of PlayAction, or manual via Throne Room).
        special = cdef.special
        if special == "merchant":
            t.merchants_played += 1
            if t.silver_played:
                notes.append(f"{label}: a Silver was already played this turn, so this Merchant gives no bonus.")
        elif special == "poacher":
            n = len(s.empty_piles())
            if n:
                t.unresolved.append(f"{label}: discard {n} card(s) from your hand ({n} empty Supply pile(s)).")
        elif special == "library":
            t.unresolved.append(cdef.manual_steps[0] if cdef.manual_steps else "Library: draw to 7.")
            target = 7 - (len(t.hand) + t.pending_draws)
            if target > 0:
                t.pending_draws += target
                notes.append(f"{label}: draw until you have 7 cards ({target} more).")
        if cdef.gain is not None:
            g = cdef.gain
            if s.supply.get(g.card, 0) > 0:
                s.supply[g.card] -= 1
                self._owned_add(s, g.card)
                if g.destination == "deck":
                    if s.deck_count is not None:
                        s.deck_count += 1
                    if s.draw_pile is not None:
                        s.draw_pile = put_on_top(s.draw_pile, [g.card])
                elif g.destination == "discard":
                    if s.discard_count is not None:
                        s.discard_count += 1
                    if s.discard_pile is not None:
                        s.discard_pile = add_to_discard(s.discard_pile, [g.card])
                notes.append(f"{label}: gained a {g.card} ({'onto your deck' if g.destination == 'deck' else 'to your discard pile'}).")
            else:
                notes.append(f"{label}: no {g.card} left in the Supply, so nothing is gained.")
        if special != "library":
            for step in cdef.manual_steps:
                t.unresolved.append(step + (" (Throne Room play)" if via_throne else ""))

    def _apply_play_action(self, s: GameState, action: PlayAction, notes: list[str]) -> None:
        t = s.turn
        cdef = self.card(action.card)
        t.hand.remove(cdef.name)
        t.in_play.append(cdef.name)
        t.actions -= 1
        notes.append(f"Played {cdef.name} (used 1 Action).")
        if cdef.special == "throne_room":
            if action.throne_target is None:
                notes.append("Throne Room: chose not to play a card.")
                return
            target = self.card(action.throne_target)
            t.hand.remove(target.name)
            t.in_play.append(target.name)
            for i in (1, 2):
                label = f"{target.name} (Throne Room play {i})"
                self._resolve_effects(s, target, notes, label, via_throne=True)
                if target.special == "moneylender":
                    t.unresolved.append(
                        f"{label}: you may trash a Copper from hand for +$3 - update your hand and coins."
                    )
                if target.special == "throne_room":
                    t.unresolved.append(
                        f"{label}: choose another Action in hand to play twice and apply its effects manually."
                    )
            return
        if cdef.special == "moneylender" and action.trash_copper:
            t.hand.remove("Copper")
            s.trash.append("Copper")
            self._owned_add(s, "Copper", -1)
            t.coins += 3
            notes.append("Moneylender: trashed a Copper for +$3.")
        self._resolve_effects(s, cdef, notes, cdef.name, via_throne=False)

    def _play_treasure(self, s: GameState, card: str, notes: list[str]) -> None:
        t = s.turn
        cdef = self.card(card)
        t.hand.remove(cdef.name)
        t.in_play.append(cdef.name)
        t.coins += cdef.treasure_value
        bonus = ""
        if cdef.name == "Silver" and not t.silver_played:
            t.silver_played = True
            if t.merchants_played:
                t.coins += t.merchants_played
                bonus = f" (+${t.merchants_played} from Merchant)"
        notes.append(f"Played {cdef.name}: +${cdef.treasure_value}{bonus}.")

    def _apply_buy(self, s: GameState, card: str, notes: list[str]) -> None:
        t = s.turn
        cdef = self.card(card)
        t.coins -= cdef.cost
        t.buys -= 1
        s.supply[cdef.name] -= 1
        t.bought.append(cdef.name)
        self._owned_add(s, cdef.name)
        if s.discard_count is not None:
            s.discard_count += 1
        if s.discard_pile is not None:
            s.discard_pile = add_to_discard(s.discard_pile, [cdef.name])
        notes.append(f"Bought {cdef.name} for ${cdef.cost} (${t.coins} and {t.buys} Buy(s) left).")
        if s.supply[cdef.name] == 0:
            notes.append(f"The {cdef.name} pile is now empty.")

    def _apply_cleanup(self, s: GameState, notes: list[str]) -> None:
        t = s.turn
        discarded = len(t.hand) + len(t.in_play)
        if s.discard_count is not None:
            s.discard_count += discarded
        if s.discard_pile is not None:
            s.discard_pile = add_to_discard(s.discard_pile, t.hand + t.in_play)
        t.hand = []
        t.in_play = []
        t.actions = t.buys = t.coins = 0
        t.phase = Phase.TURN_OVER
        t.pending_draws = 0
        notes.append(f"Clean-up: discarded {discarded} card(s) from hand and play; draw 5 for your next turn.")
        end = self.game_end_status(s)
        if end.game_over:
            notes.append("GAME OVER: " + " ".join(end.reasons))

"""Applying one event to a belief state: the reducer that :mod:`.session` replays.

``apply_event`` turns one recorded event into the next :class:`BeliefState`, or raises
:class:`InconsistentObservation` (a contradiction) / :class:`IllegalEvent` (malformed or untrackable).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from functools import lru_cache
from typing import Any

from ..rules.actions import Action, Buy, EndActionPhase, EndTurn, PlayAction, PlayAllTreasures, PlayTreasure
from ..rules.cards import UnknownCardError
from ..rules.engine import DominionRuleEngine
from ..rules.state import Phase, TurnState
from ..rules.zones import UNKNOWN_CARD, add_to_discard, expand_counts, put_on_top
from .belief import (
    BeliefState,
    Contradiction,
    World,
    cleanup_world,
    collapse,
    counts,
    describe_counts,
    determined_hand,
    draw_unobserved,
    map_worlds,
    normalise,
    observe_draw,
    take_from_discard,
    take_from_hand,
)
from .events import (
    EndTurnObservation,
    Event,
    IllegalEvent,
    InconsistentObservation,
    ObserveHand,
    RecordDiscard,
    RecordDraw,
    RecordGain,
    RecordPlay,
    RecordPurchase,
    RecordSupply,
    RecordTrash,
)

OBSERVATION_TYPES = (ObserveHand, RecordDraw, RecordSupply)
COLLAPSE_FLAG = ("max_worlds exceeded: hidden states were collapsed to what is certain; other cards are "
                 "unknown ('?'), so predictions are lower bounds")
SUPPLY_FLAG = "Supply: other players' gains are not tracked; counts may be stale unless recorded with RecordSupply"


class _Untrackable(ValueError):
    """Becomes :class:`IllegalEvent` (with the event index)."""


@lru_cache(maxsize=1)
def shared_engine() -> DominionRuleEngine:
    """One rule engine for every session (it is stateless)."""
    return DominionRuleEngine()


def apply_event(belief: BeliefState, event: Event, index: int, max_worlds: int) -> BeliefState:
    kind = getattr(event, "kind", type(event).__name__)
    try:
        return _Step(belief, max_worlds).apply(event)
    except Contradiction as e:
        raise InconsistentObservation(index, kind, str(e)) from None
    except (_Untrackable, UnknownCardError, TypeError) as e:
        raise IllegalEvent(index, kind, str(e)) from None


class _Step:
    """Mutable working copy used while applying one event; ``result()`` freezes it."""

    def __init__(self, b: BeliefState, max_worlds: int):
        self.eng = shared_engine()
        self.game = b.game.copy()
        self.worlds = b.worlds
        self.flags = list(b.flags)
        self.notes = list(b.notes)
        self.hand_observed = b.hand_observed
        self.unverified_spend = b.unverified_spend
        self.max_worlds = max_worlds

    # helpers -------------------------------------------------------------
    @property
    def t(self) -> TurnState:
        return self.game.turn

    def note(self, text: str) -> None:
        self.notes.append(f"Turn {self.game.turn_number}: {text}" if self.game.turn_number else text)

    def flag(self, text: str) -> None:
        if text not in self.flags:
            self.flags.append(text)

    def canon(self, cards: Any) -> Counter:
        if isinstance(cards, str) or not all(isinstance(c, str) for c in cards):
            raise _Untrackable("cards must be a list of card names")
        return Counter(self.eng.registry.resolve(c) for c in cards)

    def sync_hand(self) -> list[str] | None:
        hand = determined_hand(self.worlds)
        self.t.hand = list(hand) if hand is not None else []
        return hand

    def require_hand(self, what: str) -> list[str]:
        hand = self.sync_hand()
        if hand is None:
            raise _Untrackable(f"cannot {what}: the current hand is not determined (record ObserveHand first)")
        return hand

    def set_worlds(self, worlds: tuple[World, ...], collapsed: bool = False) -> None:
        self.worlds = worlds
        if collapsed:
            self.flag(COLLAPSE_FLAG)
        self.sync_hand()

    def check_owned(self, extra: Counter, with_hand: bool) -> None:
        owned = self.game.owned_cards
        if owned is None:
            return
        visible = Counter(self.t.in_play) + extra
        if with_hand:
            visible += Counter({c: n for c, n in collapse(self.worlds).hand if c != UNKNOWN_CARD})
        over = [f"{visible[c]} {c} (only {owned.get(c, 0)} owned)" for c in sorted(extra) if visible[c] > owned.get(c, 0)]
        if over:
            raise Contradiction("more copies than exist in your deck: " + ", ".join(over))

    def owned_add(self, card: str, n: int) -> None:
        owned = self.game.owned_cards
        if owned is None:
            return
        if owned.get(card, 0) + n < 0:
            raise Contradiction(f"you own no {card} to remove")
        owned[card] = owned.get(card, 0) + n
        if not owned[card]:
            del owned[card]

    def take_supply(self, card: str) -> None:
        if card not in self.game.supply:
            raise Contradiction(f"{card} is not in this game's Supply")
        if self.game.supply[card] <= 0:
            raise Contradiction(f"the {card} pile is empty")
        self.game.supply[card] -= 1

    def unobserved_draw(self, n: int) -> None:
        worlds, collapsed = draw_unobserved(self.worlds, n, self.max_worlds)
        self.set_worlds(worlds, collapsed)
        self.t.pending_draws = max(0, self.t.pending_draws - n)
        self.hand_observed = False
        hand = determined_hand(self.worlds)
        if hand is None:
            self.flag(f"Turn {self.game.turn_number}: {n} card(s) drawn but not observed; "
                      f"the hand is a probability distribution over {len(self.worlds)} hidden state(s)")
        else:
            self.note(f"{n} unobserved card(s) drawn; hand deduced exactly: {describe_counts(Counter(hand))}.")

    def engine_step(self, action: Action) -> None:
        reasons = self.eng.check(self.game, action)
        if reasons:
            raise Contradiction(f"the rules do not allow '{action.describe()}' here: " + " ".join(reasons))
        before = self.game
        result = self.eng.apply(before, action)
        self.game = result.state
        self.note(" ".join(result.notes))
        if isinstance(action, (PlayAction, PlayTreasure, PlayAllTreasures)):
            hand = counts(self.game.turn.hand)
            self.set_worlds(map_worlds(self.worlds, lambda w: replace(w, hand=hand)))
        if isinstance(action, PlayAction):
            gain = self.eng.card(action.throne_target or action.card).gain
            if gain is not None:
                gained = [gain.card] * (before.supply.get(gain.card, 0) - self.game.supply.get(gain.card, 0))
                if gain.destination == "deck":
                    self.set_worlds(map_worlds(self.worlds, lambda w: replace(w, draw=put_on_top(w.draw, gained))))
                else:
                    self.set_worlds(map_worlds(self.worlds,
                                               lambda w: replace(w, discard=add_to_discard(w.discard, gained))))

    def to_discard(self, cards: list[str]) -> None:
        self.set_worlds(map_worlds(self.worlds, lambda w: replace(w, discard=add_to_discard(w.discard, cards))))

    def drop_overcounted(self) -> None:
        """Drop worlds with more known copies of a card than I own (a "?" matched to a card whose every copy
        is known to be elsewhere); none left -> Contradiction."""
        owned = self.game.owned_cards
        if owned is None:
            return
        in_play = Counter(self.t.in_play)

        def over(w: World) -> list[str]:
            known = in_play + w.hand_counter + (w.zone_counter("draw_pile") or Counter()) \
                + (w.zone_counter("discard_pile") or Counter())
            return [f"{n} {c} (only {owned.get(c, 0)} owned)" for c, n in sorted(known.items())
                    if c != UNKNOWN_CARD and n > owned.get(c, 0)]

        kept = [w for w in self.worlds if not over(w)]
        if not kept:
            raise Contradiction("more copies than exist in your deck (counting cards known to be in other zones): "
                                + ", ".join(over(self.worlds[0])))
        if len(kept) < len(self.worlds):
            self.worlds = normalise(kept)

    def result(self) -> BeliefState:
        self.drop_overcounted()
        self.sync_hand()
        return BeliefState(self.game, self.worlds, tuple(self.flags), tuple(self.notes),
                           self.hand_observed, self.unverified_spend)

    # dispatch ------------------------------------------------------------
    def apply(self, e: Event) -> BeliefState:
        handlers = {
            ObserveHand: self._observe_hand, RecordDraw: self._record_draw, RecordPlay: self._record_play,
            RecordPurchase: self._record_purchase, RecordGain: self._record_gain, RecordTrash: self._record_trash,
            RecordDiscard: self._record_discard, RecordSupply: self._record_supply, EndTurnObservation: self._end_turn,
        }
        handler = handlers.get(type(e))  # type: ignore[arg-type]
        if handler is None:
            raise _Untrackable(f"unknown event type {type(e).__name__}")
        if not isinstance(e, RecordSupply) and self.t.unresolved:
            self.note("manual step(s) cleared - record their outcome with Gain/Trash/Discard/Draw events: "
                      + " | ".join(self.t.unresolved))
            self.t.unresolved = []
        if self.t.pending_draws and not isinstance(e, OBSERVATION_TYPES):
            self.note(f"{self.t.pending_draws} pending draw(s) were not recorded before '{e.kind}'.")
            self.unobserved_draw(self.t.pending_draws)
        handler(e)
        return self.result()

    def _observe_hand(self, e: ObserveHand) -> None:
        observed = self.canon(e.cards)
        self.check_owned(observed, with_hand=False)
        self.set_worlds(*observe_draw(self.worlds, self.t.pending_draws, observed, True, self.max_worlds))
        self.t.pending_draws = 0
        self.hand_observed = True

    def _record_draw(self, e: RecordDraw) -> None:
        if (e.cards is None) == (e.count is None):
            raise _Untrackable("give exactly one of 'cards' (observed) or 'count' (unobserved)")
        if e.count is not None:
            if not isinstance(e.count, int) or isinstance(e.count, bool) or e.count < 1:
                raise _Untrackable("count must be a positive integer")
            self.unobserved_draw(e.count)
            return
        observed = self.canon(e.cards)
        if not observed:
            raise _Untrackable("no cards given")
        self.check_owned(observed, with_hand=True)
        n = sum(observed.values())
        self.set_worlds(*observe_draw(self.worlds, n, observed, False, self.max_worlds))
        self.t.pending_draws = max(0, self.t.pending_draws - n)

    def _record_play(self, e: RecordPlay) -> None:
        card = self.eng.registry.resolve(e.card)
        target = None if e.throne_target is None else self.eng.registry.resolve(e.throne_target)
        self.require_hand(f"track playing {card}")
        cdef = self.eng.card(card)
        if cdef.is_treasure and not cdef.is_action:
            if self.t.phase == Phase.ACTION:
                self.engine_step(EndActionPhase())
            self.engine_step(PlayTreasure(card))
        else:
            self.engine_step(PlayAction(card, target, e.trash_copper))

    def _record_purchase(self, e: RecordPurchase) -> None:
        card = self.eng.registry.resolve(e.card)
        if self.t.phase == Phase.ACTION:
            self.engine_step(EndActionPhase())
        hand = self.sync_hand()
        if hand is None or self.unverified_spend:
            self._unverified_purchase(card)
            return
        if self.eng.check(self.game, Buy(card)) and not self.t.bought and any(
                self.eng.card(c).is_treasure for c in hand):
            self.engine_step(PlayAllTreasures())
        reasons = self.eng.check(self.game, Buy(card))
        if reasons:
            raise Contradiction(f"impossible resource change - cannot buy {card}: " + " ".join(reasons))
        self.engine_step(Buy(card))
        self.to_discard([card])

    def _unverified_purchase(self, card: str) -> None:
        """Buy while the hand is undetermined: affordability is evidence about the hand (Bayes, 0/1 likelihood)."""
        cdef, t = self.eng.card(card), self.t
        if t.buys < 1:
            raise Contradiction("impossible purchase: no Buys remaining")
        need = self.unverified_spend + cdef.cost

        def affordable(w: World) -> bool:
            h = w.hand_counter
            if h[UNKNOWN_CARD]:
                return True
            coins = t.coins + sum(self.eng.card(c).treasure_value * n for c, n in h.items())
            if h["Silver"] and not t.silver_played:
                coins += t.merchants_played
            return coins >= need

        kept = [w for w in self.worlds if affordable(w)]
        if not kept:
            raise Contradiction(f"impossible resource change: no possible hand pays ${need} for {card}")
        self.take_supply(card)
        self.set_worlds(normalise(kept))
        t.buys -= 1
        t.bought.append(card)
        self.owned_add(card, 1)
        self.unverified_spend = need
        self.to_discard([card])
        self.flag(f"Turn {self.game.turn_number}: bought {card} while the hand was not determined; coins are "
                  "not tracked this turn (affordability was used as evidence about the hand)")

    def _record_gain(self, e: RecordGain) -> None:
        card = self.eng.registry.resolve(e.card)
        if e.destination not in ("discard", "hand", "deck"):
            raise _Untrackable("destination must be 'discard', 'hand' or 'deck'")
        self.take_supply(card)
        self.owned_add(card, 1)
        if e.destination == "discard":
            self.to_discard([card])
        elif e.destination == "deck":
            self.set_worlds(map_worlds(self.worlds, lambda w: replace(w, draw=put_on_top(w.draw, [card]))))
        else:
            self.set_worlds(map_worlds(self.worlds, lambda w: replace(w, hand=counts(w.hand_counter + Counter([card])))))
        self.note(f"gained {card} to {e.destination}.")

    def _record_trash(self, e: RecordTrash) -> None:
        card = self.eng.registry.resolve(e.card)
        if e.zone == "hand":
            self.set_worlds(take_from_hand(self.worlds, Counter([card])))
        elif e.zone == "in_play":
            if card not in self.t.in_play:
                raise Contradiction(f"there is no {card} in play")
            self.t.in_play.remove(card)
        elif e.zone == "discard":
            self.set_worlds(take_from_discard(self.worlds, card))
        else:
            raise _Untrackable("zone must be 'hand', 'in_play' or 'discard'")
        self.owned_add(card, -1)
        self.game.trash.append(card)
        self.note(f"trashed {card} from {e.zone}.")

    def _record_discard(self, e: RecordDiscard) -> None:
        cards = self.canon(e.cards)
        self.set_worlds(take_from_hand(self.worlds, cards))
        self.to_discard(expand_counts(cards))
        self.note(f"discarded {describe_counts(cards)} from hand.")

    def _record_supply(self, e: RecordSupply) -> None:
        pile = self.eng.registry.resolve(e.pile)
        if pile not in self.game.supply:
            raise Contradiction(f"{pile} is not in this game's Supply")
        if not isinstance(e.count, int) or isinstance(e.count, bool) or e.count < 0:
            raise _Untrackable("count must be a non-negative integer")
        start = self.eng.ruleset.starting_pile_size(pile, self.game.num_players, self.eng.registry)
        if e.count > start:
            raise Contradiction(f"the {pile} pile starts with only {start} cards (observed {e.count})")
        self.game.supply[pile] = e.count
        self.note(f"Supply {pile} = {e.count} (observed; not used to infer your own gains).")

    def _end_turn(self, e: EndTurnObservation) -> None:
        if self.t.phase == Phase.ACTION:
            self.engine_step(EndActionPhase())
        in_play = list(self.t.in_play)
        self.engine_step(EndTurn())

        self.set_worlds(map_worlds(self.worlds, lambda w: cleanup_world(w, in_play)))
        rules = self.eng.ruleset
        self.game.turn = TurnState(phase=Phase.ACTION, hand=[], actions=rules.start_actions, buys=rules.start_buys,
                                   coins=rules.start_coins, pending_draws=rules.hand_size)
        if self.game.turn_number is not None:
            self.game.turn_number += 1
        self.hand_observed = False
        self.unverified_spend = 0
        if self.game.num_players > 1:
            self.flag(SUPPLY_FLAG)

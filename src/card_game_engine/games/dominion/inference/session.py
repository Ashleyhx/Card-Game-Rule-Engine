"""Observation-tracking session for one real Dominion game (my seat).

A :class:`Session` is just a config plus an event log; the belief is a deterministic
replay from the initial state. Recording returns a new Session and validates the event
by applying it: contradictions raise :class:`InconsistentObservation`, malformed or
untrackable events raise :class:`IllegalEvent`, and the session is left unchanged.
Resource and legality changes (play, buy, end turn) go through ``DominionRuleEngine``.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import lru_cache
from typing import Any, Mapping

from ..planning import PlanConfig, PlanResult
from ..planning import plan_next_action as _plan_on_state
from ..rules.actions import Action, Buy, EndActionPhase, EndTurn, PlayAction, PlayAllTreasures, PlayTreasure
from ..rules.cards import UnknownCardError
from ..rules.engine import DominionRuleEngine
from ..rules.state import GameState, Phase, TurnState, state_from_dict
from ..rules.validation import validate_state
from ..rules.zones import (
    UNKNOWN_CARD,
    DiscardKnowledge,
    DrawPileKnowledge,
    add_to_discard,
    cleanup_zones,
    derive_zone_knowledge,
    put_on_top,
)
from .belief import (
    ENUMERATION_LIMIT,
    BeliefState,
    Contradiction,
    World,
    collapse,
    counts,
    describe_counts,
    determined_hand,
    draw_unobserved,
    expand,
    fraction_str,
    map_worlds,
    normalise,
    observe_draw,
    outcome_count,
    predictive,
    sample_predictive,
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
    event_from_dict,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "num_players": 2, "kingdom": None, "player_name": "You", "max_worlds": 5000, "snapshot": None,
}
OBSERVATION_TYPES = (ObserveHand, RecordDraw, RecordSupply)
COLLAPSE_FLAG = ("max_worlds exceeded: hidden states were collapsed to what is certain; other cards are "
                 "unknown ('?'), so predictions are lower bounds")
SUPPLY_FLAG = "Supply: other players' gains are not tracked; counts may be stale unless recorded with RecordSupply"
MC_SAMPLES = 20_000


class HandNotDetermined(ValueError):
    """Planning needs the current hand; it is not determined by the recorded observations."""


class _Untrackable(ValueError):
    """Becomes :class:`IllegalEvent` (with the event index)."""


@lru_cache(maxsize=1)
def _engine() -> DominionRuleEngine:
    return DominionRuleEngine()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Session:
    """Config + event log. ``to_dict``/``from_dict`` round-trip; the belief is replayed (and cached)."""

    config: dict[str, Any]
    events: tuple[Event, ...] = ()
    domain: str = "dominion"
    _belief: BeliefState | None = field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"domain": self.domain, "config": copy.deepcopy(self.config),
                "events": [e.to_dict() for e in self.events]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Session":
        """Rebuild and validate (by replay) a session from ``to_dict`` output."""
        if not isinstance(data, Mapping) or not isinstance(data.get("events", []), list):
            raise ValueError("not a session: expected a JSON object with an 'events' list")
        if data.get("domain", "dominion") != "dominion":
            raise ValueError(f"not a Dominion session: {data.get('domain')!r}")
        session = cls(_check_config(data.get("config")), tuple(event_from_dict(e) for e in data.get("events", [])))
        infer_belief_state(session)
        return session


def _check_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    given = copy.deepcopy(dict(config or {}))
    unknown = sorted(set(given) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"unknown session config key(s): {', '.join(unknown)} (allowed: {', '.join(DEFAULT_CONFIG)})")
    cfg = {**DEFAULT_CONFIG, **given}
    mw = cfg["max_worlds"]
    if not isinstance(mw, int) or isinstance(mw, bool) or mw < 1:
        raise ValueError("max_worlds must be a positive integer")
    return cfg


def _initial_belief(cfg: dict[str, Any]) -> BeliefState:
    eng = _engine()
    if cfg["snapshot"] is not None:
        state = state_from_dict(copy.deepcopy(cfg["snapshot"]), eng.registry, eng.ruleset)
        errors = [i.message for i in validate_state(state, eng) if i.severity == "error"]
        if errors:
            raise ValueError("Snapshot is inconsistent: " + " ".join(errors))
        draw, discard, notes = derive_zone_knowledge(state, eng.registry)
        game = replace(state, draw_pile=None, discard_pile=None, deck_count=None, discard_count=None)
        world = World(Fraction(1), draw, discard, counts(state.turn.hand))
        flags = ("started_from_snapshot: history before the snapshot is unknown; only what it states or implies is used",)
        return BeliefState(game, (world,), flags, tuple(notes), hand_observed=True)
    rules = eng.ruleset
    snap = {
        "schema_version": 1, "num_players": cfg["num_players"], "turn_number": 1,
        "kingdom": list(cfg["kingdom"] or rules.first_game_kingdom),
        "player": {"name": cfg["player_name"], "phase": "action", "hand": [], "pending_draws": rules.hand_size,
                   "owned_cards": dict(rules.starting_deck)},
    }
    game = replace(state_from_dict(snap, eng.registry, rules), assumptions=[])
    deck = DrawPileKnowledge(rest=rules.starting_deck, source="initial:starting-deck")
    note = (f"Starting deck {describe_counts(Counter(rules.starting_deck))} shuffled: composition known, order "
            f"unknown; opening hand of {rules.hand_size} pending.")
    return BeliefState(game, (World(Fraction(1), deck, DiscardKnowledge(source="initial")),), (), (note,))


def initialize_session(config: Mapping[str, Any] | None = None) -> Session:
    """Start tracking: standard starting deck (or ``config['snapshot']``), turn 1, opening hand pending.

    Config keys: num_players=2, kingdom=None (First Game), player_name, max_worlds=5000, snapshot=None.
    """
    cfg = _check_config(config)
    return Session(cfg, (), "dominion", _initial_belief(cfg))


def infer_belief_state(session: Session) -> BeliefState:
    """Deterministic replay of the event log from the initial state (cached on the session)."""
    if session._belief is None:
        belief = _initial_belief(session.config)
        for i, event in enumerate(session.events):
            belief = _apply(belief, event, i, session.config["max_worlds"])
        object.__setattr__(session, "_belief", belief)
    return session._belief  # type: ignore[return-value]


def record_event(session: Session, event: Event) -> Session:
    """Validate ``event`` against the tracked state and return a new Session that includes it."""
    belief = _apply(infer_belief_state(session), event, len(session.events), session.config["max_worlds"])
    return Session(session.config, session.events + (event,), session.domain, belief)


def record_observation(session: Session, observation: Event) -> Session:
    """:func:`record_event` restricted to observations (ObserveHand, RecordDraw, RecordSupply)."""
    if not isinstance(observation, OBSERVATION_TYPES):
        raise IllegalEvent(len(session.events), getattr(observation, "kind", None),
                           "not an observation (use record_event for actions)")
    return record_event(session, observation)


def _replayed(session: Session, events: tuple[Event, ...]) -> Session:
    new = Session(session.config, events, session.domain)
    infer_belief_state(new)
    return new


def correct_event(session: Session, index: int, event: Event) -> Session:
    """Replace event ``index`` and replay everything (raises if the corrected history is inconsistent)."""
    _check_index(session, index)
    return _replayed(session, session.events[:index] + (event,) + session.events[index + 1:])


def remove_event(session: Session, index: int) -> Session:
    """Drop event ``index`` and replay everything."""
    _check_index(session, index)
    return _replayed(session, session.events[:index] + session.events[index + 1:])


def reset_session(session: Session) -> Session:
    """Same config, empty event log."""
    return _replayed(session, ())


def _check_index(session: Session, index: int) -> None:
    if not 0 <= index < len(session.events):
        raise IndexError(f"event index {index} out of range (0..{len(session.events) - 1})")


# ---------------------------------------------------------------------------
# Applying one event
# ---------------------------------------------------------------------------

def _apply(belief: BeliefState, event: Event, index: int, max_worlds: int) -> BeliefState:
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
        self.eng = _engine()
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
        self.to_discard(expand(cards))
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

        def cleanup(w: World) -> World:
            h = w.hand_counter
            unknown = h.pop(UNKNOWN_CARD, 0)
            return World(w.weight, w.draw, cleanup_zones(w.discard, expand(h), in_play, unknown), ())

        self.set_worlds(map_worlds(self.worlds, cleanup))
        rules = self.eng.ruleset
        self.game.turn = TurnState(phase=Phase.ACTION, hand=[], actions=rules.start_actions, buys=rules.start_buys,
                                   coins=rules.start_coins, pending_draws=rules.hand_size)
        if self.game.turn_number is not None:
            self.game.turn_number += 1
        self.hand_observed = False
        self.unverified_spend = 0
        if self.game.num_players > 1:
            self.flag(SUPPLY_FLAG)


# ---------------------------------------------------------------------------
# Predictions, planner bridge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandPrediction:
    """Distribution of a future hand. Exact ``Fraction`` values unless ``method == "monte_carlo"``."""

    which: str                       # "start_of_turn" (the pending hand) | "next_turn" (after this turn's Clean-up)
    turn: int | None
    size: int
    deterministic: bool
    hand: tuple[str, ...] | None     # the exact hand when deterministic
    distribution: tuple[tuple[tuple[str, ...], Fraction | None, float], ...]  # every hand, most likely first
    expected_coins: Fraction | float
    p_coins_at_least: dict[int, Fraction | float]
    p_at_least_one: dict[str, Fraction | float]
    p_reshuffle: Fraction | float
    worlds: int
    method: str                      # "exact" | "monte_carlo"
    lower_bound: bool                # some cards are unknown ("?", counted as $0)
    notes: tuple[str, ...] = ()
    top: int = 10

    def probability(self, cards: list[str] | tuple[str, ...]) -> Fraction | float:
        key = tuple(sorted(cards))
        for hand, p_exact, p in self.distribution:
            if hand == key:
                return p_exact if p_exact is not None else p
        return Fraction(0) if self.method == "exact" else 0.0

    def to_dict(self) -> dict[str, Any]:
        def num(v: Fraction | float) -> dict[str, Any]:
            return {"p_exact": fraction_str(v), "p": float(v)} if isinstance(v, Fraction) else {"p_exact": None, "p": v}

        shown = self.distribution[: self.top]
        return {
            "which": self.which, "turn": self.turn, "size": self.size, "deterministic": self.deterministic,
            "hand": None if self.hand is None else list(self.hand),
            "distribution": [{"cards": list(c), "p_exact": None if pe is None else fraction_str(pe), "p": p}
                             for c, pe, p in shown],
            "other_probability": max(0.0, 1.0 - sum(p for _, _, p in shown)),
            "outcomes": len(self.distribution),
            "expected_coins": {"exact": fraction_str(self.expected_coins) if isinstance(self.expected_coins, Fraction)
                               else None, "value": float(self.expected_coins)},
            "p_coins_at_least": {str(k): num(v) for k, v in self.p_coins_at_least.items()},
            "p_at_least_one": {k: num(v) for k, v in self.p_at_least_one.items()},
            "p_reshuffle": num(self.p_reshuffle), "worlds": self.worlds, "method": self.method,
            "lower_bound": self.lower_bound, "notes": list(self.notes),
        }


def predict_next_hand(session: Session, top: int = 10) -> HandPrediction:
    """The pending start-of-turn hand, or else the next turn's hand assuming nothing more happens this turn."""
    b = infer_belief_state(session)
    eng, t = _engine(), b.game.turn
    worlds, notes = b.worlds, []
    if t.pending_draws and t.phase == Phase.ACTION and not t.in_play and all(not w.hand for w in worlds):
        which, turn, n = "start_of_turn", b.turn, t.pending_draws
    else:
        if t.pending_draws:
            worlds, _ = draw_unobserved(worlds, t.pending_draws, session.config["max_worlds"])
            notes.append(f"The {t.pending_draws} pending draw(s) are treated as unobserved.")
        in_play = list(t.in_play)

        def cleanup(w: World) -> World:
            h = w.hand_counter
            unknown = h.pop(UNKNOWN_CARD, 0)
            return World(w.weight, w.draw, cleanup_zones(w.discard, expand(h), in_play, unknown), ())

        worlds = map_worlds(worlds, cleanup)
        which, turn, n = "next_turn", None if b.turn is None else b.turn + 1, eng.ruleset.hand_size
        notes.append("Assumes no further plays, draws or gains this turn; hand and play go to the discard pile "
                     "at Clean-up.")
    if outcome_count(worlds, n) <= ENUMERATION_LIMIT:
        exact, p_reshuffle = predictive(worlds, n)
        dist: dict[tuple[str, ...], Fraction | float] = dict(exact)
        method, zero = "exact", Fraction(0)
    else:
        sampled, p_reshuffle = sample_predictive(worlds, n, MC_SAMPLES, seed=0)
        dist, method, zero = dict(sampled), "monte_carlo", 0.0
        notes.append(f"Too many outcomes for exact enumeration: Monte Carlo estimate ({MC_SAMPLES} samples, seed 0).")

    def coins(cards: tuple[str, ...]) -> int:
        return sum(eng.card(c).treasure_value for c in cards if c != UNKNOWN_CARD)

    ordered = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
    names = sorted({c for cards in dist for c in cards if c != UNKNOWN_CARD})
    lower_bound = any(UNKNOWN_CARD in cards for cards in dist)
    if lower_bound:
        notes.append("Some cards are unknown ('?'); coin figures count them as $0 (lower bounds).")
    deterministic = len(dist) == 1 and not lower_bound
    return HandPrediction(
        which=which, turn=turn, size=n, deterministic=deterministic,
        hand=ordered[0][0] if deterministic else None,
        distribution=tuple((c, p if isinstance(p, Fraction) else None, float(p)) for c, p in ordered),
        expected_coins=sum((p * coins(c) for c, p in dist.items()), zero),
        p_coins_at_least={k: sum((p for c, p in dist.items() if coins(c) >= k), zero) for k in (5, 6, 8)},
        p_at_least_one={name: sum((p for c, p in dist.items() if name in c), zero) for name in names},
        p_reshuffle=p_reshuffle, worlds=len(worlds), method=method, lower_bound=lower_bound,
        notes=tuple(notes), top=top,
    )


def to_game_state(session: Session) -> GameState:
    """Observed state + zone knowledge for the planner; passes ``validate_state``.

    One world: exact zones. A mixture: cards whose location is certain stay known and the
    rest become "?" (a documented conservative approximation).
    """
    b = infer_belief_state(session)
    w = b.worlds[0] if len(b.worlds) == 1 else collapse(b.worlds)
    g = b.game.copy()
    hand = b.determined_hand()
    if hand is None:
        g.turn.hand = expand(tuple((c, n) for c, n in w.hand if c != UNKNOWN_CARD))
        g.assumptions.append("The hand is not determined; only cards certainly in hand are listed.")
    g.draw_pile = w.draw if w.draw.size_known else None
    g.deck_count = w.draw.size if w.draw.size_known else None
    g.discard_pile = w.discard if w.discard.size_known else None
    g.discard_count = w.discard.size if w.discard.size_known else None
    if len(b.worlds) > 1:
        g.assumptions.append(f"The belief is a mixture of {len(b.worlds)} hidden states; cards whose location is "
                             "uncertain are shown as unknown ('?') - a conservative approximation.")
    if b.unverified_spend:
        g.turn.bought = []
        g.assumptions.append("Purchases this turn were recorded without coin tracking (hand not determined).")
    return g


def plan_next_action(session: Session, config: PlanConfig | dict[str, Any] | None = None) -> PlanResult:
    """Run the planner on the inferred state; refuses if the current hand is not determined."""
    b = infer_belief_state(session)
    if b.pending_draws or b.determined_hand() is None:
        why = (f"{b.pending_draws} draw(s) are pending" if b.pending_draws
               else f"it is a distribution over {len(b.worlds)} hidden state(s) or contains unknown cards")
        raise HandNotDetermined(f"Cannot plan: the current hand is not determined ({why}). "
                                "Record ObserveHand (or the drawn cards) first; see predict_next_hand for odds.")
    if b.unverified_spend:
        raise HandNotDetermined("Cannot plan: coins this turn were not tracked (purchases recorded before the hand "
                                "was known).")
    cfg = PlanConfig.from_dict(config) if isinstance(config, dict) else config
    return _plan_on_state(to_game_state(session), cfg, _engine())

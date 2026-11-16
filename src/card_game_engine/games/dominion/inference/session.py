"""Observation-tracking session for one real Dominion game (my seat).

A :class:`Session` is just a config plus an event log; the belief is a deterministic
replay from the initial state. Recording returns a new Session and validates the event
by applying it: contradictions raise :class:`InconsistentObservation`, malformed or
untrackable events raise :class:`IllegalEvent`, and the session is left unchanged.
Resource and legality changes (play, buy, end turn) go through ``DominionRuleEngine``.
Applying a single event lives in :mod:`.replay`; this module holds the session API and predictions.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, Mapping

from ..planning import PlanConfig, PlanResult
from ..planning import plan_next_action as _plan_on_state
from ..planning.analysis import MC_SAMPLES
from ..rules.state import GameState, Phase, state_from_dict
from ..rules.validation import validate_state
from ..rules.zones import (
    UNKNOWN_CARD,
    DiscardKnowledge,
    DrawPileKnowledge,
    derive_zone_knowledge,
    expand_counts,
)
from .belief import (
    ENUMERATION_LIMIT,
    BeliefState,
    World,
    cleanup_world,
    collapse,
    counts,
    describe_counts,
    draw_unobserved,
    fraction_str,
    map_worlds,
    outcome_count,
    predictive,
    sample_predictive,
)
from .events import Event, IllegalEvent, event_from_dict
from .replay import OBSERVATION_TYPES, apply_event, shared_engine

DEFAULT_CONFIG: dict[str, Any] = {
    "num_players": 2, "kingdom": None, "player_name": "You", "max_worlds": 5000, "snapshot": None,
}


class HandNotDetermined(ValueError):
    """Planning needs the current hand; it is not determined by the recorded observations."""


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
        if not isinstance(data, Mapping) or not isinstance(data.get("events"), list):
            raise ValueError("not a session: expected a JSON object with an 'events' list")
        if data.get("domain", "dominion") != "dominion":
            raise ValueError(f"not a Dominion session: {data.get('domain')!r}")
        session = cls(_check_config(data.get("config")), tuple(event_from_dict(e) for e in data["events"]))
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
    eng = shared_engine()
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
            belief = apply_event(belief, event, i, session.config["max_worlds"])
        object.__setattr__(session, "_belief", belief)
    return session._belief  # type: ignore[return-value]


def record_event(session: Session, event: Event) -> Session:
    """Validate ``event`` against the tracked state and return a new Session that includes it."""
    belief = apply_event(infer_belief_state(session), event, len(session.events), session.config["max_worlds"])
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


DRAW_POINTS = ("auto", "now", "next_turn")


def draw_source(session: Session, at: str = "auto") -> tuple[str, int | None, tuple[World, ...], list[str]]:
    """The worlds future draws come from: ``(at, turn, worlds, notes)`` with ``at`` resolved.

    ``now``: the current piles (pending draws are the first draws). ``next_turn``: after this turn's
    Clean-up (pending draws treated as unobserved; hand and play to the discard pile). ``auto``: ``now``
    while the start-of-turn hand is still pending, else ``next_turn`` (the choice ``predict_next_hand`` makes).
    """
    if at not in DRAW_POINTS:
        raise ValueError(f"at must be one of {', '.join(DRAW_POINTS)}, not {at!r}")
    b = infer_belief_state(session)
    t = b.game.turn
    if at == "auto":
        start = t.pending_draws and t.phase == Phase.ACTION and not t.in_play and all(not w.hand for w in b.worlds)
        at = "now" if start else "next_turn"
    if at == "now":
        return at, b.turn, b.worlds, []
    worlds, notes = b.worlds, []
    if t.pending_draws:
        worlds, _ = draw_unobserved(worlds, t.pending_draws, session.config["max_worlds"])
        notes.append(f"The {t.pending_draws} pending draw(s) are treated as unobserved.")
    in_play = list(t.in_play)
    worlds = map_worlds(worlds, lambda w: cleanup_world(w, in_play))
    notes.append("Assumes no further plays, draws or gains this turn; hand and play go to the discard pile "
                 "at Clean-up.")
    return at, None if b.turn is None else b.turn + 1, worlds, notes


def predict_next_hand(session: Session, top: int = 10) -> HandPrediction:
    """The pending start-of-turn hand, or else the next turn's hand assuming nothing more happens this turn."""
    b = infer_belief_state(session)
    eng = shared_engine()
    at, turn, worlds, notes = draw_source(session)
    which, n = ("start_of_turn", b.pending_draws) if at == "now" else ("next_turn", eng.ruleset.hand_size)
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
        g.turn.hand = expand_counts(tuple((c, n) for c, n in w.hand if c != UNKNOWN_CARD))
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
    return _plan_on_state(to_game_state(session), cfg, shared_engine())

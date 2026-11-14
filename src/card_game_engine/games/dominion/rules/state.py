"""Typed game-state models and JSON (de)serialisation of snapshots.

A *snapshot* describes the current player's turn as seen from their seat: what is
known for certain (hand, cards in play, Supply, resources) plus optional context
(deck composition, opponents' scores). Unknown information is left unknown -
the engine never invents it.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .cards import CardRegistry, UnknownCardError, load_registry
from .ruleset import Ruleset, load_ruleset
from .zones import DiscardKnowledge, DrawPileKnowledge

SCHEMA_VERSION = 1
MAX_PROJECTED_HANDS = 4
PLANNING_HORIZONS = ("turn", "next_hand", f"hands:<N> (1-{MAX_PROJECTED_HANDS})")


def projected_hand_count(horizon: str) -> int:
    """Number of future hands a horizon projects: ``turn`` -> 0, ``next_hand`` -> 1, ``hands:N`` -> N.

    Single source of truth for planning horizons (snapshot ``planning.horizon`` and the planner's ``PlanConfig``).
    Raises ``ValueError`` for anything else.
    """
    if horizon == "turn":
        return 0
    if horizon == "next_hand":
        return 1
    raw = horizon.split(":", 1)[1] if isinstance(horizon, str) and horizon.startswith("hands:") else ""
    if raw.isdigit() and 1 <= int(raw) <= MAX_PROJECTED_HANDS:
        return int(raw)
    raise ValueError(f"horizon must be one of {', '.join(PLANNING_HORIZONS)} (got {horizon!r})")


class Phase(str, Enum):
    ACTION = "action"
    BUY = "buy"
    TURN_OVER = "turn_over"


class SnapshotError(ValueError):
    """A snapshot could not be parsed. ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid snapshot:\n  - " + "\n  - ".join(errors))


@dataclass
class TurnState:
    phase: Phase
    hand: list[str]
    in_play: list[str] = field(default_factory=list)
    actions: int = 1
    buys: int = 1
    coins: int = 0
    bought: list[str] = field(default_factory=list)
    merchants_played: int = 0
    silver_played: bool = False
    pending_draws: int = 0
    unresolved: list[str] = field(default_factory=list)


@dataclass
class OpponentInfo:
    name: str
    vp: int | None = None
    owned_cards: dict[str, int] | None = None


@dataclass
class GameState:
    num_players: int
    kingdom: list[str]
    supply: dict[str, int]
    turn: TurnState
    player_name: str = "You"
    turn_number: int | None = None
    trash: list[str] = field(default_factory=list)
    owned_cards: dict[str, int] | None = None
    deck_count: int | None = None
    discard_count: int | None = None
    opponents: list[OpponentInfo] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)  # one {"turn", "hand", "bought"} per finished turn
    assumptions: list[str] = field(default_factory=list)
    draw_pile: DrawPileKnowledge | None = None
    discard_pile: DiscardKnowledge | None = None
    planning_horizon: str | None = None  # "turn" | "next_hand" | "hands:N"

    def copy(self) -> "GameState":
        return copy.deepcopy(self)

    def empty_piles(self) -> list[str]:
        return [name for name, n in self.supply.items() if n <= 0]

    # serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        t = self.turn
        out: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "num_players": self.num_players,
            "turn_number": self.turn_number,
            "kingdom": list(self.kingdom),
            "supply": dict(self.supply),
            "trash": list(self.trash),
            "player": {
                "name": self.player_name,
                "phase": t.phase.value,
                "hand": list(t.hand),
                "in_play": list(t.in_play),
                "actions": t.actions,
                "buys": t.buys,
                "coins": t.coins,
                "bought_this_turn": list(t.bought),
                "merchants_played": t.merchants_played,
                "silver_played": t.silver_played,
                "pending_draws": t.pending_draws,
                "unresolved": list(t.unresolved),
                "owned_cards": dict(self.owned_cards) if self.owned_cards is not None else None,
                "deck_count": self.deck_count,
                "discard_count": self.discard_count,
            },
            "opponents": [
                {"name": o.name, "vp": o.vp, "owned_cards": o.owned_cards} for o in self.opponents
            ],
            "history": [dict(h) for h in self.history],
        }
        if self.draw_pile is not None:
            out["player"]["draw_pile"] = self.draw_pile.to_snapshot()
        if self.discard_pile is not None:
            out["player"]["discard_pile"] = self.discard_pile.to_snapshot()
        if self.planning_horizon is not None:
            out["planning"] = {"horizon": self.planning_horizon}
        return out


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_card_counts(value: Any, registry: CardRegistry, field_name: str, errors: list[str]) -> Counter:
    """Accept ``["Copper", "Copper"]`` or ``{"Copper": 2}`` and return canonical counts."""
    counts: Counter = Counter()
    if value is None:
        return counts
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = ((v, 1) for v in value)
    else:
        errors.append(f"{field_name}: expected a list of card names or a {{name: count}} object.")
        return counts
    for name, n in items:
        if not isinstance(name, str):
            errors.append(f"{field_name}: card names must be strings (got {name!r}).")
            continue
        try:
            canonical = registry.resolve(name)
        except UnknownCardError as e:
            errors.append(f"{field_name}: {e}")
            continue
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            errors.append(f"{field_name}: count for {canonical} must be a non-negative integer.")
            continue
        counts[canonical] += n
    return counts


def _expand(counts: Counter) -> list[str]:
    out: list[str] = []
    for name in sorted(counts):
        out.extend([name] * counts[name])
    return out


def _int_field(data: dict, key: str, default: int | None, errors: list[str], where: str, minimum: int = 0):
    if key not in data or data[key] is None:
        return default
    v = data[key]
    if not isinstance(v, int) or isinstance(v, bool):
        errors.append(f"{where}.{key}: must be an integer.")
        return default
    if v < minimum:
        errors.append(f"{where}.{key}: must be >= {minimum}.")
        return default
    return v


def _card_list(value: Any, registry: CardRegistry, where: str, errors: list[str]) -> tuple[str, ...]:
    """An ordered list of card names (order matters, so no ``{name: count}`` form)."""
    if not isinstance(value, list):
        errors.append(f"{where}: expected a list of card names, top card first.")
        return ()
    out: list[str] = []
    for name in value:
        if not isinstance(name, str):
            errors.append(f"{where}: card names must be strings (got {name!r}).")
            continue
        try:
            out.append(registry.resolve(name))
        except UnknownCardError as e:
            errors.append(f"{where}: {e}")
    return tuple(out)


def _check_keys(raw: dict, allowed: tuple[str, ...], where: str, errors: list[str]) -> bool:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        errors.append(f"{where}: unknown key(s) {', '.join(unknown)} (allowed: {', '.join(allowed)}).")
    return not unknown


def _parse_draw_pile(raw: Any, registry: CardRegistry, errors: list[str]) -> DrawPileKnowledge | None:
    where = "player.draw_pile"
    if raw is None:
        return None
    if not isinstance(raw, dict):
        errors.append(f"{where}: expected an object with 'order', or 'top'/'composition'/'unknown_count'.")
        return None
    if not _check_keys(raw, ("order", "top", "composition", "unknown_count"), where, errors):
        return None
    if "order" in raw:
        if len(raw) > 1:
            errors.append(f"{where}: 'order' (fully known order) cannot be combined with other keys.")
            return None
        return DrawPileKnowledge(top=_card_list(raw["order"], registry, f"{where}.order", errors))
    top = _card_list(raw.get("top", []), registry, f"{where}.top", errors)
    rest = parse_card_counts(raw.get("composition"), registry, f"{where}.composition", errors)
    unknown = _int_field(raw, "unknown_count", 0, errors, where)
    return DrawPileKnowledge(top=top, rest=rest, unknown=unknown)


def _parse_discard_pile(raw: Any, registry: CardRegistry, errors: list[str]) -> DiscardKnowledge | None:
    where = "player.discard_pile"
    if raw is None:
        return None
    if not isinstance(raw, dict):
        errors.append(f"{where}: expected an object with 'composition' and optional 'unknown_count'.")
        return None
    if not _check_keys(raw, ("composition", "unknown_count"), where, errors):
        return None
    cards = parse_card_counts(raw.get("composition"), registry, f"{where}.composition", errors)
    return DiscardKnowledge(cards=cards, unknown=_int_field(raw, "unknown_count", 0, errors, where))


def _parse_planning(raw: Any, errors: list[str]) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        errors.append('planning: expected an object such as {"horizon": "turn"}.')
        return None
    if not _check_keys(raw, ("horizon",), "planning", errors):
        return None
    horizon = raw.get("horizon")
    if horizon is not None:
        try:
            projected_hand_count(horizon)
        except ValueError:
            errors.append(f"planning.horizon: must be one of {', '.join(PLANNING_HORIZONS)} (got {horizon!r}).")
            return None
    return horizon


def state_from_dict(
    data: dict[str, Any],
    registry: CardRegistry | None = None,
    ruleset: Ruleset | None = None,
) -> GameState:
    """Parse a snapshot dict into a :class:`GameState`.

    Raises :class:`SnapshotError` listing *all* structural problems at once.
    Rule-level consistency checks live in :mod:`card_game_engine.games.dominion.rules.validation`.
    """
    registry = registry or load_registry()
    ruleset = ruleset or load_ruleset()
    errors: list[str] = []
    assumptions: list[str] = []

    if not isinstance(data, dict):
        raise SnapshotError(["Snapshot must be a JSON object."])

    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        errors.append(f"schema_version {version} is not supported (expected {SCHEMA_VERSION}).")

    num_players = data.get("num_players")
    if not isinstance(num_players, int) or num_players not in ruleset.supported_player_counts:
        errors.append(
            f"num_players must be one of {list(ruleset.supported_player_counts)} (got {num_players!r})."
        )
        num_players = 2  # keep parsing to collect more errors

    # Kingdom ---------------------------------------------------------------
    kingdom_raw = data.get("kingdom")
    kingdom: list[str] = []
    if not isinstance(kingdom_raw, list):
        errors.append("kingdom: expected a list of 10 Kingdom card names.")
    else:
        for name in kingdom_raw:
            try:
                cdef = registry.get(str(name))
            except UnknownCardError as e:
                errors.append(f"kingdom: {e}")
                continue
            if not cdef.kingdom:
                errors.append(f"kingdom: {cdef.name} is a basic card, not a Kingdom card.")
                continue
            if cdef.name in kingdom:
                errors.append(f"kingdom: {cdef.name} listed twice.")
                continue
            kingdom.append(cdef.name)
        if len(kingdom_raw) != ruleset.kingdom_pile_count:
            errors.append(
                f"kingdom: expected exactly {ruleset.kingdom_pile_count} Kingdom cards (got {len(kingdom_raw)})."
            )

    # Supply ----------------------------------------------------------------
    supply: dict[str, int] = {}
    try:
        supply = ruleset.starting_supply(kingdom, num_players, registry)
    except ValueError as e:
        errors.append(str(e))
    supply_raw = data.get("supply")
    if supply_raw is None:
        assumptions.append("Supply counts were not provided; every pile is assumed to be at its starting size.")
    elif not isinstance(supply_raw, dict):
        errors.append("supply: expected an object mapping pile name -> cards remaining.")
    else:
        given: set[str] = set()
        for name, n in supply_raw.items():
            try:
                canonical = registry.resolve(name)
            except UnknownCardError as e:
                errors.append(f"supply: {e}")
                continue
            if canonical not in supply:
                errors.append(f"supply: {canonical} is not a Supply pile in this game (not in the kingdom).")
                continue
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                errors.append(f"supply: count for {canonical} must be a non-negative integer.")
                continue
            supply[canonical] = n
            given.add(canonical)
        missing = [p for p in supply if p not in given]
        if missing:
            assumptions.append(
                "Supply counts not given for: " + ", ".join(missing) + " - assumed full."
            )

    trash = _expand(parse_card_counts(data.get("trash"), registry, "trash", errors))
    turn_number = _int_field(data, "turn_number", None, errors, "snapshot", minimum=1)

    # Player ------------------------------------------------------------------
    p = data.get("player")
    if not isinstance(p, dict):
        errors.append("player: missing or not an object.")
        p = {}
    phase_raw = str(p.get("phase", "action")).lower().strip()
    phase_aliases = {"action": Phase.ACTION, "actions": Phase.ACTION, "a": Phase.ACTION,
                     "buy": Phase.BUY, "b": Phase.BUY, "turn_over": Phase.TURN_OVER}
    phase = phase_aliases.get(phase_raw)
    if phase is None:
        errors.append(f"player.phase: must be 'action', 'buy' or 'turn_over' (got {p.get('phase')!r}).")
        phase = Phase.ACTION

    hand = _expand(parse_card_counts(p.get("hand", []), registry, "player.hand", errors))
    in_play = _expand(parse_card_counts(p.get("in_play", []), registry, "player.in_play", errors))
    bought = _expand(parse_card_counts(p.get("bought_this_turn", []), registry, "player.bought_this_turn", errors))

    actions = _int_field(p, "actions", ruleset.start_actions, errors, "player")
    buys = _int_field(p, "buys", ruleset.start_buys, errors, "player")
    coins = _int_field(p, "coins", ruleset.start_coins, errors, "player")
    pending_draws = _int_field(p, "pending_draws", 0, errors, "player")
    merchants_played = _int_field(p, "merchants_played", in_play.count("Merchant"), errors, "player")
    silver_played = p.get("silver_played")
    if silver_played is None:
        silver_played = "Silver" in in_play
    elif not isinstance(silver_played, bool):
        errors.append("player.silver_played: must be true/false.")
        silver_played = "Silver" in in_play
    unresolved = p.get("unresolved") or []
    if not isinstance(unresolved, list) or not all(isinstance(u, str) for u in unresolved):
        errors.append("player.unresolved: must be a list of strings.")
        unresolved = []

    owned: dict[str, int] | None = None
    if p.get("owned_cards") is not None:
        owned_counter = parse_card_counts(p.get("owned_cards"), registry, "player.owned_cards", errors)
        owned = {k: v for k, v in owned_counter.items() if v > 0}
    else:
        assumptions.append(
            "Your full deck composition (owned_cards) was not provided; VP totals and deck-fit heuristics are unavailable."
        )

    deck_count = _int_field(p, "deck_count", None, errors, "player")
    discard_count = _int_field(p, "discard_count", None, errors, "player")
    draw_pile = _parse_draw_pile(p.get("draw_pile"), registry, errors)
    discard_pile = _parse_discard_pile(p.get("discard_pile"), registry, errors)
    planning_horizon = _parse_planning(data.get("planning"), errors)

    opponents: list[OpponentInfo] = []
    opp_raw = data.get("opponents", [])
    if not isinstance(opp_raw, list):
        errors.append("opponents: expected a list.")
        opp_raw = []
    for i, o in enumerate(opp_raw):
        if not isinstance(o, dict):
            errors.append(f"opponents[{i}]: expected an object.")
            continue
        o_owned = None
        if o.get("owned_cards") is not None:
            oc = parse_card_counts(o["owned_cards"], registry, f"opponents[{i}].owned_cards", errors)
            o_owned = {k: v for k, v in oc.items() if v > 0}
        vp = o.get("vp")
        if vp is not None and (not isinstance(vp, int) or isinstance(vp, bool)):
            errors.append(f"opponents[{i}].vp: must be an integer.")
            vp = None
        opponents.append(OpponentInfo(name=str(o.get("name", f"Opponent {i + 1}")), vp=vp, owned_cards=o_owned))
    history: list[dict[str, Any]] = []
    hist_raw = data.get("history", [])
    if not isinstance(hist_raw, list):
        errors.append("history: expected a list.")
        hist_raw = []
    for i, h in enumerate(hist_raw):
        if not isinstance(h, dict):
            errors.append(f"history[{i}]: expected an object.")
            continue
        history.append({
            "turn": _int_field(h, "turn", None, errors, f"history[{i}]", minimum=1),
            "hand": _expand(parse_card_counts(h.get("hand", []), registry, f"history[{i}].hand", errors)),
            "bought": _expand(parse_card_counts(h.get("bought", []), registry, f"history[{i}].bought", errors)),
        })

    if isinstance(num_players, int) and len(opponents) > num_players - 1:
        errors.append(f"opponents: {len(opponents)} listed but only {num_players - 1} other player(s) in the game.")

    if errors:
        raise SnapshotError(errors)

    turn = TurnState(
        phase=phase,
        hand=hand,
        in_play=in_play,
        actions=actions,
        buys=buys,
        coins=coins,
        bought=bought,
        merchants_played=merchants_played,
        silver_played=bool(silver_played),
        pending_draws=pending_draws,
        unresolved=list(unresolved),
    )
    return GameState(
        num_players=num_players,
        kingdom=kingdom,
        supply=supply,
        turn=turn,
        player_name=str(p.get("name", "You")),
        turn_number=turn_number,
        trash=trash,
        owned_cards=owned,
        deck_count=deck_count,
        discard_count=discard_count,
        opponents=opponents,
        history=history,
        assumptions=assumptions,
        draw_pile=draw_pile,
        discard_pile=discard_pile,
        planning_horizon=planning_horizon,
    )


def new_game_turn_state(
    kingdom: list[str],
    num_players: int,
    hand: list[str],
    registry: CardRegistry | None = None,
    ruleset: Ruleset | None = None,
) -> GameState:
    """Build the state of a player's first turn (starting deck, full Supply)."""
    registry = registry or load_registry()
    ruleset = ruleset or load_ruleset()
    supply = ruleset.starting_supply(kingdom, num_players, registry)
    return GameState(
        num_players=num_players,
        kingdom=list(kingdom),
        supply=supply,
        turn=TurnState(
            phase=Phase.ACTION,
            hand=sorted(registry.resolve(h) for h in hand),
            actions=ruleset.start_actions,
            buys=ruleset.start_buys,
            coins=ruleset.start_coins,
        ),
        turn_number=1,
        owned_cards=dict(ruleset.starting_deck),
        opponents=[OpponentInfo(name=f"Opponent {i + 1}", vp=3) for i in range(num_players - 1)],
    )

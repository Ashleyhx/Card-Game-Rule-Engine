"""Derived, read-only facts about a snapshot that the heuristics use."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..rules import GameState, DominionRuleEngine


@dataclass
class DeckStats:
    total: int
    counts: Counter
    terminals: int
    villages: int
    actions: int
    money: int  # total $ from Treasures

    @property
    def money_density(self) -> float:
        return self.money / self.total if self.total else 0.0


@dataclass
class GameContext:
    stage: str
    stage_reason: str
    provinces_left: int
    provinces_start: int
    empty_piles: list[str]
    low_piles: list[str]
    deck: DeckStats | None
    my_vp: int | None
    opponent_vps: list[int] | None  # None if any opponent's VP is unknown
    unknowns: list[str] = field(default_factory=list)

    @property
    def max_opponent_vp(self) -> int | None:
        if self.opponent_vps is None:
            return None
        return max(self.opponent_vps) if self.opponent_vps else 0

    @property
    def vp_known(self) -> bool:
        return self.my_vp is not None and self.opponent_vps is not None


def deck_stats(owned: dict[str, int], engine: DominionRuleEngine) -> DeckStats:
    counts = Counter(owned)
    terminals = villages = actions = money = 0
    for name, n in counts.items():
        c = engine.card(name)
        if c.is_action:
            actions += n
            if c.is_terminal:
                terminals += n
            if c.is_village:
                villages += n
        money += c.treasure_value * n
    return DeckStats(sum(counts.values()), counts, terminals, villages, actions, money)


def determine_stage(state: GameState, engine: DominionRuleEngine, deck: DeckStats | None) -> tuple[str, str]:
    start = engine.ruleset.starting_province_count(state.num_players)
    left = state.supply.get("Province", start)
    dance_point = math.ceil(start * 5 / 8)
    late_point = math.ceil(start * 4 / 8)
    end_point = math.ceil(start * 2 / 8)
    empties = len(state.empty_piles())
    low = len([p for p, n in state.supply.items() if 0 < n <= 2])
    piles_needed = engine.ruleset.empty_piles_to_end
    if left <= end_point:
        return "endgame", f"only {left} Province(s) left"
    if empties >= 1 and empties + low >= piles_needed:
        return "endgame", f"{empties} empty pile(s) and {low} nearly empty - a 3-pile ending is close"
    if left <= late_point:
        return "late", f"{left}/{start} Provinces left"
    if left <= dance_point:
        return "dance", f"{left}/{start} Provinces left (greening has started)"
    visible = state.turn.hand + state.turn.in_play + state.turn.bought
    developed = (
        left < start
        or (state.turn_number is not None and state.turn_number > 6)
        or (deck is not None and (deck.counts.get("Gold", 0) > 0 or deck.counts.get("Silver", 0) >= 5 or deck.total > 16))
        or "Gold" in visible
    )
    if developed:
        return "mid", "deck is developing; Provinces mostly untouched"
    return "early", "opening turns - building economy"


def build_context(state: GameState, engine: DominionRuleEngine) -> GameContext:
    unknowns: list[str] = []
    deck = deck_stats(state.owned_cards, engine) if state.owned_cards is not None else None
    if deck is None:
        unknowns.append("Deck composition unknown: deck-fit (terminal collision, duplicates) and your VP total are not evaluated.")
    my_vp = engine.victory_points(state.owned_cards) if state.owned_cards is not None else None

    opp_vps: list[int] | None = []
    n_opp = state.num_players - 1
    if len(state.opponents) < n_opp:
        opp_vps = None
    else:
        for o in state.opponents:
            if o.vp is not None:
                opp_vps.append(o.vp)
            elif o.owned_cards is not None:
                opp_vps.append(engine.victory_points(o.owned_cards))
            else:
                opp_vps = None
                break
    if opp_vps is None:
        unknowns.append("Opponent VP unknown: endgame safety checks (winning/losing buys, Penultimate Province) are limited.")

    stage, reason = determine_stage(state, engine, deck)
    start = engine.ruleset.starting_province_count(state.num_players)
    return GameContext(
        stage=stage,
        stage_reason=reason,
        provinces_left=state.supply.get("Province", start),
        provinces_start=start,
        empty_piles=state.empty_piles(),
        low_piles=[p for p, n in state.supply.items() if 0 < n <= 2],
        deck=deck,
        my_vp=my_vp,
        opponent_vps=opp_vps,
        unknowns=unknowns,
    )

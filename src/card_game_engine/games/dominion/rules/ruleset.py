"""Game-level rule constants (setup sizes, end conditions) loaded from bundled JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from .cards import CardRegistry

RULES_DATA_FILE = "rules_base_2e.json"


@dataclass(frozen=True)
class Ruleset:
    ruleset_id: str
    supported_player_counts: tuple[int, ...]
    starting_deck: dict[str, int]
    hand_size: int
    start_actions: int
    start_buys: int
    start_coins: int
    kingdom_pile_count: int
    kingdom_pile_size: int
    base_supply: tuple[str, ...]
    pile_sizes: dict[str, dict[int, int]]
    empty_piles_to_end: int
    no_treasures_after_buying: bool
    first_game_kingdom: tuple[str, ...]
    primary_source: str

    def check_player_count(self, num_players: int) -> None:
        if num_players not in self.supported_player_counts:
            raise ValueError(
                f"V1 supports {min(self.supported_player_counts)}-{max(self.supported_player_counts)} "
                f"players; got {num_players}."
            )

    def starting_pile_size(self, card: str, num_players: int, registry: CardRegistry) -> int:
        """Size of a Supply pile at game start for this player count."""
        self.check_player_count(num_players)
        if card in self.pile_sizes:
            return self.pile_sizes[card][num_players]
        cdef = registry.get(card)
        if cdef.kingdom and cdef.is_victory:
            return self.pile_sizes["victory_kingdom"][num_players]
        if cdef.kingdom:
            return self.kingdom_pile_size
        raise ValueError(f"No pile size known for {card}")

    def starting_supply(self, kingdom: list[str], num_players: int, registry: CardRegistry) -> dict[str, int]:
        piles = list(self.base_supply) + list(kingdom)
        return {name: self.starting_pile_size(name, num_players, registry) for name in piles}

    def starting_province_count(self, num_players: int) -> int:
        return self.pile_sizes["Province"][num_players]


@lru_cache(maxsize=1)
def load_ruleset() -> Ruleset:
    raw = json.loads(
        resources.files("card_game_engine.games.dominion.data").joinpath(RULES_DATA_FILE).read_text(encoding="utf-8")
    )
    pile_sizes = {
        card: {int(k): int(v) for k, v in counts.items()} for card, counts in raw["pile_sizes"].items()
    }
    return Ruleset(
        ruleset_id=raw["ruleset_id"],
        supported_player_counts=tuple(raw["supported_player_counts"]),
        starting_deck=dict(raw["starting_deck"]),
        hand_size=int(raw["hand_size"]),
        start_actions=int(raw["turn_start"]["actions"]),
        start_buys=int(raw["turn_start"]["buys"]),
        start_coins=int(raw["turn_start"]["coins"]),
        kingdom_pile_count=int(raw["kingdom_pile_count"]),
        kingdom_pile_size=int(raw["kingdom_pile_size"]),
        base_supply=tuple(raw["base_supply"]),
        pile_sizes=pile_sizes,
        empty_piles_to_end=int(raw["game_end"]["empty_supply_piles"]),
        no_treasures_after_buying=bool(raw["buy_phase"]["no_treasures_after_buying"]),
        first_game_kingdom=tuple(raw["first_game_kingdom"]),
        primary_source=raw["primary_source"],
    )

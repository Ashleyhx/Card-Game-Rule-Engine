"""Card definitions and the bundled card registry.

All card data is loaded from ``games/dominion/data/cards_base_2e.json`` which is
shipped with the package, so no network access is needed at runtime.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Iterable

CARD_DATA_FILE = "cards_base_2e.json"


class UnknownCardError(ValueError):
    """Raised when a card name cannot be resolved against the registry."""

    def __init__(self, name: str, suggestions: list[str]):
        self.name = name
        self.suggestions = suggestions
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"Unknown card '{name}'.{hint}")


@dataclass(frozen=True)
class CardEffects:
    """Simple, deterministic numeric bonuses ("vanilla" effects)."""

    cards: int = 0
    actions: int = 0
    buys: int = 0
    coins: int = 0


@dataclass(frozen=True)
class GainEffect:
    """A deterministic 'gain a specific card' effect (e.g. Bandit gains a Gold)."""

    card: str
    destination: str  # "discard" | "deck" | "hand"


@dataclass(frozen=True)
class CardDef:
    name: str
    cost: int
    types: tuple[str, ...]
    kingdom: bool
    text: str
    automation: str  # "full" | "partial" | "manual"
    effects: CardEffects = field(default_factory=CardEffects)
    treasure_value: int = 0
    vp: int | None = None
    special: str | None = None
    gain: GainEffect | None = None
    manual_steps: tuple[str, ...] = ()
    rulings: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    gain_limit: int | None = None
    gain_limit_bonus: int | None = None

    # --- type helpers -------------------------------------------------
    @property
    def is_action(self) -> bool:
        return "Action" in self.types

    @property
    def is_treasure(self) -> bool:
        return "Treasure" in self.types

    @property
    def is_victory(self) -> bool:
        return "Victory" in self.types

    @property
    def is_curse(self) -> bool:
        return "Curse" in self.types

    @property
    def is_attack(self) -> bool:
        return "Attack" in self.types

    @property
    def is_reaction(self) -> bool:
        return "Reaction" in self.types

    @property
    def is_terminal(self) -> bool:
        """An Action that gives no +Actions (playing it ends your chain).

        Throne Room is excluded: its terminality depends on its target.
        """
        return self.is_action and self.effects.actions == 0 and self.special != "throne_room"

    @property
    def is_village(self) -> bool:
        """Gives a net gain of Actions (+2 Actions or more)."""
        return self.is_action and self.effects.actions >= 2

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cost": self.cost,
            "types": list(self.types),
            "kingdom": self.kingdom,
            "text": self.text,
            "automation": self.automation,
            "effects": {
                "cards": self.effects.cards,
                "actions": self.effects.actions,
                "buys": self.effects.buys,
                "coins": self.effects.coins,
            },
            "treasure_value": self.treasure_value,
            "vp": self.vp,
            "special": self.special,
            "gain": (
                {"card": self.gain.card, "destination": self.gain.destination}
                if self.gain
                else None
            ),
            "manual_steps": list(self.manual_steps),
            "rulings": list(self.rulings),
            "sources": list(self.sources),
        }


def _card_from_json(raw: dict) -> CardDef:
    eff = raw.get("effects", {})
    gain = raw.get("gain")
    return CardDef(
        name=raw["name"],
        cost=int(raw["cost"]),
        types=tuple(raw["types"]),
        kingdom=bool(raw["kingdom"]),
        text=raw["text"],
        automation=raw.get("automation", "full"),
        effects=CardEffects(
            cards=int(eff.get("cards", 0)),
            actions=int(eff.get("actions", 0)),
            buys=int(eff.get("buys", 0)),
            coins=int(eff.get("coins", 0)),
        ),
        treasure_value=int(raw.get("treasure_value", 0)),
        vp=raw.get("vp"),
        special=raw.get("special"),
        gain=GainEffect(gain["card"], gain["destination"]) if gain else None,
        manual_steps=tuple(raw.get("manual_steps", ())),
        rulings=tuple(raw.get("rulings", ())),
        sources=tuple(raw.get("sources", ())),
        gain_limit=raw.get("gain_limit"),
        gain_limit_bonus=raw.get("gain_limit_bonus"),
    )


class CardRegistry:
    """Lookup table of verified card definitions."""

    def __init__(self, cards: Iterable[CardDef], sources: dict | None = None, meta: dict | None = None):
        self._cards: dict[str, CardDef] = {}
        for c in cards:
            if c.name.lower() in self._cards:
                raise ValueError(f"Duplicate card definition: {c.name}")
            self._cards[c.name.lower()] = c
        self.sources = sources or {}
        self.meta = meta or {}

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip().lower() in self._cards

    def __len__(self) -> int:
        return len(self._cards)

    def get(self, name: str) -> CardDef:
        try:
            return self._cards[name.strip().lower()]
        except KeyError:
            raise UnknownCardError(name, self.suggest(name)) from None

    def resolve(self, name: str) -> str:
        """Return the canonical card name for user input (case-insensitive)."""
        return self.get(name).name

    def suggest(self, name: str, n: int = 3) -> list[str]:
        lowered = name.strip().lower()
        pool = {c.name.lower(): c.name for c in self._cards.values()}
        prefix = [v for k, v in pool.items() if k.startswith(lowered)] if lowered else []
        close = [pool[m] for m in difflib.get_close_matches(lowered, list(pool), n=n, cutoff=0.6)]
        out: list[str] = []
        for s in prefix + close:
            if s not in out:
                out.append(s)
        return out[:n]

    def all(self) -> list[CardDef]:
        return sorted(self._cards.values(), key=lambda c: (not c.kingdom, c.cost, c.name))

    def kingdom_cards(self) -> list[CardDef]:
        return sorted((c for c in self._cards.values() if c.kingdom), key=lambda c: (c.cost, c.name))

    def base_cards(self) -> list[CardDef]:
        return [c for c in self.all() if not c.kingdom]


@lru_cache(maxsize=1)
def load_registry() -> CardRegistry:
    """Load the bundled Base Set 2E card registry (cached)."""
    raw = json.loads(
        resources.files("card_game_engine.games.dominion.data").joinpath(CARD_DATA_FILE).read_text(encoding="utf-8")
    )
    cards = [_card_from_json(c) for c in raw["cards"]]
    meta = {k: v for k, v in raw.items() if k not in ("cards", "sources")}
    return CardRegistry(cards, sources=raw.get("sources", {}), meta=meta)

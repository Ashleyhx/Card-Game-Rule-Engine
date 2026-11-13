"""Parsing of short human card lists like '3 Copper, Silver x2, throne room'."""

from __future__ import annotations

import re

from ..rules import CardRegistry, UnknownCardError

_LEADING = re.compile(r"^(\d+)\s*[x*]?\s+(.+)$", re.I)
_TRAILING = re.compile(r"^(.+?)\s*[x*]\s*(\d+)$", re.I)


def parse_card_list(text: str, registry: CardRegistry) -> list[str]:
    """Parse a comma-separated card list. Raises ValueError with suggestions on bad names."""
    cards: list[str] = []
    text = (text or "").strip()
    if not text or text.lower() in ("none", "-", "nothing"):
        return cards
    for token in re.split(r"[,;]", text):
        token = token.strip()
        if not token:
            continue
        n, name = 1, token
        m = _LEADING.match(token) or None
        if m:
            n, name = int(m.group(1)), m.group(2)
        else:
            m = _TRAILING.match(token)
            if m:
                name, n = m.group(1), int(m.group(2))
        try:
            canonical = registry.resolve(name.strip())
        except UnknownCardError as e:
            raise ValueError(str(e)) from None
        cards.extend([canonical] * n)
    return cards


def parse_pile_overrides(text: str, registry: CardRegistry) -> dict[str, int]:
    """Parse 'Province=5, Smithy=7'."""
    out: dict[str, int] = {}
    text = (text or "").strip()
    if not text:
        return out
    for token in re.split(r"[,;]", text):
        token = token.strip()
        if not token:
            continue
        if "=" not in token and ":" not in token:
            raise ValueError(f"Expected 'Pile=count', got '{token}'.")
        name, _, value = token.replace(":", "=").partition("=")
        try:
            canonical = registry.resolve(name.strip())
        except UnknownCardError as e:
            raise ValueError(str(e)) from None
        if not value.strip().isdigit():
            raise ValueError(f"Count for {canonical} must be a non-negative integer.")
        out[canonical] = int(value.strip())
    return out


def parse_deltas(text: str, registry: CardRegistry) -> dict[str, int]:
    """Parse '+Silver, -2 Estate, +Gold x2' into {card: delta}."""
    out: dict[str, int] = {}
    text = (text or "").strip()
    if not text:
        return out
    for token in re.split(r"[,;]", text):
        token = token.strip()
        if not token:
            continue
        sign = 1
        if token[0] in "+-":
            sign = -1 if token[0] == "-" else 1
            token = token[1:].strip()
        for c in parse_card_list(token, registry):
            out[c] = out.get(c, 0) + sign
    return out

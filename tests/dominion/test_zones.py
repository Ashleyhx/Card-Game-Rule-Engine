"""Zone knowledge: snapshot fields, validation, deterministic zone rules and derivation."""

import pytest

from card_game_engine.games.dominion.cli.commands import load_example
from card_game_engine.games.dominion.rules import (
    DiscardKnowledge, DrawPileKnowledge, InformationLevel, SnapshotError, add_to_discard, cleanup_zones,
    derive_zone_knowledge, has_errors, put_on_top, reshuffle, state_from_dict, validate_state,
)
from tests.dominion.conftest import FIRST_GAME


def snap(planning=None, **player):
    out = {"schema_version": 1, "num_players": 2, "kingdom": FIRST_GAME,
           "player": {"phase": "action", "hand": [], **player}}
    if planning is not None:
        out["planning"] = planning
    return out


def errors_of(engine, **player):
    return [i.message for i in validate_state(state_from_dict(snap(**player)), engine) if i.severity == "error"]


# --- snapshot parsing ---------------------------------------------------------

def test_parse_fully_known_order_canonicalises_names():
    s = state_from_dict(snap(draw_pile={"order": ["gold", "COPPER"]}))
    assert s.draw_pile == DrawPileKnowledge(top=("Gold", "Copper"))
    assert s.draw_pile.level == InformationLevel.FULLY_KNOWN_ORDER


def test_parse_composition_partial_and_discard():
    s = state_from_dict(snap(
        draw_pile={"top": ["Silver"], "composition": {"copper": 3}, "unknown_count": 2},
        discard_pile={"composition": ["Estate", "Estate"], "unknown_count": 1},
        planning={"horizon": "next_hand"},
    ))
    assert s.draw_pile.top == ("Silver",)
    assert s.draw_pile.rest == (("Copper", 3),)
    assert s.draw_pile.unknown == 2 and s.draw_pile.size == 6
    assert s.discard_pile.cards == (("Estate", 2),) and s.discard_pile.unknown == 1
    assert s.planning_horizon == "next_hand"


def test_new_fields_roundtrip():
    original = snap(draw_pile={"top": ["Silver"], "composition": {"Copper": 3}, "unknown_count": 2},
                    discard_pile={"composition": {"Estate": 2}}, planning={"horizon": "turn"})
    s = state_from_dict(original)
    again = state_from_dict(s.to_dict())
    assert (again.draw_pile, again.discard_pile, again.planning_horizon) == (s.draw_pile, s.discard_pile, "turn")
    order = state_from_dict(snap(draw_pile={"order": ["Gold", "Copper"]}))
    assert order.to_dict()["player"]["draw_pile"] == {"order": ["Gold", "Copper"]}


def test_old_snapshots_unchanged():
    s = state_from_dict(snap(hand=["Copper"]))
    assert (s.draw_pile, s.discard_pile, s.planning_horizon) == (None, None, None)
    out = s.to_dict()
    assert "draw_pile" not in out["player"] and "discard_pile" not in out["player"] and "planning" not in out
    for name in ("opening_5_2", "saved_game_turn_3", "unknown_deck_minimal"):
        ex = state_from_dict(load_example(name))
        assert ex.draw_pile is None and ex.discard_pile is None
        assert "planning" not in ex.to_dict()


def test_parse_errors_collected():
    with pytest.raises(SnapshotError) as exc:
        state_from_dict(snap(
            draw_pile={"order": ["Gold"], "composition": {"Copper": 1}},
            discard_pile={"composition": {"Coppper": 1}, "unknown_count": -1},
            planning={"horizon": "forever"},
        ))
    msgs = "\n".join(exc.value.errors)
    for fragment in ["'order'", "Coppper", "discard_pile.unknown_count", "planning.horizon"]:
        assert fragment in msgs


@pytest.mark.parametrize("player,fragment", [
    ({"draw_pile": ["Copper"]}, "player.draw_pile: expected an object"),
    ({"draw_pile": {"cards": []}}, "unknown key(s) cards"),
    ({"draw_pile": {"top": {"Copper": 1}}}, "player.draw_pile.top: expected a list"),
    ({"discard_pile": {"top": ["Gold"]}}, "player.discard_pile: unknown key(s) top"),
])
def test_parse_error_messages(player, fragment):
    with pytest.raises(SnapshotError) as exc:
        state_from_dict(snap(**player))
    assert any(fragment in e for e in exc.value.errors)


# --- validation -----------------------------------------------------------------

def test_validation_size_mismatch(engine):
    errs = errors_of(engine, draw_pile={"composition": {"Copper": 3}}, deck_count=4,
                     discard_pile={"composition": {"Estate": 1}}, discard_count=2)
    assert any("draw_pile holds 3" in e and "deck_count is 4" in e for e in errs)
    assert any("discard_pile holds 1" in e and "discard_count is 2" in e for e in errs)


def test_validation_known_cards_exceed_owned(engine):
    errs = errors_of(engine, hand=["Copper", "Copper"], owned_cards={"Copper": 7, "Estate": 3},
                     draw_pile={"composition": {"Copper": 4}}, discard_pile={"composition": {"Copper": 2}})
    assert errs == ["owned_cards lists 7 Copper but 8 are in your hand, play area and known draw/discard piles."]


def test_validation_zone_card_not_in_game(engine):
    errs = errors_of(engine, draw_pile={"composition": {"Witch": 1}})
    assert any("Witch (in draw/discard pile)" in e for e in errs)


def test_validation_consistent_zones_ok(engine):
    assert not errors_of(engine, hand=["Copper"] * 5, owned_cards={"Copper": 7, "Estate": 3}, deck_count=3,
                         discard_count=2, draw_pile={"composition": {"Estate": 3}},
                         discard_pile={"composition": {"Copper": 2}})


# --- information levels ---------------------------------------------------------

@pytest.mark.parametrize("pile,level", [
    (DrawPileKnowledge(top=("Gold", "Copper")), InformationLevel.FULLY_KNOWN_ORDER),
    (DrawPileKnowledge(), InformationLevel.FULLY_KNOWN_ORDER),
    (DrawPileKnowledge(top=("Gold",), rest={"Copper": 2}), InformationLevel.KNOWN_COMPOSITION),
    (DrawPileKnowledge(rest={"Copper": 2}, unknown=1), InformationLevel.PARTIALLY_KNOWN),
    (DrawPileKnowledge(unknown=5), InformationLevel.PARTIALLY_KNOWN),
    (DrawPileKnowledge(top=("Gold",), size_known=False), InformationLevel.INSUFFICIENT),
])
def test_information_levels(pile, level):
    assert pile.level == level
    assert pile.to_dict()["level"] == level.value


def test_rest_normalised_and_negative_rejected():
    assert DrawPileKnowledge(rest={"Estate": 1, "Copper": 2, "Gold": 0}).rest == (("Copper", 2), ("Estate", 1))
    with pytest.raises(ValueError):
        DiscardKnowledge(cards={"Copper": -1})


# --- deterministic mechanics ---------------------------------------------------

def test_reshuffle_forms_unordered_deck():
    draw, discard = reshuffle(DiscardKnowledge(cards={"Copper": 2}, unknown=1))
    assert draw.top == () and draw.rest == (("Copper", 2),) and draw.unknown == 1
    assert discard.size == 0 and discard.size_known
    unknown_draw, _ = reshuffle(DiscardKnowledge(size_known=False))
    assert unknown_draw.level == InformationLevel.INSUFFICIENT


def test_put_on_top_add_to_discard_and_cleanup():
    assert put_on_top(DrawPileKnowledge(top=("Gold",)), ["Silver"]).top == ("Silver", "Gold")
    unknown = DrawPileKnowledge(size_known=False)
    assert put_on_top(unknown, ["Silver"]) is unknown
    discard = cleanup_zones(DiscardKnowledge(cards={"Silver": 1}), ["Copper", "Estate"], ["Village"], 2)
    assert discard.counter == {"Silver": 1, "Copper": 1, "Estate": 1, "Village": 1} and discard.unknown == 2
    unknown_discard = DiscardKnowledge(size_known=False)
    assert add_to_discard(unknown_discard, ["Gold"]) is unknown_discard


# --- derivation -----------------------------------------------------------------

OWNED = {"Copper": 7, "Estate": 3}


def derive(**player):
    return derive_zone_knowledge(state_from_dict(snap(**player)))


def test_derive_given_draw_pile_and_discard_from_owned():
    draw, discard, notes = derive(hand=["Copper"] * 3, owned_cards=OWNED, draw_pile={"composition": {"Copper": 4}})
    assert draw.source == "given" and draw.rest == (("Copper", 4),)
    assert discard.counter == {"Estate": 3} and discard.source == "derived:owned-minus-visible"
    assert len(notes) == 2


def test_derive_owned_minus_visible():
    draw, discard, _ = derive(hand=["Copper"] * 4, owned_cards=OWNED, discard_pile={"composition": {"Estate": 2}})
    assert draw.rest_counter == {"Copper": 3, "Estate": 1}
    assert draw.source == "derived:owned-minus-visible" and draw.level == InformationLevel.KNOWN_COMPOSITION
    assert discard.source == "given"


def test_derive_empty_discard():
    draw, discard, _ = derive(hand=["Copper"] * 5, owned_cards=OWNED, discard_count=0)
    assert draw.rest_counter == {"Copper": 2, "Estate": 3} and draw.source == "derived:empty-discard"
    assert discard.size == 0 and discard.size_known


def test_no_guessing_when_discard_contents_unknown():
    draw, discard, _ = derive(hand=["Copper"] * 5, owned_cards=OWNED, discard_count=2, deck_count=3)
    assert draw.unknown == 3 and draw.rest == () and draw.source == "derived:deck-count"
    assert discard.unknown == 2 and discard.cards == ()
    draw, discard, _ = derive(hand=["Copper"] * 5, owned_cards=OWNED, discard_count=2)
    assert draw.level == InformationLevel.INSUFFICIENT and discard.unknown == 2


def test_derive_nothing_known():
    draw, discard, notes = derive()
    assert draw.level == InformationLevel.INSUFFICIENT and not discard.size_known
    assert any("unknown" in n for n in notes)


def test_negative_subtraction_falls_through_with_note():
    draw, _, notes = derive(hand=["Copper"] * 5, owned_cards={"Copper": 4}, discard_count=0, deck_count=2)
    assert draw.source == "derived:deck-count"
    assert any("too few Copper" in n for n in notes)


def test_deck_count_conflict_falls_through():
    draw, _, notes = derive(hand=["Copper"] * 5, owned_cards=OWNED, discard_count=0, deck_count=4)
    assert draw.source == "derived:deck-count" and draw.unknown == 4
    assert any("deck_count is 4" in n for n in notes)


def test_engine_keeps_known_zones_consistent_for_saved_snapshots():
    """Buys and Clean-up update known discard contents; a hidden draw drops zone knowledge instead of guessing."""
    from card_game_engine.games.dominion.rules import Buy, DominionRuleEngine, EndTurn, PlayAction

    engine = DominionRuleEngine()
    snap = {
        "num_players": 2,
        "kingdom": ["Cellar", "Market", "Merchant", "Militia", "Mine", "Moat", "Remodel", "Smithy", "Village", "Workshop"],
        "player": {
            "phase": "buy", "hand": ["Estate"], "in_play": ["Silver", "Copper"], "coins": 3,
            "owned_cards": {"Copper": 7, "Estate": 3, "Silver": 1},
            "draw_pile": {"composition": {"Copper": 4, "Estate": 1}},
            "discard_pile": {"composition": {"Copper": 2, "Estate": 1}},
            "deck_count": 5, "discard_count": 3,
        },
    }
    s = state_from_dict(snap)
    s = engine.apply(s, Buy("Silver")).state
    assert s.discard_pile.counter == {"Copper": 2, "Estate": 1, "Silver": 1}
    assert not has_errors(validate_state(s, engine))
    s = engine.apply(s, EndTurn()).state
    assert s.discard_pile.counter == {"Copper": 3, "Estate": 2, "Silver": 2}
    assert s.discard_count == s.discard_pile.size
    assert not has_errors(validate_state(state_from_dict(s.to_dict()), engine))  # saved snapshot reloads cleanly

    snap["player"].update(phase="action", hand=["Village", "Copper"], in_play=[], coins=0,
                          owned_cards={"Copper": 7, "Estate": 2, "Village": 1})
    snap["player"]["draw_pile"] = {"composition": {"Copper": 4, "Estate": 1}}
    snap["player"]["discard_pile"] = {"composition": {"Copper": 2, "Estate": 1}}
    s = engine.apply(state_from_dict(snap), PlayAction("Village")).state
    s = engine.resolve_pending(s, drawn=["Copper"])
    assert s.draw_pile is None and s.discard_pile is None

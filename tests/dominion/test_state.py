"""Snapshot parsing and validation."""

import pytest

from card_game_engine.games.dominion.cli.commands import load_example
from card_game_engine.games.dominion.rules import (
    Buy, EndTurn, PlayAllTreasures, SnapshotError, has_errors, state_from_dict, validate_state,
)
from tests.dominion.conftest import FIRST_GAME


def base(**player):
    return {"schema_version": 1, "num_players": 2, "kingdom": FIRST_GAME,
            "player": {"phase": "action", "hand": [], **player}}


def test_card_names_canonicalised_and_counts_accepted():
    s = state_from_dict(base(hand={"copper": 3, "SMITHY": 1}))
    assert s.turn.hand == ["Copper", "Copper", "Copper", "Smithy"]


def test_missing_supply_recorded_as_assumption():
    s = state_from_dict(base())
    assert any("assumed" in a for a in s.assumptions)
    assert s.supply["Province"] == 8


def test_partial_supply():
    snap = base()
    snap["supply"] = {"Province": 5}
    s = state_from_dict(snap)
    assert s.supply["Province"] == 5
    assert any("not given" in a for a in s.assumptions)


def test_collects_all_errors_at_once():
    snap = {"num_players": 7, "kingdom": ["Smithy", "Smithy", "Feast"],
            "player": {"phase": "dance", "hand": ["Smithee"], "coins": -1}}
    with pytest.raises(SnapshotError) as exc:
        state_from_dict(snap)
    msgs = "\n".join(exc.value.errors)
    for fragment in ["num_players", "listed twice", "Feast", "exactly 10", "player.phase", "Smithee", "coins"]:
        assert fragment in msgs


def test_basic_card_in_kingdom_rejected():
    snap = base()
    snap["kingdom"] = FIRST_GAME[:9] + ["Gold"]
    with pytest.raises(SnapshotError):
        state_from_dict(snap)


def test_supply_pile_not_in_game_rejected():
    snap = base()
    snap["supply"] = {"Witch": 3}
    with pytest.raises(SnapshotError):
        state_from_dict(snap)


def test_too_many_opponents():
    snap = base()
    snap["opponents"] = [{"name": "A"}, {"name": "B"}]
    with pytest.raises(SnapshotError):
        state_from_dict(snap)


def test_roundtrip():
    s = state_from_dict(base(hand=["Copper", "Village"], owned_cards={"Copper": 7, "Estate": 3, "Village": 1}))
    s2 = state_from_dict(s.to_dict())
    assert s2.turn.hand == s.turn.hand and s2.owned_cards == s.owned_cards and s2.supply == s.supply


def test_validation_flags_inconsistencies(engine):
    s = state_from_dict(base(hand=["Copper", "Copper"], in_play=["Silver"], owned_cards={"Copper": 1, "Estate": 3}))
    issues = validate_state(s, engine)
    assert has_errors(issues)
    text = " ".join(i.message for i in issues)
    assert "Treasures are in play but it is the Action phase" in text
    assert "owned_cards lists 1 Copper" in text


def test_validation_supply_over_start(engine):
    snap = base()
    snap["supply"] = {"Province": 9}
    issues = validate_state(state_from_dict(snap), engine)
    assert any("starts with only 8" in i.message for i in issues)


def test_validation_coin_sanity(engine):
    s = state_from_dict(base(phase="buy", in_play=["Gold"], coins=0))
    assert any(i.severity == "warning" and "Coins" in i.message for i in validate_state(s, engine))


def test_valid_snapshot_has_no_errors(engine):
    s = state_from_dict(base(phase="buy", hand=["Estate"], in_play=["Silver", "Copper"], coins=3,
                             owned_cards={"Copper": 7, "Estate": 3, "Silver": 1}))
    assert not has_errors(validate_state(s, engine))


def test_validation_rejects_coins_from_unplayed_treasures(engine):
    s = state_from_dict(base(phase="buy", hand=["Copper", "Copper", "Copper"], coins=3))  # nothing played yet
    issues = validate_state(s, engine)
    assert has_errors(issues) and any("Treasures still in your hand" in i.message for i in issues)


def test_saved_finished_turn_reloads_and_validates(engine):
    s = state_from_dict(load_example("opening_5_2"))
    for a in (PlayAllTreasures(), Buy("Moat"), EndTurn()):
        s = engine.apply(s, a).state
    reloaded = state_from_dict(s.to_dict())  # what `interactive --load` sees
    assert reloaded.turn.phase.value == "turn_over" and reloaded.turn.bought == ["Moat"]
    assert not has_errors(validate_state(reloaded, engine))

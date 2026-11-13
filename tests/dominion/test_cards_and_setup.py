"""Card registry and setup rules verified against the 2E rulebook (see docs/RULES_RESEARCH.md)."""

import pytest

from card_game_engine.games.dominion.rules import UnknownCardError, load_registry, load_ruleset

EXPECTED_KINGDOM_COSTS = {
    "Cellar": 2, "Chapel": 2, "Moat": 2,
    "Harbinger": 3, "Merchant": 3, "Vassal": 3, "Village": 3, "Workshop": 3,
    "Bureaucrat": 4, "Gardens": 4, "Militia": 4, "Moneylender": 4, "Poacher": 4,
    "Remodel": 4, "Smithy": 4, "Throne Room": 4,
    "Bandit": 5, "Council Room": 5, "Festival": 5, "Laboratory": 5, "Library": 5,
    "Market": 5, "Mine": 5, "Sentry": 5, "Witch": 5,
    "Artisan": 6,
}


def test_registry_has_26_kingdom_and_7_basic_cards():
    reg = load_registry()
    assert len(reg.kingdom_cards()) == 26
    assert len(reg.base_cards()) == 7
    assert len(reg) == 33


def test_removed_first_edition_cards_absent():
    reg = load_registry()
    for name in ["Chancellor", "Woodcutter", "Feast", "Spy", "Thief", "Adventurer"]:
        assert name not in reg


@pytest.mark.parametrize("name,cost", sorted(EXPECTED_KINGDOM_COSTS.items()))
def test_kingdom_costs(name, cost):
    assert load_registry().get(name).cost == cost


def test_basic_cards():
    reg = load_registry()
    assert (reg.get("Copper").cost, reg.get("Copper").treasure_value) == (0, 1)
    assert (reg.get("Silver").cost, reg.get("Silver").treasure_value) == (3, 2)
    assert (reg.get("Gold").cost, reg.get("Gold").treasure_value) == (6, 3)
    assert (reg.get("Estate").cost, reg.get("Estate").vp) == (2, 1)
    assert (reg.get("Duchy").cost, reg.get("Duchy").vp) == (5, 3)
    assert (reg.get("Province").cost, reg.get("Province").vp) == (8, 6)
    assert (reg.get("Curse").cost, reg.get("Curse").vp) == (0, -1)


@pytest.mark.parametrize(
    "name,cards,actions,buys,coins",
    [
        ("Village", 1, 2, 0, 0), ("Smithy", 3, 0, 0, 0), ("Laboratory", 2, 1, 0, 0),
        ("Market", 1, 1, 1, 1), ("Festival", 0, 2, 1, 2), ("Council Room", 4, 0, 1, 0),
        ("Moat", 2, 0, 0, 0), ("Witch", 2, 0, 0, 0), ("Militia", 0, 0, 0, 2), ("Vassal", 0, 0, 0, 2),
        ("Poacher", 1, 1, 0, 1), ("Merchant", 1, 1, 0, 0), ("Harbinger", 1, 1, 0, 0),
        ("Sentry", 1, 1, 0, 0), ("Cellar", 0, 1, 0, 0),
    ],
)
def test_vanilla_effects(name, cards, actions, buys, coins):
    e = load_registry().get(name).effects
    assert (e.cards, e.actions, e.buys, e.coins) == (cards, actions, buys, coins)


def test_types():
    reg = load_registry()
    for attack in ["Bandit", "Bureaucrat", "Militia", "Witch"]:
        assert reg.get(attack).is_attack
    assert reg.get("Moat").is_reaction
    assert reg.get("Gardens").is_victory and not reg.get("Gardens").is_action
    assert reg.get("Smithy").is_terminal
    assert not reg.get("Laboratory").is_terminal
    assert reg.get("Village").is_village and reg.get("Festival").is_village


def test_name_resolution_is_case_insensitive_and_suggests():
    reg = load_registry()
    assert reg.resolve("throne room") == "Throne Room"
    with pytest.raises(UnknownCardError) as exc:
        reg.get("Smithe")
    assert "Smithy" in exc.value.suggestions


@pytest.mark.parametrize(
    "players,copper,victory,curse",
    [(2, 46, 8, 10), (3, 39, 12, 20), (4, 32, 12, 30)],
)
def test_supply_sizes(players, copper, victory, curse):
    reg, rules = load_registry(), load_ruleset()
    kingdom = list(rules.first_game_kingdom)
    supply = rules.starting_supply(kingdom, players, reg)
    assert supply["Copper"] == copper
    assert supply["Silver"] == 40 and supply["Gold"] == 30
    assert supply["Estate"] == supply["Duchy"] == supply["Province"] == victory
    assert supply["Curse"] == curse
    assert all(supply[k] == 10 for k in kingdom)
    assert len(supply) == 17


def test_gardens_pile_scales_like_victory_cards():
    reg, rules = load_registry(), load_ruleset()
    assert rules.starting_pile_size("Gardens", 2, reg) == 8
    assert rules.starting_pile_size("Gardens", 4, reg) == 12


def test_unsupported_player_count():
    with pytest.raises(ValueError):
        load_ruleset().check_player_count(5)


def test_every_card_cites_a_source():
    reg = load_registry()
    for c in reg.all():
        assert c.sources, c.name
        for s in c.sources:
            assert s in reg.sources, (c.name, s)

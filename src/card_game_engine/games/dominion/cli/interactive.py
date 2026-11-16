"""Interactive snapshot builder and turn-by-turn assistant.

Input/output functions are injectable so the whole flow is testable.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

from ..policy import HeuristicPolicy
from ..rules import (
    DominionRuleEngine, GameState, Phase, SnapshotError, TurnState, has_errors, state_from_dict, validate_state,
)
from .parsing import parse_card_list, parse_deltas, parse_pile_overrides
from .render import fmt_cards, render_advice, render_issues


class Quit(Exception):
    pass


class Prompter:
    def __init__(self, input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print):
        self._input = input_fn
        self.say = print_fn

    def ask(self, prompt: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        try:
            raw = self._input(f"{prompt}{suffix}: ")
        except EOFError:
            raise Quit() from None
        raw = (raw or "").strip()
        if raw.lower() in ("q", "quit", "exit"):
            raise Quit()
        return raw or default

    def ask_int(self, prompt: str, default: int | None, minimum: int = 0, maximum: int | None = None,
                optional: bool = False) -> int | None:
        while True:
            raw = self.ask(prompt, "" if default is None else str(default))
            if raw == "" and optional:
                return None
            try:
                v = int(raw)
            except ValueError:
                self.say("  Please enter a whole number" + (" (or leave blank)" if optional else "") + ".")
                continue
            if v < minimum or (maximum is not None and v > maximum):
                self.say(f"  Must be between {minimum} and {maximum if maximum is not None else 'infinity'}.")
                continue
            return v

    def ask_choice(self, prompt: str, choices: list[str], default: str) -> str:
        while True:
            raw = self.ask(f"{prompt} ({'/'.join(choices)})", default).lower()
            for c in choices:
                if raw == c or (raw and c.startswith(raw)):
                    return c
            self.say(f"  Choose one of: {', '.join(choices)}")

    def ask_cards(self, prompt: str, registry, default: str = "") -> list[str]:
        while True:
            raw = self.ask(prompt, default)
            try:
                return parse_card_list(raw, registry)
            except ValueError as e:
                self.say(f"  {e}")

    def ask_yes(self, prompt: str, default: bool = True) -> bool:
        return self.ask_choice(prompt, ["y", "n"], "y" if default else "n") == "y"


def build_snapshot(p: Prompter, engine: DominionRuleEngine) -> dict:
    """Ask questions and return a snapshot dict (schema v1). Never invents unknown facts."""
    reg, rules = engine.registry, engine.ruleset
    p.say("Build a snapshot of your current turn. Press Enter to accept [defaults]; type 'q' to quit.")
    p.say("Card lists accept forms like: 3 Copper, Silver x2, throne room")
    n = p.ask_int("Number of players", 2, minimum=2, maximum=4)

    preset = p.ask_choice("Kingdom: use the rulebook's 'First Game' set, or enter your own 10 cards?",
                          ["first", "custom"], "first")
    if preset == "first":
        kingdom = list(rules.first_game_kingdom)
    else:
        names = ", ".join(c.name for c in reg.kingdom_cards())
        p.say(f"  Kingdom cards: {names}")
        while True:
            kingdom = p.ask_cards("Enter the 10 Kingdom cards", reg)
            bad = [c for c in kingdom if not reg.get(c).kingdom]
            if bad:
                p.say(f"  Not Kingdom cards: {', '.join(bad)}")
            elif len(set(kingdom)) != 10 or len(kingdom) != 10:
                p.say(f"  Need exactly 10 different Kingdom cards (got {len(set(kingdom))}).")
            else:
                break
    p.say("  Kingdom: " + ", ".join(kingdom))

    turn_number = p.ask_int("Your turn number (1 or 2 = quick start from the starting deck; blank if unknown)",
                            None, minimum=1, optional=True)
    if turn_number in (1, 2):
        return _opening_snapshot(p, engine, n, kingdom, turn_number)

    supply = rules.starting_supply(kingdom, n, reg)
    if not p.ask_yes("Are all Supply piles still full (start of game)?", default=False):
        p.say("  Enter the piles that changed, e.g. 'Province=5, Silver=34, Smithy=7'.")
        p.say("  Piles you don't mention are recorded as full - an explicit assumption in the output.")
        while True:
            try:
                overrides = parse_pile_overrides(p.ask("Changed piles", ""), reg)
                unknown = [k for k in overrides if k not in supply]
                if unknown:
                    p.say(f"  Not in this game's Supply: {', '.join(unknown)}")
                    continue
                supply.update(overrides)
                break
            except ValueError as e:
                p.say(f"  {e}")

    phase = p.ask_choice("Phase", ["action", "buy"], "action")
    hand = p.ask_cards("Cards in your hand", reg)
    in_play = p.ask_cards("Cards already played this turn", reg, "none")

    player: dict = {"phase": phase, "hand": hand, "in_play": in_play}
    if in_play:
        # Only cards already played produce Actions/Buys/coins; with nothing in play the turn-start values apply.
        player["actions"] = p.ask_int("Actions remaining", 0)
        player["buys"] = p.ask_int("Buys remaining", 1)
        default_coins = sum(reg.get(c).treasure_value for c in in_play)
        player["coins"] = p.ask_int("Coins ($) from cards already played (not Treasures still in hand)", default_coins)
    if phase == "buy":
        player["bought_this_turn"] = p.ask_cards("Cards already bought this turn", reg, "none")

    mode = p.ask_choice(
        "Your whole deck (for VP and deck-fit advice): skip, starting deck + changes, or full list?",
        ["skip", "changes", "full"], "changes" if turn_number else "skip",
    )
    if mode == "changes":
        owned = dict(rules.starting_deck)
        p.say("  Starting deck is 7 Copper + 3 Estate. Enter changes like '+Silver x2, +Smithy, -Estate'.")
        while True:
            try:
                for c, d in parse_deltas(p.ask("Changes", ""), reg).items():
                    owned[c] = owned.get(c, 0) + d
                if any(v < 0 for v in owned.values()):
                    p.say("  That removes more cards than you own; try again.")
                    owned = dict(rules.starting_deck)
                    continue
                break
            except ValueError as e:
                p.say(f"  {e}")
        player["owned_cards"] = {k: v for k, v in owned.items() if v > 0}
    elif mode == "full":
        player["owned_cards"] = p.ask_cards("Every card you own (hand, play, deck, discard)", reg)

    opponents = []
    for i in range(n - 1):
        vp = p.ask_int(f"Opponent {i + 1} VP (blank if unknown)", None, minimum=-100, optional=True)
        opponents.append({"name": f"Opponent {i + 1}", "vp": vp})

    return {
        "schema_version": 1,
        "num_players": n,
        "turn_number": turn_number,
        "kingdom": kingdom,
        "supply": supply,
        "player": player,
        "opponents": opponents,
    }


def _opening_snapshot(p: Prompter, engine: DominionRuleEngine, n: int, kingdom: list[str], turn_number: int) -> dict:
    """Turns 1-2: you still hold the starting 7 Copper + 3 Estate, so only the Copper/Estate split is needed."""
    reg, rules = engine.registry, engine.ruleset
    p.say("  Quick start: your deck is the starting 7 Copper + 3 Estate.")
    bought_t1: list[str] = []
    if turn_number == 2:
        bought_t1 = p.ask_cards("What did you buy on turn 1? (blank = nothing)", reg, "none")
    coppers = p.ask_int("How many Coppers are in your hand? (the rest are Estates)", None, minimum=0, maximum=5)
    hand = ["Copper"] * coppers + ["Estate"] * (5 - coppers)

    owned = dict(rules.starting_deck)
    for c in bought_t1:
        owned[c] = owned.get(c, 0) + 1
    opponents = []
    for i in range(n - 1):
        vp = p.ask_int(f"Opponent {i + 1} VP (blank if unknown)", 3, minimum=-100, optional=True)
        opponents.append({"name": f"Opponent {i + 1}", "vp": vp})
    snapshot = {
        "schema_version": 1,
        "num_players": n,
        "turn_number": turn_number,
        "kingdom": kingdom,
        # Supply omitted on purpose: the parser assumes full piles and reports that assumption.
        "player": {"phase": "buy", "hand": hand, "owned_cards": owned},  # starting hands hold no Actions
        "opponents": opponents,
    }
    if turn_number == 2:
        snapshot["history"] = [{"turn": 1, "hand": [], "bought": bought_t1}]
    return snapshot


def _resolve_followups(p: Prompter, engine: DominionRuleEngine, state: GameState) -> GameState:
    reg = engine.registry
    t = state.turn
    if t.unresolved:
        p.say("Resolve these steps at the table:")
        for step in t.unresolved:
            p.say(f"  * {step}")
    if t.pending_draws:
        drawn = []
        while True:
            drawn = p.ask_cards(f"You draw {t.pending_draws} card(s). Which cards did you draw?", reg)
            if len(drawn) > t.pending_draws:
                p.say(f"  At most {t.pending_draws} card(s).")
                continue
            break
        state = engine.resolve_pending(state, drawn=drawn, clear_unresolved=False)
    if state.turn.unresolved:
        p.say(f"  Current hand: {fmt_cards(state.turn.hand)}")
        while True:
            new_hand = p.ask("Your hand now, after resolving (blank = unchanged)", "")
            try:
                hand = parse_card_list(new_hand, reg) if new_hand else None
                break
            except ValueError as e:
                p.say(f"  {e}")
        state = engine.resolve_pending(state, hand=hand)
        while True:
            try:
                supply = parse_pile_overrides(p.ask("Any Supply piles changed (e.g. 'Curse=8')", ""), reg)
                break
            except ValueError as e:
                p.say(f"  {e}")
        for k, v in supply.items():
            if k in state.supply:
                state.supply[k] = v
        if state.owned_cards is not None:
            while True:
                try:
                    deltas = parse_deltas(p.ask("Cards you gained/trashed (e.g. '+Silver, -Estate')", ""), reg)
                    break
                except ValueError as e:
                    p.say(f"  {e}")
            for c, d in deltas.items():
                state.owned_cards[c] = max(0, state.owned_cards.get(c, 0) + d)
        coins = p.ask_int("Coins ($) now", state.turn.coins)
        state.turn.coins = coins
    return state


def _next_turn_hand_default(state: GameState, rules) -> str:
    """On turn 2 the hand is whatever the starting deck didn't give you on turn 1."""
    last = state.history[-1] if state.history else None
    if not (last and last["turn"] == 1 and len(last["hand"]) == 5):
        return ""
    rest = Counter(rules.starting_deck)
    rest.subtract(last["hand"])
    if any(v < 0 for v in rest.values()):
        return ""
    return fmt_cards([c for c, k in rest.items() for _ in range(k)])


def _apply_gains(s: GameState, deltas: dict[str, int]) -> list[str]:
    """Gains come out of the Supply; losses go to the trash. Returns notes about anything odd."""
    notes = []
    for card, d in deltas.items():
        if d > 0:
            if card not in s.supply:
                notes.append(f"{card} is not in this game's Supply - ignored.")
                continue
            if s.supply[card] < d:
                notes.append(f"Only {s.supply[card]} {card} left in the Supply.")
            s.supply[card] = max(0, s.supply[card] - d)
        else:
            s.trash.extend([card] * -d)
    return notes


def _start_next_turn(p: Prompter, engine: DominionRuleEngine, state: GameState) -> GameState:
    """Carry kingdom, Supply, your deck, trash and history into your next turn."""
    # TODO: rebuild this multi-turn loop on top of inference.Session (see docs/ROADMAP.md, item 1),
    # so `interactive` and `session` share one game file and the snapshot `history` field can go.
    reg, rules = engine.registry, engine.ruleset
    s = state.copy()
    p.say("")
    p.say("Opponents' turns: what did each of them gain or trash? Blank = nothing (or you didn't see).")
    p.say("  e.g. 'Silver', 'Province, Copper', '-Estate x2' (trashed). Supply piles and VP update automatically.")
    for o in s.opponents:
        while True:
            try:
                deltas = parse_deltas(p.ask(f"{o.name}", ""), reg)
                break
            except ValueError as e:
                p.say(f"  {e}")
        for note in _apply_gains(s, deltas):
            p.say(f"  {note}")
        if o.vp is not None:
            vp_change = sum(d * (reg.get(c).vp or 0) for c, d in deltas.items())
            o.vp += vp_change
        else:
            o.vp = p.ask_int(f"{o.name} VP (blank if unknown)", None, minimum=-100, optional=True)

    if s.owned_cards is not None:
        while True:
            try:
                mine = parse_deltas(p.ask("Did YOU gain or lose cards on their turns? (e.g. '+Curse' from Witch)", ""), reg)
                break
            except ValueError as e:
                p.say(f"  {e}")
        for note in _apply_gains(s, mine):
            p.say(f"  {note}")
        for c, d in mine.items():
            s.owned_cards[c] = max(0, s.owned_cards.get(c, 0) + d)
        s.owned_cards = {c: k for c, k in s.owned_cards.items() if k > 0}

    if s.turn_number is not None:
        s.turn_number += 1
    while True:
        hand = p.ask_cards(f"Your new hand for turn {s.turn_number or '?'}", reg, _next_turn_hand_default(s, rules))
        not_owned = sorted(c for c, k in Counter(hand).items()
                           if s.owned_cards is not None and s.owned_cards.get(c, 0) < k)
        if not hand:
            p.say("  Enter the cards you drew, e.g. 'Silver, Moat, Copper x3'.")
        elif not_owned:
            p.say(f"  Your deck doesn't have that many: {', '.join(not_owned)}. Check your hand.")
        else:
            break
    has_action = any(reg.get(c).is_action for c in hand)
    s.turn = TurnState(
        phase=Phase.ACTION if has_action else Phase.BUY,  # no Actions in hand: go straight to buying
        hand=sorted(hand),
        actions=rules.start_actions,
        buys=rules.start_buys,
        coins=rules.start_coins,
    )
    s.deck_count = s.discard_count = None  # draws/shuffles are not tracked
    return s


def run_session(p: Prompter, engine: DominionRuleEngine, state: GameState, save_path: str | None = None, top: int = 5) -> int:
    policy = HeuristicPolicy(engine)
    start_hand = sorted(state.turn.hand + state.turn.in_play)
    while True:
        if state.turn.phase == Phase.TURN_OVER:
            if save_path:
                _save(p, state, save_path)  # autosave after every turn
            end = engine.game_end_status(state)
            if end.game_over:
                p.say("GAME OVER: " + " ".join(end.reasons))
                return 0
            try:
                if not p.ask_yes("Continue to your next turn?", default=True):
                    return 0
                state = _start_next_turn(p, engine, state)
            except Quit:
                return 0
            end = engine.game_end_status(state)
            if end.game_over:
                p.say("GAME OVER (on an opponent's turn): " + " ".join(end.reasons))
                break
            start_hand = sorted(state.turn.hand)
            continue

        issues = validate_state(state, engine)
        shown = [i for i in issues if i.severity != "info"]
        if shown:
            p.say(render_issues(shown))
        if has_errors(issues):
            p.say("Fix the errors above (edit the saved JSON or restart) - no recommendation for an invalid state.")
            return 1
        advice = policy.recommend(state)
        p.say("")
        p.say(render_advice(advice, state, engine, top=top))
        p.say("")
        options = advice.ranked[:top]
        p.say("Next: [Enter] apply the recommendation, a number to apply that option, "
              "'s' to save the snapshot, 'q' to quit.")
        try:
            choice = p.ask("Choice", "").lower()
        except Quit:
            break
        if choice == "s":
            _save(p, state, save_path)
            continue
        if choice == "":
            action = advice.best.action
        elif choice.isdigit() and 1 <= int(choice) <= len(options):
            action = options[int(choice) - 1].action
        else:
            p.say("  Unrecognised choice.")
            continue
        transition = engine.apply(state, action)
        for note in transition.notes:
            p.say(f"  - {note}")
        try:
            state = _resolve_followups(p, engine, transition.state)
        except Quit:
            break
        if state.turn.phase == Phase.TURN_OVER:
            state.history.append({"turn": state.turn_number, "hand": start_hand, "bought": list(state.turn.bought)})
            p.say("Turn finished. Bought: " + (fmt_cards(state.turn.bought) if state.turn.bought else "nothing") + ".")
    if save_path:
        _save(p, state, save_path)
    return 0


def _save(p: Prompter, state: GameState, path: str | None) -> None:
    target = path or p.ask("Save snapshot to", "game.json")
    Path(target).write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")
    p.say(f"  Saved to {target}")


def interactive_main(engine: DominionRuleEngine, load: str | None = None, save: str | None = None,
                     input_fn=input, print_fn=print) -> int:
    p = Prompter(input_fn, print_fn)
    try:
        if load:
            data = json.loads(Path(load).read_text(encoding="utf-8"))
        else:
            data = build_snapshot(p, engine)
        try:
            state = state_from_dict(data, engine.registry, engine.ruleset)
        except SnapshotError as e:
            p.say(str(e))
            return 1
        if save and not load:
            Path(save).write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")
            p.say(f"Snapshot saved to {save}")
        return run_session(p, engine, state, save_path=save)
    except Quit:
        p.say("Bye.")
        return 0

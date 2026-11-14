"""``cardgame dominion session <verb> FILE ...``: track one real game in a session file.

The session file is ``Session.to_dict()`` JSON (config + event log); every command
replays it, so the belief is reproducible. Mutating verbs validate the new event by
replaying it and save atomically (temp file + replace) only if it is consistent:
contradictions and illegal events exit 1 and leave the file byte-identical.
``session interactive FILE`` is a small REPL over the same verbs.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from ..inference import (
    EndTurnObservation,
    HandNotDetermined,
    ObserveHand,
    RecordDiscard,
    RecordDraw,
    RecordGain,
    RecordPlay,
    RecordPurchase,
    RecordSupply,
    RecordTrash,
    Session,
    SessionError,
    correct_event,
    infer_belief_state,
    initialize_session,
    plan_next_action,
    predict_next_hand,
    record_event,
    remove_event,
    reset_session,
    to_game_state,
)
from ..inference.belief import ENUMERATION_LIMIT, outcome_count, predictive
from ..inference.session import HandPrediction
from ..rules.zones import UNKNOWN_CARD
from ..rules import DominionRuleEngine
from .commands import EXIT_INVALID, EXIT_OK, EXIT_USAGE, _plan_config, _run, add_planner_flags
from .parsing import parse_card_list, parse_pile_overrides
from .render import RULE, fmt_cards, render_plan

Say = Callable[[str], None]

MUTATING = ("start", "hand", "draw", "play", "buy", "gain", "trash", "discard", "supply", "end-turn",
            "undo", "correct", "reset")
EVENT_VERBS = ("hand", "draw", "play", "buy", "gain", "trash", "discard", "supply", "end-turn")


class Rejected(Exception):
    """The request cannot be applied to this session (exit 1, file unchanged)."""


# ---------------------------------------------------------------- file I/O

def load_session(path: str | Path) -> Session:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No session file '{p}'. Start one with: cardgame dominion session start {p}")
    return Session.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_session(path: str | Path, session: Session) -> None:
    """Write atomically: a temp file in the same folder, then ``os.replace``."""
    p = Path(path)
    data = json.dumps(session.to_dict(), indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------- formatting

def fmt_p(v: Fraction | float) -> str:
    """``5/7 = 0.714`` for exact values, ``1`` / ``0`` when certain, ``~0.714`` for estimates."""
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator} = {float(v):.3f}"
    return f"~{v:.3f}"


def describe_event(e) -> str:
    kind = e.kind
    if kind == "observe_hand":
        return f"hand {fmt_cards(list(e.cards))}"
    if kind == "record_draw":
        return f"draw {e.count} unobserved" if e.cards is None else f"draw {fmt_cards(list(e.cards))}"
    if kind == "record_play":
        extra = f" (Throne Room target {e.throne_target})" if e.throne_target else ""
        return f"play {e.card}{extra}" + (" (trash a Copper)" if e.trash_copper else "")
    if kind == "record_purchase":
        return f"buy {e.card}"
    if kind == "record_gain":
        return f"gain {e.card} to {e.destination}"
    if kind == "record_trash":
        return f"trash {e.card} from {e.zone}"
    if kind == "record_discard":
        return f"discard {fmt_cards(list(e.cards))}"
    if kind == "record_supply":
        return f"supply {e.pile}={e.count}"
    return "end-turn" if kind == "end_turn" else kind


def deduced_hand(session: Session) -> list[str] | None:
    """The whole hand when draws are still pending but their cards are logically determined, else None.

    E.g. the start-of-turn hand after the previous Clean-up left exactly 5 known cards on the draw pile.
    Presentation helper only: the belief keeps the draws pending (inference semantics unchanged).
    """
    b = infer_belief_state(session)
    hand = b.determined_hand()
    if not b.pending_draws or hand is None or outcome_count(b.worlds, b.pending_draws) > ENUMERATION_LIMIT:
        return None
    drawn, _ = predictive(b.worlds, b.pending_draws)
    if len(drawn) != 1:
        return None
    cards = next(iter(drawn))
    return None if UNKNOWN_CARD in cards else sorted(hand + list(cards))


def current_hand(session: Session) -> dict[str, Any]:
    b = infer_belief_state(session)
    deduced = deduced_hand(session)
    if deduced is not None:
        return {"status": "deduced", "cards": deduced, "pending_draws": b.pending_draws,
                "source": f"the {b.pending_draws} pending draw(s) are fully determined by the tracked draw pile"}
    hand = b.determined_hand()
    return {"status": "pending" if b.pending_draws else b.zone_status("hand"),
            "cards": None if hand is None else sorted(hand), "pending_draws": b.pending_draws}


def belief_summary(session: Session, top: int = 5) -> dict[str, Any]:
    """JSON summary returned by ``show --json`` and every mutating verb with ``--json``."""
    pred = predict_next_hand(session, top=top)
    return {"events": len(session.events), "current_hand": current_hand(session),
            "belief": infer_belief_state(session).to_dict(), "next_hand": pred.to_dict()}


def _zone_line(label: str, info: dict[str, Any], suffix: str = "") -> str:
    status, size = info["status"], info["size"]
    if size is None:
        return f"{label} ({status}): size unknown"
    certain = [c for c, n in info["certain"].items() for _ in range(n)]
    text = fmt_cards(certain) if certain or not info["uncertain_count"] else ""
    if info["uncertain_count"]:
        what = "unknown ('?')" if status == "unknown" else "uncertain"
        text = (text + " + " if text else "") + f"{info['uncertain_count']} {what}"
    if info.get("expected"):
        text += "; expected " + ", ".join(f"{c} {v}" for c, v in info["expected"].items())
    return f"{label} ({status}, {size} card{'s' if size != 1 else ''}{suffix}): {text or '(empty)'}"


def _hand_lines(session: Session, pred: HandPrediction) -> list[str]:
    b = infer_belief_state(session)
    cur = current_hand(session)
    if cur["status"] == "deduced" and b.pending_draws:
        return [f"Hand (deduced): {fmt_cards(cur['cards'])}",
                f"  (the {b.pending_draws} pending draw(s) are determined by the draw pile - no need to enter them)"]
    if b.pending_draws and pred.which == "start_of_turn":
        reshuffle = ("a reshuffle is certain" if pred.p_reshuffle == 1
                     else f"P(reshuffle) {fmt_p(pred.p_reshuffle)}")
        return [f"Hand: {b.pending_draws} card(s) to draw - not determined: {len(pred.distribution)} possible hands, "
                f"{reshuffle} (see 'predict'; enter it with: hand \"...\")"]
    line = _zone_line("Hand", b.zone_summary("hand"))
    if b.pending_draws:
        line += f"  + {b.pending_draws} pending draw(s)"
    return [line]


def _next_hand_line(pred: HandPrediction) -> list[str]:
    if pred.which == "start_of_turn":
        return []
    head = f"Next hand (turn {pred.turn if pred.turn is not None else '?'}, if nothing else happens this turn): "
    if pred.deterministic:
        return [head + f"known exactly - {fmt_cards(list(pred.hand or ()))}"]
    return [head + f"{len(pred.distribution)} possible hands ({pred.method}), P(reshuffle) {fmt_p(pred.p_reshuffle)}, "
                   f"E[$] {fmt_p(pred.expected_coins)} (see 'predict')"]


def render_belief(session: Session, full: bool = False) -> list[str]:
    b = infer_belief_state(session)
    pred = predict_next_hand(session, top=5)
    t = b.game.turn
    lines = []
    if full:
        lines += [f"Dominion session - {len(session.events)} event(s) recorded", RULE]
    lines.append(f"Turn {b.turn if b.turn is not None else '?'} | Phase {b.phase.upper()} | Actions {t.actions} | "
                 f"Buys {t.buys} | ${t.coins}" + (f" | pending draws {b.pending_draws}" if b.pending_draws else ""))
    lines += _hand_lines(session, pred)
    if full or t.in_play:
        lines.append(f"In play: {fmt_cards(t.in_play)}")
    if t.bought:
        lines.append(f"Bought this turn: {fmt_cards(t.bought)}")
    draw = b.zone_summary("draw_pile")
    order = ""
    if draw.get("size"):
        top = draw.get("top") or []
        order = f", top first: {', '.join(top)}" if top else ""
        order += (", rest in unknown order" if top else ", order unknown") if draw["size"] > len(top) else ""
    lines.append(_zone_line("Draw pile", draw, order))
    lines.append(_zone_line("Discard", b.zone_summary("discard_pile")))
    lines += _next_hand_line(pred)
    if full:
        g = b.game
        if g.owned_cards is not None:
            lines.append(f"Owned ({sum(g.owned_cards.values())} cards): "
                         + fmt_cards([c for c, n in g.owned_cards.items() for _ in range(n)]))
        if g.trash:
            lines.append(f"Trash: {fmt_cards(g.trash)}")
        lines.append(f"Hidden states: {len(b.worlds)}    Information level: {b.information_level}")
        lines.append("Observed: hand (when entered), plays, buys, gains, trashes, Supply counts you record. "
                     "Deduced: certain consequences. Probabilistic: exact mixture over hidden states.")
        if b.flags:
            lines += ["Incomplete history (not contradictions):"] + [f"  - {f}" for f in b.flags]
        if b.notes:
            lines += ["Recent notes:"] + [f"  - {n}" for n in b.notes[-5:]]
    return lines


def render_prediction(pred: HandPrediction) -> list[str]:
    which = ("Start-of-turn hand" if pred.which == "start_of_turn"
             else "Next turn's hand (assuming no further plays, draws or gains this turn)")
    lines = [f"{which} - turn {pred.turn if pred.turn is not None else '?'}, {pred.size} cards, "
             f"method {pred.method}, hidden states {pred.worlds}"]
    if pred.deterministic:
        lines.append(f"Known exactly: {fmt_cards(list(pred.hand or ()))}")
    lines.append(f"P(reshuffle while drawing): {fmt_p(pred.p_reshuffle)}")
    if not pred.deterministic:
        lines.append(f"{len(pred.distribution)} possible hands (most likely first):")
        for cards, p_exact, p in pred.distribution[:pred.top]:
            lines.append(f"  {fmt_p(p_exact if p_exact is not None else p):<20} {fmt_cards(list(cards))}")
        rest = pred.distribution[pred.top:]
        if rest:
            other = sum((pe for _, pe, _ in rest), Fraction(0)) if pred.method == "exact" else sum(p for *_, p in rest)
            lines.append(f"  {fmt_p(other):<20} ({len(rest)} other hands; use --top)")
        total = (sum((pe for _, pe, _ in pred.distribution), Fraction(0)) if pred.method == "exact"
                 else sum(p for *_, p in pred.distribution))
        lines.append(f"  Sum over all {len(pred.distribution)} hands: {fmt_p(total)}")
    lines.append(f"E[$ from Treasures]: {fmt_p(pred.expected_coins)}")
    lines.append("P($ >= k): " + "   ".join(f"{k}: {fmt_p(v)}" for k, v in pred.p_coins_at_least.items()))
    if pred.p_at_least_one:
        lines.append("P(at least one):")
        lines += [f"  {name:<12} {fmt_p(v)}" for name, v in pred.p_at_least_one.items()]
    lines += [f"Note: {n}" for n in pred.notes]
    return lines


# ---------------------------------------------------------------- events from arguments

def _cards(words: list[str], registry) -> list[str]:
    return parse_card_list(" ".join(words), registry)


def _one_card(words: list[str], registry) -> str:
    cards = _cards(words, registry)
    if len(cards) != 1:
        raise ValueError(f"Expected exactly one card, got {' '.join(words)!r}.")
    return cards[0]


def build_events(args, registry) -> list:
    """Event(s) for an event verb; card names are resolved here (typos get suggestions)."""
    v = args.verb
    if v == "hand":
        return [ObserveHand(_cards(args.cards, registry))]
    if v == "draw":
        if bool(args.cards) == (args.count is not None):
            raise ValueError("draw: give the drawn cards, or --count N for cards you did not see (not both).")
        return [RecordDraw(count=args.count) if args.cards == [] else RecordDraw(cards=_cards(args.cards, registry))]
    if v == "play":
        target = None if args.throne_target is None else _one_card([args.throne_target], registry)
        return [RecordPlay(_one_card(args.card, registry), target, True if args.trash_copper else None)]
    if v == "buy":
        return [RecordPurchase(_one_card(args.card, registry))]
    if v == "gain":
        return [RecordGain(_one_card(args.card, registry), args.to)]
    if v == "trash":
        return [RecordTrash(_one_card(args.card, registry), args.zone)]
    if v == "discard":
        cards = _cards(args.cards, registry)
        if not cards:
            raise ValueError("discard: name the discarded cards.")
        return [RecordDiscard(cards)]
    if v == "supply":
        piles = parse_pile_overrides(" ".join(args.piles), registry)
        if not piles:
            raise ValueError("supply: expected PILE=N, e.g. supply Province=6")
        return [RecordSupply(pile, n) for pile, n in piles.items()]
    if v == "end-turn":
        return [EndTurnObservation()]
    raise ValueError(f"not an event verb: {v}")


# ---------------------------------------------------------------- verbs

class _Ctx:
    def __init__(self, args, engine: DominionRuleEngine, say: Say, err: Say, verb_parser: Callable[[], Any]):
        self.args, self.engine, self.say, self.err, self.verb_parser = args, engine, say, err, verb_parser
        self.json = getattr(args, "json", False)

    def emit(self, obj: Any, lines: list[str]) -> None:
        self.say(json.dumps(obj, indent=2) if self.json else "\n".join(lines))


def _start(ctx: _Ctx) -> int:
    a = ctx.args
    path = Path(a.file)
    if path.exists() and not a.force:
        raise FileExistsError(f"'{path}' already exists; use --force to overwrite it.")
    config: dict[str, Any] = {}
    if a.from_snapshot:
        if a.players is not None or a.kingdom is not None:
            raise ValueError("--from-snapshot already states the players and Kingdom; drop --players/--kingdom.")
        config["snapshot"] = json.loads(Path(a.from_snapshot).read_text(encoding="utf-8"))
    if a.players is not None:
        config["num_players"] = a.players
    if a.kingdom is not None and a.kingdom.strip().lower() not in ("first", "first game"):
        config["kingdom"] = parse_card_list(a.kingdom, ctx.engine.registry)
    session = initialize_session(config)
    save_session(path, session)
    b = infer_belief_state(session)
    head = [f"Started Dominion session in {path} ({b.game.num_players} players; Kingdom: "
            f"{', '.join(b.game.kingdom)})."]
    if a.from_snapshot:
        head.append(f"Started from snapshot {a.from_snapshot}: history before it is unknown (not invented).")
    else:
        deck = [c for c, n in ctx.engine.ruleset.starting_deck.items() for _ in range(n)]
        head.append(f"Initial deck loaded: {fmt_cards(deck)} (shuffled: composition known, order unknown)")
    ctx.emit({"ok": True, "file": str(path), **belief_summary(session)}, head + render_belief(session))
    return EXIT_OK


def _save_and_report(ctx: _Ctx, session: Session, headline: list[str], extra: dict[str, Any]) -> int:
    save_session(ctx.args.file, session)
    ctx.emit({"ok": True, **extra, **belief_summary(session)}, headline + render_belief(session))
    return EXIT_OK


def _record(ctx: _Ctx, session: Session) -> int:
    events = build_events(ctx.args, ctx.engine.registry)
    start = len(session.events)
    for e in events:
        session = record_event(session, e)
    recorded = [f"Recorded #{start + i}: {describe_event(e)}" for i, e in enumerate(events)]
    return _save_and_report(ctx, session, recorded, {"recorded": [e.to_dict() for e in events]})


def _undo(ctx: _Ctx, session: Session) -> int:
    if not session.events:
        raise Rejected("Nothing to undo: the session has no events.")
    last = len(session.events) - 1
    gone = session.events[last]
    return _save_and_report(ctx, remove_event(session, last), [f"Undid #{last}: {describe_event(gone)}"],
                            {"removed": {"index": last, "event": gone.to_dict()}})


def _correct(ctx: _Ctx, session: Session) -> int:
    a = ctx.args
    rest = list(a.event)
    if "--json" in rest:
        rest.remove("--json")
        ctx.json = True
    if len(rest) == 1 and " " in rest[0].strip():
        rest = shlex.split(rest[0])
    if not rest or rest[0] not in EVENT_VERBS:
        raise ValueError(f"correct INDEX <verb> [args]: verb must be one of {', '.join(EVENT_VERBS)}")
    sub = ctx.verb_parser().parse_args(rest)
    events = build_events(sub, ctx.engine.registry)
    if len(events) != 1:
        raise ValueError("correct replaces one event: give a single PILE=N.")
    if not 0 <= a.index < len(session.events):
        raise Rejected(f"No event #{a.index} (the session has {len(session.events)} event(s); see 'history').")
    old = session.events[a.index]
    new = correct_event(session, a.index, events[0])
    return _save_and_report(ctx, new, [f"Corrected #{a.index}: {describe_event(old)} -> {describe_event(events[0])}"
                                       " (history replayed)"],
                            {"corrected": {"index": a.index, "old": old.to_dict(), "new": events[0].to_dict()}})


def _reset(ctx: _Ctx, session: Session) -> int:
    n = len(session.events)
    return _save_and_report(ctx, reset_session(session), [f"Reset: removed {n} event(s); back to the initial state."],
                            {"removed_events": n})


def _show(ctx: _Ctx, session: Session) -> int:
    ctx.emit(belief_summary(session), render_belief(session, full=True))
    return EXIT_OK


def _predict(ctx: _Ctx, session: Session) -> int:
    if ctx.args.top < 1:
        raise ValueError("--top must be >= 1")
    pred = predict_next_hand(session, top=ctx.args.top)
    ctx.emit(pred.to_dict(), render_prediction(pred))
    return EXIT_OK


def _plan(ctx: _Ctx, session: Session) -> int:
    cfg = _plan_config(ctx.args)
    notes = ["Zone knowledge comes from this session's tracked belief (the planner's notes call it 'given in the "
             "snapshot')."]
    hand = deduced_hand(session)
    if hand is not None:
        # Same belief, with the deduced start-of-turn hand recorded (not saved): the planner needs a known hand.
        session = record_event(session, ObserveHand(hand))
        notes.append(f"Planning on the deduced hand: {fmt_cards(hand)}.")
    result = plan_next_action(session, cfg)
    state = to_game_state(session)
    out = result.to_dict()
    out["session_notes"] = notes
    ctx.emit(out, notes + [render_plan(result, state, ctx.engine, cfg)])
    return EXIT_OK


def _history(ctx: _Ctx, session: Session) -> int:
    turn = session.config["snapshot"].get("turn_number") if session.config.get("snapshot") else 1
    rows, lines = [], []
    for i, e in enumerate(session.events):
        rows.append({"index": i, "turn": turn, "event": e.to_dict(), "text": describe_event(e)})
        lines.append(f"  #{i:<3} T{turn if turn is not None else '?':<3} {describe_event(e)}")
        if e.kind == "end_turn" and turn is not None:
            turn += 1
    ctx.emit(rows, lines or ["  (no events yet)"])
    return EXIT_OK


HANDLERS = {"undo": _undo, "correct": _correct, "reset": _reset, "show": _show, "predict": _predict,
            "plan": _plan, "history": _history}


def run_verb(args, engine: DominionRuleEngine, say: Say = print, err: Say | None = None,
             verb_parser: Callable[[], Any] | None = None) -> int:
    """Run one verb on ``args.file``. Exit 1 (file untouched) if the session rejects the request."""
    err = err or (lambda m: print(m, file=sys.stderr))
    ctx = _Ctx(args, engine, say, err, verb_parser or (lambda: build_verb_parser(_QuietParser)))
    try:
        if args.verb == "start":
            return _start(ctx)
        session = load_session(args.file)
        if args.verb in EVENT_VERBS:
            return _record(ctx, session)
        return HANDLERS[args.verb](ctx, session)
    except (SessionError, HandNotDetermined, Rejected) as e:
        kind = ("inconsistent" if type(e).__name__ == "InconsistentObservation" else
                "illegal" if isinstance(e, SessionError) else "not_determined" if isinstance(e, HandNotDetermined)
                else "rejected")
        if ctx.json:
            say(json.dumps({"ok": False, "error": kind, "message": str(e)}, indent=2))
        else:
            err({"rejected": "Rejected", "not_determined": "Not possible"}.get(kind, f"Rejected ({kind})") + f": {e}")
            if args.verb in MUTATING:
                err("The session file was not changed.")
        return EXIT_INVALID
    except ValueError as e:
        if args.verb in EVENT_VERBS + ("correct",):  # bad card name / arguments for an event: reject it
            err(f"Rejected (illegal): {e}\nThe session file was not changed.")
            return EXIT_INVALID
        raise
    except FileExistsError as e:
        err(f"Error: {e}")
        return EXIT_USAGE


# ---------------------------------------------------------------- parsers

def add_verbs(sub, with_file: bool) -> None:
    """Add every session verb to ``sub`` (``with_file``: the CLI form, ``verb FILE ...``)."""

    def verb(name: str, help_: str, json_flag: bool = True):
        sp = sub.add_parser(name, help=help_, description=help_)
        if with_file:
            sp.add_argument("file", metavar="FILE", help="Session JSON file")
        if json_flag:
            sp.add_argument("--json", action="store_true", help="Machine-readable JSON output")
        sp.set_defaults(verb=name)
        return sp

    if with_file:
        s = verb("start", "Start a new session (standard starting deck, opening hand pending)")
        s.add_argument("--players", type=int, help="Number of players (default 2)")
        s.add_argument("--kingdom", help='"first" (default: the rulebook First Game) or "A, B, ..." (10 cards)')
        s.add_argument("--from-snapshot", metavar="S.json", help="Start mid-game from a partial snapshot")
        s.add_argument("--force", action="store_true", help="Overwrite an existing session file")
    verb("hand", 'Your hand, e.g. hand "4 Copper, Estate" (resolves the start-of-turn draw)').add_argument(
        "cards", nargs="+", help="Card list")
    d = verb("draw", 'Cards drawn mid-turn: draw "Silver, Copper", or --count N if you did not see them')
    d.add_argument("cards", nargs="*", help="Card list")
    d.add_argument("--count", type=int, help="Number of unobserved cards drawn")
    p = verb("play", "Play a card from hand (Action or Treasure)")
    p.add_argument("card", nargs="+")
    p.add_argument("--throne-target", metavar="X", help="Throne Room: the Action played twice")
    p.add_argument("--trash-copper", action="store_true", help="Moneylender: trash a Copper")
    verb("buy", "Record a purchase (Treasures in hand are played first if needed)").add_argument("card", nargs="+")
    g = verb("gain", "Record a card gained without buying it")
    g.add_argument("card", nargs="+")
    g.add_argument("--to", choices=("discard", "hand", "deck"), default="discard")
    t = verb("trash", "Record one of your cards being trashed")
    t.add_argument("card", nargs="+")
    t.add_argument("--from", dest="zone", choices=("hand", "in_play", "discard"), default="hand")
    verb("discard", 'Cards discarded from hand (Cellar, an opponent\'s Militia): discard "2 Estate"').add_argument(
        "cards", nargs="+")
    verb("supply", "An observed Supply count, e.g. supply Province=6 (never used to infer your gains)").add_argument(
        "piles", nargs="+", metavar="PILE=N")
    verb("end-turn", "Your turn ended (Clean-up; the next hand of 5 is pending)")
    verb("show", "Belief: observed vs deduced vs probabilistic, per zone")
    verb("predict", "Distribution of the next hand (exact where feasible)").add_argument(
        "--top", type=int, default=10, help="How many hands to list (default 10)")
    add_planner_flags(verb("plan", "Run the lookahead planner on the tracked state"))
    verb("history", "List recorded events")
    verb("undo", "Remove the last event")
    c = verb("correct", 'Replace event INDEX and replay, e.g. correct 3 buy Silver  (or "buy Silver")')
    c.add_argument("index", type=int, metavar="INDEX")
    c.add_argument("event", nargs=argparse.REMAINDER, metavar="VERB ARGS")
    verb("reset", "Clear all events (same start)")
    if with_file:
        i = verb("interactive", "Interactive loop over the same verbs (without FILE)", json_flag=False)
        i.set_defaults(verb="interactive")


class _QuietParser(argparse.ArgumentParser):
    """ArgumentParser that raises instead of exiting (for the REPL and ``correct``)."""

    say: Say = print

    def error(self, message: str):
        raise ValueError(f"{self.prog}: {message}")

    def exit(self, status: int = 0, message: str | None = None):
        if message:
            self.say(message.rstrip())
        raise _HelpShown()

    def _print_message(self, message: str, file=None) -> None:
        if message:
            self.say(message.rstrip("\n"))


class _HelpShown(Exception):
    pass


def build_verb_parser(parser_class=_QuietParser) -> argparse.ArgumentParser:
    root = parser_class(prog="session", add_help=False)
    add_verbs(root.add_subparsers(dest="verb", required=True, metavar="VERB", parser_class=parser_class), False)
    return root


REPL_HELP = """Verbs (same as `cardgame dominion session <verb> FILE`, without FILE):
  hand 4 Copper, Estate      draw Silver | draw --count 2     play Smithy [--throne-target X]
  buy Silver                 gain Silver [--to deck]          trash Copper [--from hand]
  discard 2 Estate           supply Province=6                end-turn
  show   predict [--top K]   plan [--horizon hands:2 --objective future_money ...]
  history   undo   correct INDEX <verb> [args]   reset   help [VERB]   quit
Add --json for machine-readable output. The file is saved after each successful change."""


def interactive_session(path: str | Path, engine: DominionRuleEngine | None = None,
                        input_fn: Callable[[str], str] = input, print_fn: Say = print) -> int:
    """REPL over the session verbs; saves after each successful mutation. EOF or 'quit' ends it."""
    engine = engine or DominionRuleEngine()
    parser_class = type("_ReplParser", (_QuietParser,), {"say": staticmethod(print_fn)})
    parser = build_verb_parser(parser_class)
    path = Path(path)
    if not path.exists():
        save_session(path, initialize_session())
        print_fn(f"New session in {path} (2 players, First Game Kingdom, standard starting deck).")
    load_session(path)  # fail early on a bad file
    print_fn(f"Dominion session {path}. Type 'help' for verbs, 'quit' to leave.")
    print_fn("\n".join(render_belief(load_session(path))))
    while True:
        try:
            line = input_fn("session> ")
        except (EOFError, KeyboardInterrupt):
            print_fn("")
            return EXIT_OK
        line = (line or "").strip()
        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            return EXIT_OK
        try:
            words = shlex.split(line)
        except ValueError as e:
            print_fn(f"  {e}")
            continue
        if words[0] in ("help", "?"):
            if len(words) > 1:
                try:
                    parser.parse_args([words[1], "--help"])
                except (_HelpShown, ValueError):
                    pass
            else:
                print_fn(REPL_HELP)
            continue
        if words[0] in ("start", "interactive"):
            print_fn(f"  '{words[0]}' is not available here; use reset to start over.")
            continue
        try:
            args = parser.parse_args(words)
        except _HelpShown:
            continue
        except ValueError as e:
            print_fn(f"  {e}  (type 'help')")
            continue
        args.file = str(path)
        try:
            run_verb(args, engine, say=print_fn, err=print_fn, verb_parser=lambda: build_verb_parser(parser_class))
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print_fn(f"Error: {e}")


def cmd_session(args, engine: DominionRuleEngine) -> int:
    if args.verb == "interactive":
        return interactive_session(args.file, engine)
    return run_verb(args, engine)


def register_session_cli(sub) -> None:
    """Add ``session`` to the ``cardgame dominion`` sub-commands."""
    sp = sub.add_parser("session", help="Track a real game across turns: record events, infer hidden cards, predict",
                        description="Track one real game in a session file (event log; replayed on every command).")
    add_verbs(sp.add_subparsers(dest="verb", required=True, metavar="VERB"), True)
    sp.set_defaults(func=_run(cmd_session))

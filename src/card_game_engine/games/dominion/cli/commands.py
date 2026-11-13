"""Dominion sub-commands for the ``cardgame`` CLI: ``cardgame dominion <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

from ..policy import HeuristicPolicy
from ..rules import DominionRuleEngine, SnapshotError, action_to_dict, has_errors, state_from_dict, validate_state
from .interactive import interactive_main
from .render import render_advice, render_issues, render_state

EXIT_OK, EXIT_INVALID, EXIT_USAGE = 0, 1, 2


def example_names() -> list[str]:
    folder = resources.files("card_game_engine.games.dominion.examples")
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".json"))


def load_example(name: str) -> dict:
    path = resources.files("card_game_engine.games.dominion.examples").joinpath(f"{name}.json")
    if not path.is_file():
        raise FileNotFoundError(f"No example named '{name}'. Available: {', '.join(example_names())}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_snapshot(args) -> dict:
    if getattr(args, "example", None):
        return load_example(args.example)
    if not getattr(args, "state", None):
        raise FileNotFoundError("Provide a snapshot file (STATE) or --example NAME. See: cardgame dominion examples")
    if args.state == "-":
        return json.load(sys.stdin)
    return json.loads(Path(args.state).read_text(encoding="utf-8"))


def _emit(obj, as_json: bool, text: str) -> None:
    if as_json:
        print(json.dumps(obj, indent=2))
    else:
        print(text)


def _parse_state(args, engine):
    data = _load_snapshot(args)
    state = state_from_dict(data, engine.registry, engine.ruleset)
    issues = validate_state(state, engine)
    return state, issues


# ---------------------------------------------------------------- commands

def cmd_recommend(args, engine) -> int:
    state, issues = _parse_state(args, engine)
    if has_errors(issues):
        _emit({"valid": False, "issues": [i.to_dict() for i in issues]}, args.json,
              "Snapshot is not consistent with the rules:\n" + render_issues(issues))
        return EXIT_INVALID
    advice = HeuristicPolicy(engine).recommend(state)
    if args.json:
        out = advice.to_dict()
        out["issues"] = [i.to_dict() for i in issues]
        print(json.dumps(out, indent=2))
    else:
        warn = [i for i in issues if i.severity == "warning"]
        if warn:
            print(render_issues(warn))
        print(render_advice(advice, state, engine, top=args.top, verbose=not args.brief))
    return EXIT_OK


def cmd_legal(args, engine) -> int:
    state, issues = _parse_state(args, engine)
    legal = engine.legal_actions(state)
    blocking = engine.blocking_issues(state)
    illegal = engine.illegal_explanations(state) if args.explain else []
    obj = {
        "legal_actions": [action_to_dict(a) for a in legal],
        "blocking": blocking,
        "illegal_buys": [{"action": action_to_dict(a), "reasons": r} for a, r in illegal],
        "issues": [i.to_dict() for i in issues],
    }
    lines = render_state(state, engine) + ["", "Legal actions:"]
    lines += [f"  - {a.describe()}" for a in legal] or ["  (none)"]
    if blocking:
        lines += ["Blocked:"] + [f"  - {b}" for b in blocking]
    if illegal:
        lines += ["", "Not allowed:"] + [f"  - {a.describe()}: {'; '.join(r)}" for a, r in illegal]
    _emit(obj, args.json, "\n".join(lines))
    return EXIT_INVALID if has_errors(issues) else EXIT_OK


def cmd_validate(args, engine) -> int:
    state, issues = _parse_state(args, engine)
    valid = not has_errors(issues)
    _emit({"valid": valid, "issues": [i.to_dict() for i in issues]}, args.json,
          ("VALID" if valid else "INVALID") + "\n" + render_issues(issues))
    return EXIT_OK if valid else EXIT_INVALID


def cmd_examples(args, engine) -> int:
    if args.show:
        print(json.dumps(load_example(args.show), indent=2))
        return EXIT_OK
    rows = []
    for name in example_names():
        rows.append({"name": name, "description": load_example(name).get("description", "")})
    _emit(rows, args.json, "\n".join(f"  {r['name']:<30} {r['description']}" for r in rows)
          + "\n\nRun one with: cardgame dominion recommend --example NAME")
    return EXIT_OK


def cmd_cards(args, engine) -> int:
    reg = engine.registry
    if args.name:
        c = reg.get(args.name)
        d = c.to_dict()
        d["source_urls"] = [reg.sources[s]["url"] for s in c.sources if s in reg.sources]
        lines = [f"{c.name}  (${c.cost}, {' - '.join(c.types)}, V1 support: {c.automation})", f"  {c.text}"]
        lines += [f"  Ruling: {r}" for r in c.rulings]
        lines += [f"  Manual step: {m}" for m in c.manual_steps]
        lines += [f"  Source: {u}" for u in d["source_urls"]]
        _emit(d, args.json, "\n".join(lines))
        return EXIT_OK
    cards = reg.kingdom_cards() if args.kingdom else reg.all()
    _emit([c.to_dict() for c in cards], args.json,
          "\n".join(f"  ${c.cost:<2} {c.name:<13} {'-'.join(c.types):<16} [{c.automation:<7}] {c.text}" for c in cards))
    return EXIT_OK


def cmd_sources(args, engine) -> int:
    reg = engine.registry
    obj = {"primary_rules_source": engine.ruleset.primary_source, "card_sources": reg.sources,
           "research_date": reg.meta.get("research_date")}
    lines = [f"Rules researched {obj['research_date']}. Primary source: {obj['primary_rules_source']}", ""]
    lines += [f"  [{k}] {v['title']}\n      {v['url']} (accessed {v['accessed']})" for k, v in reg.sources.items()]
    lines.append("\nFull notes: docs/RULES_RESEARCH.md")
    _emit(obj, args.json, "\n".join(lines))
    return EXIT_OK


def cmd_interactive(args, engine) -> int:
    return interactive_main(engine, load=args.load, save=args.save)


# ---------------------------------------------------------------- parser

def _run(func):
    """Wrap a command: build the engine and turn expected errors into exit codes."""

    def runner(args) -> int:
        engine = DominionRuleEngine()
        try:
            return func(args, engine)
        except SnapshotError as e:
            if getattr(args, "json", False):
                print(json.dumps({"valid": False, "errors": e.errors}, indent=2))
            else:
                print(str(e), file=sys.stderr)
            return EXIT_INVALID
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return EXIT_USAGE
        except KeyboardInterrupt:
            return EXIT_OK

    return runner


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Attach Dominion's commands to ``parser`` (the ``cardgame dominion`` sub-parser)."""
    sub = parser.add_subparsers(dest="command", required=True)

    def add_state_args(sp):
        sp.add_argument("state", nargs="?", help="Snapshot JSON file ('-' for stdin)")
        sp.add_argument("--example", "-e", help="Use a bundled example snapshot instead of a file")
        sp.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    r = sub.add_parser("recommend", help="Rank legal actions and explain the best one")
    add_state_args(r)
    r.add_argument("--top", type=int, default=5, help="How many options to show (default 5)")
    r.add_argument("--brief", action="store_true", help="Hide per-option score breakdowns")
    r.set_defaults(func=_run(cmd_recommend))

    l = sub.add_parser("legal", help="List legal actions for a snapshot")
    add_state_args(l)
    l.add_argument("--explain", action="store_true", help="Also explain why other buys are not allowed")
    l.set_defaults(func=_run(cmd_legal))

    v = sub.add_parser("validate", help="Check a snapshot against the rules")
    add_state_args(v)
    v.set_defaults(func=_run(cmd_validate))

    e = sub.add_parser("examples", help="List bundled example snapshots")
    e.add_argument("--show", metavar="NAME", help="Print an example's JSON (use as a template)")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=_run(cmd_examples))

    c = sub.add_parser("cards", help="Show the verified card registry")
    c.add_argument("name", nargs="?", help="Show one card in detail")
    c.add_argument("--kingdom", action="store_true", help="Only Kingdom cards")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=_run(cmd_cards))

    s = sub.add_parser("sources", help="Show the rule research sources")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_run(cmd_sources))

    i = sub.add_parser("interactive", help="Build a snapshot step by step and get live recommendations")
    i.add_argument("--load", help="Start from an existing snapshot JSON instead of answering questions")
    i.add_argument("--save", help="Save the snapshot to this path")
    i.set_defaults(func=_run(cmd_interactive))

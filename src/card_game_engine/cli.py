"""Top-level command line: ``cardgame <game> <command> ...``.

Each game plugin registers its own sub-commands via ``GameDefinition.register_cli``.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .core import all_games


def _cmd_games(args) -> int:
    games = all_games()
    if args.json:
        print(json.dumps([{"id": g.id, "name": g.name, "ruleset": g.ruleset, "description": g.description}
                          for g in games], indent=2))
    else:
        for g in games:
            print(f"  {g.id:<12} {g.name} - {g.ruleset}")
            if g.description:
                print(f"  {'':<12} {g.description}")
        print("\nUsage: cardgame <game> --help")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cardgame",
        description="Card Game Rule Engine: offline rule engines and decision policies for card games.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="game", required=True, metavar="{games," + ",".join(g.id for g in all_games()) + "}")

    g = sub.add_parser("games", help="List supported games")
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=_cmd_games)

    for game in all_games():
        gp = sub.add_parser(game.id, help=f"{game.name} ({game.ruleset})", description=game.description)
        game.register_cli(gp)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

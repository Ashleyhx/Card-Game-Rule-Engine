# Development

*Back to the [README](../README.md).*

## Project layout

```
pyproject.toml
src/card_game_engine/
  core/          game-agnostic contracts (RuleEngine, Policy, GameAction, Transition, GameDefinition) + registry;
                 search.py: generic expectimax + Monte Carlo with budgets;
                 session.py: initialize_session(domain, config) dispatch
  cli.py         `cardgame <game> <command>` entry point
  games/
    dominion/    first game plugin
      data/      cards_base_2e.json, rules_base_2e.json  (verified research output)
      rules/     DominionRuleEngine: cards, ruleset, state, actions, engine, validation, zones (pile knowledge)
      policy/    HeuristicPolicy: context, weights, scoring
      planning/  lookahead planner: draws, transitions, evaluation (objectives), analysis (draw odds),
                 planner (public API)
      inference/ game tracking: events, belief (weighted worlds), replay (applies one event),
                 analysis (deck analysis / draw odds), session (public API)
      cli/       commands, interactive assistant, session_cli (`session` verbs), parsing, rendering
      examples/  bundled example snapshots
benchmarks/      planner_vs_heuristic.py
tests/
  test_core.py   registry, interface conformance, a game-agnostic agent-style loop
  test_search.py, test_benchmark.py   generic search, benchmark smoke test
  dominion/      cards/setup, engine, state, zones, draws, transitions, planner, inference, analysis,
                 policy, CLI, session CLI, offline check
docs/            user guides and design notes (see the README)
```

Rule engines never import policies, planning never imports inference, and `core` imports no game.
To add a new card game, follow the steps in [ARCHITECTURE.md](ARCHITECTURE.md#adding-another-card-game):
make a `games/<id>/` package, implement the core contracts, expose `GAME`, and register it.

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

The suite covers:

- **Core:** the registry and interfaces, including a game-agnostic agent-style loop, and the generic
  search.
- **Dominion rules:** every card's cost/effect against the research, setup sizes for 2/3/4 players,
  legal-action enumeration and constraint messages, transitions (Merchant, Throne Room, Moneylender,
  Poacher, Library, Bandit...), end-of-game detection and scoring, snapshot validation.
- **Policy and CLI:** policy behaviour on each example, every command, interactive mode with scripted
  input, and a check that the runtime makes **no network calls**.
- **Planner:** zone derivation, exact draw enumeration (probabilities sum to 1), card conservation in
  transitions, hand-verifiable values (e.g. Village→Smithy E[$]=11/3), lower-bound / partial
  labelling, and a benchmark smoke test.
- **Game tracking:** belief updates and contradiction detection, next-hand predictions, and draw odds
  checked against brute-force enumeration, including ranges that must contain the true answer when
  some cards are unknown.

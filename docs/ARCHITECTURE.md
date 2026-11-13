# Architecture

![Flowchart](flowchart.svg)

## Layers

```
card_game_engine/
├── core/                    GAME-AGNOSTIC: contracts every game implements + registry
│   ├── interfaces.py          RuleEngine, Policy, GameAction protocols; Transition; IllegalActionError;
│   │                          GameDefinition (how a game plugs in)
│   └── registry.py            available_games(), get_game("dominion")
├── cli.py                   `cardgame <game> <command>` - dispatches to each game's commands
└── games/
    └── dominion/            GAME PLUGIN: Dominion 2E Base Set  (exposes GAME = GameDefinition)
        ├── data/              Verified card + rule data (JSON research output)
        ├── rules/             RULE ENGINE - legality only, no strategy, no I/O
        │   ├── cards.py         CardDef / CardRegistry
        │   ├── ruleset.py       setup sizes, end conditions
        │   ├── state.py         GameState / TurnState + snapshot (de)serialisation
        │   ├── actions.py       PlayAction, EndActionPhase, PlayTreasure, PlayAllTreasures, Buy, EndTurn
        │   ├── engine.py        DominionRuleEngine: legal_actions, check, apply, resolve_pending, game_end_status
        │   └── validation.py    rule-level snapshot consistency checks
        ├── policy/            DECISION POLICY - ranks what the engine says is legal
        │   ├── context.py       derived facts: stage, VP, deck stats, unknowns
        │   ├── weights.py       every strategy number in one place (heuristics, not rules)
        │   ├── scoring.py       score = sum of named, explained components
        │   └── heuristic.py     HeuristicPolicy: rank(), plan(), recommend() -> Recommendation
        ├── cli/               `cardgame dominion ...` commands, interactive assistant, rendering
        └── examples/          bundled snapshots
```

Dependencies only point one way: `cli → policy → rules → data`, and every game depends on
`core`. `core` imports no game (the registry loads games lazily by module path). A rule
engine never imports a policy.

## Core contracts (`card_game_engine.core`)

| Contract | Purpose |
|----------|---------|
| `RuleEngine.legal_actions(state) -> list[Action]` | The only source of legality. Policies/agents choose from this list. |
| `RuleEngine.check(state, action) -> list[str]` | Human-readable reasons an action is illegal (empty = legal). |
| `RuleEngine.apply(state, action) -> Transition` | Pure transition: returns a *new* state + notes. Raises `IllegalActionError`. |
| `RuleEngine.blocking_issues(state)` | Hidden info / manual steps that must be resolved before continuing. |
| `Policy.recommend(state)` | Ranks legal actions. The V1 policy is heuristic. |
| `GameAction` | Immutable action with `kind` and `describe()`. |
| `GameDefinition` | `id`, `name`, `ruleset`, `create_engine`, `create_policy`, `parse_state`, `register_cli`. |

`tests/test_core.py::test_game_agnostic_loop_uses_only_core_contracts` shows the loop a
future agent runs: ask a policy → check the action is in `legal_actions` → `apply` →
stop when `blocking_issues` says hidden information is needed.

## Adding another card game

1. Create `src/card_game_engine/games/<id>/` with its own `data/`, `rules/`, `policy/`, `cli/`.
2. Implement a rule engine with `legal_actions`, `check`, `apply` (returning
   `core.Transition`) and `blocking_issues`, plus at least one policy with `recommend`.
3. Expose `GAME = GameDefinition(id="<id>", ...)` in the package `__init__.py`.
4. Add `"<id>": "card_game_engine.games.<id>"` to `_GAME_MODULES` in `core/registry.py`.
5. Add its JSON files to `[tool.setuptools.package-data]` in `pyproject.toml`.
6. Put tests in `tests/<id>/`. `cardgame <id> ...` will work immediately.

## Hidden information policy

A rule engine never guesses. In Dominion, `+N Cards` increments `pending_draws`. Effects
with player choices or hidden knowledge (Chapel, Remodel, Witch's attack, Vassal's reveal...)
go to `unresolved`. While either is non-empty, `legal_actions()` returns `[]` and the CLI
asks the user what happened. So every state the engine reasons about is *true*.

## How future phases plug in (not implemented yet)

- **LLM tool-calling agent:** expose `get_game(id)`, `legal_actions`, `check`, `apply`,
  `recommend` as tools. Actions already serialise to JSON (`action_to_dict` / `action_from_dict`).
- **Search planner / Monte Carlo:** another `Policy` that uses `legal_actions` + `apply`
  and a sampler for hidden draws. No rule-engine changes needed.
- **Screenshot recognition / external game adapters:** produce a snapshot dict and pass it
  through `GameDefinition.parse_state` (+ the game's validation).
- **General decision environments:** `parse_state` + `legal_actions` + `apply` already form
  an environment interface (observation, action space, transition).

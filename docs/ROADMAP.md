# Roadmap

*Back to the [README](../README.md).*

## Done

**v1.1: rule engine and heuristic policy**

- Dominion 2E Base Set rules from researched sources ([RULES_RESEARCH.md](RULES_RESEARCH.md)), legal
  moves with explanations, snapshot validation.
- Transparent heuristic policy, `recommend` / `legal` / `validate` / `interactive` commands.

**v1.2: lookahead planner**

- Search-based planner: game-agnostic expectimax in `core/search.py`, Dominion model in `planning/`,
  using only `legal_actions` + `apply` from the rule engine.
- Exact draw probabilities from a draw/discard knowledge model (`rules/zones.py`), with reshuffles,
  and seeded Monte Carlo fallback with standard errors.
- Objectives with provenance labels (exact / expected / monte_carlo / heuristic / lower_bound / partial),
  `cardgame dominion plan` / `compare`, examples and a benchmark against the heuristic.

**v1.3: game tracking and belief state**

- Session tracking from the standard starting deck (or a partial mid-game snapshot): typed events,
  exact Bayesian belief over hidden zones (weighted worlds), contradiction vs incomplete-history
  detection, correct/undo/reset with full replay (`inference/`, `core/session.py`).
- Exact cross-turn next-hand predictions (reshuffles, purchases, correlations between turns).
- Planner: multi-hand horizon `hands:N` + `future_money` objective, explicit tie reporting.
- `cardgame dominion session ...` verbs and an interactive session REPL.

**Unreleased: deck analysis and draw odds**

- Deck composition per zone by card and type; exact P(0) / P(≥1) / P(=k) for any number of draws;
  guaranteed min/max counts; reshuffle-aware (`analyze_deck`, `analyze_draw`, `get_guaranteed_cards`).
- `cardgame dominion analyze`, `session deck|draws|guaranteed`, and a `target:<T>>=<K>` planner objective.
- Ranges stay honest when cards are unknown: they always contain the true answer.

## Next

1. **One game file for `interactive` and `session`.** Both currently track a game across turns:
   `interactive` keeps a per-turn `history` in a snapshot and gives buy advice, while `session` keeps an
   event log with a belief about your hidden cards. The plan is to rebuild `interactive`'s multi-turn
   loop on top of `Session`, so one save file works with both commands.
2. **Exact belief-aware planning over mixtures**: search over the weighted worlds directly instead of
   the conservative `?` approximation, with decisions that depend on newly observed cards.
3. **Opponent modelling**: opponents' turns, attacks on you, Moat reactions, Supply depletion.
4. **LLM agent orchestration**: expose `initialize_session` / `record_event` / `predict_next_hand` /
   `plan_next_action` plus `legal_actions` / `check` / `apply` as tools (all JSON-serialisable).
   The agent proposes, the rule engine checks.
5. **More card games**: new plugins under `games/` (see [ARCHITECTURE.md](ARCHITECTURE.md#adding-another-card-game)).
6. **Screenshot-based state recognition**: produce snapshots from images via `parse_state`.
7. **External game adapters**: e.g. read state from online play logs.
8. **General decision environment**: `parse_state` + `legal_actions` + `apply` as an env API.
9. Dominion: 5–6 players, Intrigue and later expansions.

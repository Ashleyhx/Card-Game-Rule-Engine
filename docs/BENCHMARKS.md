# Benchmark: lookahead planner vs heuristic policy

Script: [`benchmarks/planner_vs_heuristic.py`](../benchmarks/planner_vs_heuristic.py) (standard library only).

```bash
python benchmarks/planner_vs_heuristic.py          # table
python benchmarks/planner_vs_heuristic.py --json   # machine-readable
python benchmarks/planner_vs_heuristic.py --examples stranded_draw_vs_money opening_5_2
```

## What is measured

For every bundled example snapshot with enough information:

- **heuristic first action**: `HeuristicPolicy.recommend(state).best` (the V1 policy);
- **planner first action**: `plan_next_action(state)` with the `money` objective, horizon `turn`,
  `max_depth=8`, `max_nodes=20000`, `seed=0`;
- **E[$] for both actions**, computed by the same evaluator (`compare_actions` with the same settings):
  the expected coins generated **this turn** (coins left + cost of cards bought), by exact enumeration
  of the draws given the snapshot's zone knowledge;
- **gap** = planner value - heuristic value, the **verdict**, the planner's runtime and nodes expanded.

Verdicts: `better` / `equal` / `worse` by the measured values; `equal` also when both pick the same
action. When a value is a **lower bound** (`>=`: cards of unknown identity are counted as $0), a
comparison is only counted as better/worse if the bound decides it; otherwise it is `inconclusive`.

Skipped: snapshots whose draw pile size is unknown (information level `insufficient`), because draws
cannot be modelled at all there.

## Results (measured 2026-11-13, Python 3.13, 2-CPU Linux container)

| Example | Heuristic first action | Planner first action | E[$] heuristic | E[$] planner | Gap | Verdict | Planner time | Nodes |
|---|---|---|---:|---:|---:|---|---:|---:|
| endgame_penultimate_province | Play all Treasures | Play all Treasures | 8.00 | 8.00 | +0.00 | equal | 0.014 s | 38 |
| endgame_winning_province | Play all Treasures | Play all Treasures | 8.00 | 8.00 | +0.00 | equal | 0.011 s | 38 |
| multi_buy_festival | Play all Treasures | Play all Treasures | 10.00 | 10.00 | +0.00 | equal | 0.568 s | 1142 |
| opening_5_2 | Play all Treasures | Play all Treasures | 5.00 | 5.00 | +0.00 | equal | 0.010 s | 34 |
| ordered_deck_known | Play Village | Play Village | 8.00 | 8.00 | +0.00 | equal | 0.038 s | 129 |
| partial_knowledge | Play Smithy | Play Smithy | >=7.80 | >=7.80 | +0.00 | equal | 0.077 s | 258 |
| reshuffle_next_hand | Buy Witch | Buy Gold | 6.00 | 6.00 | +0.00 | equal | 0.010 s | 34 |
| small_deck_village_smithy | Play Village | Play Village | 5.20 | 5.20 | +0.00 | equal | 0.099 s | 341 |
| stranded_draw_vs_money | Play Smithy | Play Militia | 3.00 | 5.00 | +2.00 | better | 0.021 s | 82 |
| three_pile_ending | Play all Treasures | Play all Treasures | 3.00 | 3.00 | +0.00 | equal | 0.005 s | 18 |
| throne_room_moneylender | Play Throne Room on Smithy | Play Moneylender (trash a Copper for +$3) | >=2.00 | 4.00 | +2.00 | inconclusive | 0.031 s | 103 |
| village_smithy_action | Play Village | Play Village | >=2.00 | >=2.00 | +0.00 | equal | 0.017 s | 63 |
| witch_midgame_buy | Play all Treasures | Play all Treasures | 7.00 | 7.00 | +0.00 | equal | 0.010 s | 36 |

Skipped: `saved_game_turn_3`, `unknown_deck_minimal` (draw pile size unknown).

**Totals over 13 snapshots: planner better 1, equal 11, worse 0, inconclusive 1. Mean planner
runtime 0.070 s; 2316 nodes in total.** Runtimes depend on the machine; values and nodes are
deterministic for the fixed seed.

## How to read this (honest notes)

- The measure is **expected coins this turn**. It is not the whole game: it ignores which card is
  bought, deck quality, VP, opponents and future turns. A higher number means "more coins this
  turn", nothing more. The heuristic optimises other things too (it is a strategy policy), so these
  numbers are not a verdict on which policy plays Dominion better.
- **Small, hand-made sample.** 13 snapshots. Five of them (`small_deck_village_smithy`,
  `stranded_draw_vs_money`, `ordered_deck_known`, `reshuffle_next_hand`, `partial_knowledge`) were
  written for Phase 2, and `stranded_draw_vs_money` was built specifically to show a case where
  greedy action ranking strands an Action. The one `better` row is that case.
- `reshuffle_next_hand`: the actions differ (Buy Witch vs Buy Gold) but coins this turn are the
  same, so it counts as `equal`. The `money` objective does not judge purchases. Under the
  `next_hand_money` objective the planner prefers Buy Gold (E[next-hand $] = 5.00 vs 4.20 for no buy).
- `throne_room_moneylender`: only `deck_count` is known, so the drawn cards are unknown and the
  heuristic's line (Throne Room on Smithy draws 6 cards) is valued at a lower bound of $2. The real
  value may be higher than the planner's $4, so the comparison is `inconclusive`.
- With a lower bound on both sides (`partial_knowledge`, `village_smithy_action`) both policies chose
  the same action, so nothing is compared.

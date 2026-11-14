# Lookahead planner (Phase 2)

*Reference for `cardgame dominion plan` / `compare`. Back to the [README](../README.md).*

`cardgame dominion plan` looks ahead through the rest of your turn (optionally until your next hand is
drawn) and picks the action with the best **expected value**. Card effects come only from the rule
engine (`legal_actions` / `apply`); the planner adds **chance nodes** for draws, whose outcomes it
enumerates exactly (multivariate hypergeometric, grouped by multiset), including the reshuffle when
the draw pile runs out. It falls back to seeded Monte Carlo sampling only when a draw has more than
200 distinct outcomes. It never invents hidden cards: a card of unknown identity stays `"?"`.

## Describing your piles (snapshot fields, all optional)

```jsonc
"player": {
  ...,
  "draw_pile":    {"order": ["Gold", "Estate", "Silver"]}                    // order known, top first
               |  {"composition": {"Copper": 3, "Estate": 2}}                 // cards known, order unknown
               |  {"top": ["Gold"], "composition": {"Copper": 2}, "unknown_count": 2},  // partial
  "discard_pile": {"composition": {"Copper": 2}, "unknown_count": 0}
},
"planning": {"horizon": "turn"}          // or "next_hand" / "hands:N" (1-4): default horizon for the planner
```

Without `draw_pile`, the planner derives what it can by exact subtraction (`owned_cards` minus hand,
play and a known discard pile), else uses `deck_count` as that many unknown cards. The derivation is
reported in the output. The **information level** of the draw pile is one of:

| Level | Meaning |
|---|---|
| `fully_known_order` | the whole draw pile order is known: draws are deterministic |
| `known_composition` | the cards are known, the order is not (a known top prefix is allowed) |
| `partially_known` | some cards are unknown (`unknown_count > 0`, or only `deck_count`) |
| `insufficient` | the draw pile size is unknown: draws stop at a boundary |

## Objectives and provenance labels

| `--objective` | Value | Kind |
|---|---|---|
| `money` (default) | E[coins generated this turn] (coins left + cost of cards bought) | exact; does **not** judge which card you buy |
| `threshold:N` | P(at least $N this turn) | exact probability |
| `next_hand_money` | E[coins in the hand drawn after Clean-up] (needs horizon `next_hand` or `hands:N`) | exact |
| `future_money` | E[total Treasure coins over the N projected hands] (horizon `hands:N`) | exact (Monte Carlo over budget; `?` = $0, lower bound) |
| `purchase` | heuristic value of the cards bought (V1 policy weights) | heuristic score, not coins or a probability |

Every value carries a provenance label: `exact` (no randomness), `expected` (probability-weighted
over an exact enumeration), `monte_carlo` (sampled estimate, shown `± stderr`), `heuristic`,
`lower_bound` (the line can reach cards of unknown identity, counted as $0; shown as `>=`) or
`partial` (a player choice the planner does not model, e.g. Workshop's gain, evaluated as if it did
nothing). Threshold probabilities are labelled the same way. When the recommended action beats an
alternative whose value is a lower bound or partial, the output says the comparison is **not
conclusive**.

**Horizons.** `turn` (default) stops at your Clean-up. `hands:N` (1 <= N <= 4; `next_hand` = `hands:1`)
also projects your next N start-of-turn hands with **no decisions in future turns**: each projected
hand goes to the discard pile at its Clean-up and the next one is drawn, reshuffling by the rules.
Per projected hand the output reports E[$] and P(hand >= $5).

**Ties.** When the best and second-best candidates have equal value under the objective (|delta| <= 1e-9),
the result says so (`tie: true`, `tied_actions`, `>> TIE ...`) instead of presenting an arbitrary winner:
the decision is beyond that objective/horizon. Example (real output, trimmed) on this turn-1 snapshot
`t1.json` (First Game Kingdom listed as 10 names, elided here):

```json
{"num_players": 2, "turn_number": 1, "kingdom": ["Cellar", "...", "Workshop"],
 "player": {"phase": "buy", "hand": ["Estate"], "in_play": ["Copper", "Copper", "Copper", "Copper"], "coins": 4,
            "owned_cards": {"Copper": 7, "Estate": 3},
            "draw_pile": {"composition": {"Copper": 3, "Estate": 2}}, "discard_pile": {"composition": {}}}}
```

```text
$ cardgame dominion compare t1.json --horizon next_hand --objective next_hand_money --actions "Buy Silver" "End turn (Clean-up)"
>> TIE under objective next_hand_money (horizon next_hand): Buy Silver, End turn (Clean-up)    E[next-hand $] = 3.00
   No difference under this objective/horizon - the decision is beyond it (try another objective or a longer horizon).

$ cardgame dominion compare t1.json --horizon hands:2 --objective future_money --actions "Buy Silver" "End turn (Clean-up)"
>> RECOMMENDED: Buy Silver    E[future $] = 7.09 [expected; search exact]
   1. Buy Silver                                   E[future $] = 7.09            [expected; search exact]
          hand 1: E[$] = 3.00, P($>=5) = 0.00  (exact)
          hand 2: E[$] = 4.09, P($>=5) = 0.35  (exact)
   2. End turn (Clean-up)                          E[future $] = 6.50            [expected; search exact]
          hand 1: E[$] = 3.00, P($>=5) = 0.00  (exact)
          hand 2: E[$] = 3.50, P($>=5) = 0.08  (exact)
```

The next hand is already fixed (the 5 cards left on the draw pile), so the Silver only shows up in
hand 2, after the reshuffle.

## Examples (actual output, trimmed)

```text
$ cardgame dominion plan -e stranded_draw_vs_money
Horizon: turn    Method: exact enumeration of draws
Information level: known_composition
>> RECOMMENDED: Play Militia    E[$] = 5.00 [exact; search exact]
     Threshold probabilities: P($>=5)=1.00, P($>=6)=0.00, P($>=8)=0.00  (exact)
     Top future outcomes (exact):
       p=1.00  coins=$5, bought Duchy
     Principal line: Play Militia -> End Action phase (go to Buy phase) -> Play all Treasures -> Buy Duchy -> End turn (Clean-up)

Alternatives:
   2. Play Smithy                                  E[$] = 3.00            [expected; search exact]
   3. End Action phase (go to Buy phase)           E[$] = 3.00            [exact; search exact]

Why:
  - Recommended Play Militia: E[$]=5.00 vs next best Play Smithy: 3.00 (difference +2.00).
...
Search:
  nodes expanded 82 (budget 20000), max depth reached 5 (limit 8), budget exhausted: no
  chance nodes 1 (exact 1, sampled 0), cache hits 0, seed 0
Limitations:
  - The 'money' objective does not judge which card to buy (coins generated do not depend on it; the
    purchase is only a tie-break), so the Buy in the line is not buy advice: use the 'purchase'
    objective (--objective purchase, heuristic) or `cardgame dominion recommend` for that.
...
```

```text
$ cardgame dominion compare -e small_deck_village_smithy
Side by side (best first):
   1. Play Village                                 E[$] = 5.20            [expected; search exact]
        Threshold probabilities: P($>=5)=0.80, P($>=6)=0.40, P($>=8)=0.00  (exact)
        Top future outcomes (exact):
          p=0.40  coins=$5, bought Duchy
          p=0.40  coins=$6, bought Gold
          p=0.20  coins=$4, bought Militia
   2. Play Smithy                                  E[$] = 4.40            [expected; search exact]
        Threshold probabilities: P($>=5)=0.50, P($>=6)=0.10, P($>=8)=0.00  (exact)
...
```

```text
$ cardgame dominion compare -e throne_room_moneylender --actions "Play Moneylender (trash a Copper for +$3)" "play throne room on smithy"
Why:
  - Play Moneylender (trash a Copper for +$3) (E[$]=4.00) vs Play Throne Room on Smithy (>= 2.00): difference +2.00; ...
  - Not conclusive: Play Throne Room on Smithy's E[$] is only a lower bound (unknown cards counted as $0),
    so it may be as good as or better than Play Moneylender (trash a Copper for +$3).
```

```bash
cardgame dominion plan -e reshuffle_next_hand --horizon next_hand --objective next_hand_money --json
# -> recommended "Buy Gold", expected_value 5.0 (3 cards from {Copper 3, Silver 2, Gold}), alternatives Buy Silver 4.5, End turn 4.2
cardgame dominion plan -e small_deck_village_smithy --objective threshold:6
```

Options shared by `plan` and `compare`: `STATE` (or `--state PATH`, or `-e NAME`), `--json`
(the full `to_dict()` result), `--objective`, `--horizon turn|next_hand|hands:N`, `--depth` (decision steps,
default 8), `--nodes` (node budget, default 20000), `--samples` (Monte Carlo samples per oversized
chance node, default 32), `--seed` (default 0; results are deterministic for a seed), `--top`
(outcomes/alternatives shown). `compare --actions` takes action labels as printed by
`cardgame dominion legal` (case-insensitive); an unknown label lists the legal ones.
`python -m card_game_engine dominion plan ...` works too.

A benchmark against the heuristic is in [`docs/BENCHMARKS.md`](BENCHMARKS.md)
(`python benchmarks/planner_vs_heuristic.py`).

## Planner limitations

- Decisions are searched for the current turn only; `hands:N` projects later hands without decisions
  in those turns. Opponents' turns are not simulated and attacks only affect opponents (cleared).
- Player sub-choices (Chapel, Cellar, Remodel, Mine, Workshop, Artisan, Harbinger, Vassal, Sentry,
  Poacher...) stop the line at a `manual:<card>` boundary (value `partial`). Library keeps every card.
- The search is depth- and budget-limited: results are best-under-budget, never claimed optimal;
  budget cut-offs and sampling are reported.
- `money` ignores what you buy; `purchase` uses the V1 heuristic weights.

---

# Session tracking & belief state (Phase 2b)

*Reference for `cardgame dominion session ...`. Back to the [README](../README.md).*

Instead of describing a snapshot every turn, you can track one real game from the start. The engine
loads the standard starting deck (7 Copper, 3 Estate, shuffled: composition known, order unknown),
and you record **only what it cannot deduce**: your hand when it is uncertain, your plays and buys,
cards gained/trashed/discarded, and Supply counts you want to correct. The session file is just the
event log (`Session.to_dict()` JSON); every command replays it from the start, so results are
reproducible, and writes are atomic.

What the engine knows, and how it labels it:

| Label | Meaning |
|---|---|
| `observed` | you entered it (hand, plays, buys, gains, trashes, Supply counts) |
| `deduced` | certain consequence of the rules and the history (e.g. "the draw pile is exactly 3 Copper, 2 Estate", or a start-of-turn hand the draw pile forces) |
| `probabilistic` | an exact mixture over hidden states (weighted "worlds", exact fractions); draws are without replacement and correlated across turns |
| `unknown` | identity or size not known (shown as `?`; never guessed) |

A **contradiction** (an observation no hidden state allows, more copies than you own, an unaffordable
buy) is rejected with exit code 1 and the file is left unchanged. An **incomplete** history (unobserved
draws, a mid-game start, untracked opponents' gains) is not an error: it is listed under
"Incomplete history" and probabilities stay honest about it.

Verbs (`cardgame dominion session <verb> FILE ...`; all read verbs and mutating verbs take `--json`):

| Verb | Records / does |
|---|---|
| `start [--players N] [--kingdom first\|"A, B, ..."] [--from-snapshot S.json] [--force]` | new session (opening hand pending) |
| `hand "4 Copper, Estate"` | your hand (resolves the start-of-turn draw; otherwise checks it) |
| `draw "Silver"` / `draw --count N` | mid-turn draws, seen or unseen |
| `play CARD [--throne-target X] [--trash-copper]` | a play, through the rule engine |
| `buy CARD` | a purchase (Treasures in hand are played first if needed) |
| `gain CARD [--to discard\|hand\|deck]`, `trash CARD [--from hand\|in_play\|discard]`, `discard "cards"` | other card movements (e.g. Workshop, Chapel, an opponent's Militia) |
| `supply PILE=N` | an observed Supply count |
| `end-turn` | Clean-up; the next 5 cards become the pending hand |
| `show`, `predict [--top K]`, `plan [planner flags]`, `history` | belief, next-hand distribution, lookahead planner, event log |
| `undo`, `correct INDEX <verb> [args]`, `reset` | fix the log (the whole history is replayed and re-validated) |
| `interactive` | a small REPL with the same verbs without FILE (`hand 4 Copper, Estate`, `buy Silver`, `predict`, `help`, `quit`); saves after each change |

The spec workflow, as actually run (output trimmed):

```text
$ cardgame dominion session start game.session.json
Initial deck loaded: Copper x7, Estate x3 (shuffled: composition known, order unknown)
Turn 1 | Phase ACTION | Actions 1 | Buys 1 | $0 | pending draws 5
Hand: 5 card(s) to draw - not determined: 4 possible hands, P(reshuffle) 0 (see 'predict'; enter it with: hand "...")
$ cardgame dominion session hand game.session.json "4 Copper, Estate"
Recorded #0: hand Copper x4, Estate
Draw pile (deduced, 5 cards, order unknown): Copper x3, Estate x2
Next hand (turn 2, if nothing else happens this turn): known exactly - Copper x3, Estate x2
$ cardgame dominion session buy game.session.json Silver
Recorded #1: buy Silver
Discard (deduced, 1 card): Silver
$ cardgame dominion session end-turn game.session.json
Recorded #2: end-turn
Turn 2 | Phase ACTION | Actions 1 | Buys 1 | $0 | pending draws 5
Hand (deduced): Copper x3, Estate x2
  (the 5 pending draw(s) are determined by the draw pile - no need to enter them)
$ cardgame dominion session buy game.session.json Silver
Recorded #3: buy Silver
Next hand (turn 3, if nothing else happens this turn): 12 possible hands (exact), P(reshuffle) 1, E[$] 55/12 = 4.583 (see 'predict')
$ cardgame dominion session end-turn game.session.json
Hand: 5 card(s) to draw - not determined: 12 possible hands, a reshuffle is certain (see 'predict'; enter it with: hand "...")
Discard (deduced, 12 cards): Copper x7, Estate x3, Silver x2
$ cardgame dominion session predict game.session.json --top 4
Start-of-turn hand - turn 3, 5 cards, method exact, hidden states 1
P(reshuffle while drawing): 1
12 possible hands (most likely first):
  35/132 = 0.265       Copper x3, Estate, Silver
  7/44 = 0.159         Copper x2, Estate x2, Silver
  35/264 = 0.133       Copper x4, Estate
  35/264 = 0.133       Copper x3, Estate x2
  41/132 = 0.311       (8 other hands; use --top)
  Sum over all 12 hands: 1
E[$ from Treasures]: 55/12 = 4.583
$ cardgame dominion session hand game.session.json "3 Copper, Estate, Silver"
Draw pile (deduced, 7 cards, order unknown): Copper x4, Estate x2, Silver
$ cardgame dominion session end-turn game.session.json
$ cardgame dominion session predict game.session.json --top 3
Start-of-turn hand - turn 4, 5 cards, method exact, hidden states 1
  8/21 = 0.381         Copper x3, Estate, Silver
  2/7 = 0.286          Copper x2, Estate x2, Silver
  4/21 = 0.190         Copper x3, Estate x2
  1/7 = 0.143          (2 other hands; use --top)
  Sum over all 5 hands: 1
P(at least one):
  Copper       1
  Estate       20/21 = 0.952
  Silver       5/7 = 0.714
$ cardgame dominion session hand game.session.json "3 Copper, Silver x2"
Rejected (inconsistent): Event #7 (observe_hand): impossible card counts in the observed hand: 2 Silver (at most 1 possible)
The session file was not changed.                                       # exit code 1
```

`plan` runs the lookahead planner on the tracked state (with the same flags as `cardgame dominion plan`,
e.g. `--horizon hands:2 --objective future_money`); a hand that is pending but deduced is used as is.
`predict --json` returns `HandPrediction.to_dict()` (exact fractions as `"a/b"` strings).

**Starting mid-game.** `session start FILE --from-snapshot S.json` takes a (partial) snapshot in the
usual format. Only what it states or implies is used; the rest stays unknown:

```text
$ cardgame dominion session start mid.json --from-snapshot partial_knowledge.json
Started from snapshot partial_knowledge.json: history before it is unknown (not invented).
Turn 10 | Phase ACTION | Actions 1 | Buys 1 | $0
Hand (observed, 5 cards): Copper x2, Militia, Silver, Smithy
Draw pile (unknown, 6 cards, top first: Gold, rest in unknown order): Copper x2, Estate, Gold + 2 unknown ('?')
Discard (unknown, 3 cards): Copper x2 + 1 unknown ('?')
```

## Session limitations

- Single-player tracking: only your own deck is tracked. Opponents' effects on you (Militia, Witch,
  Bandit...) must be recorded manually with `discard` / `gain` / `trash`.
- The Supply is never used to infer your own gains; other players' buys are not tracked, so record
  counts you care about with `supply PILE=N`.
- If the number of hidden states would exceed `max_worlds` (default 5000), they are collapsed to what
  is certain and the rest becomes `?` (flagged; predictions become lower bounds).
- The planner works on one state: with a mixture of hidden states, cards whose location is uncertain
  are passed as `?` (a conservative approximation, reported in the plan's notes); it refuses to plan
  while the current hand is undetermined.
- A purchase recorded while your hand is still unknown is used as evidence about that hand: hands
  that could not afford it are ruled out, and the remaining hands keep their relative probabilities
  (this assumes what you chose to buy did not depend on which affordable hand you had). Enter the hand
  first to avoid this assumption.

---

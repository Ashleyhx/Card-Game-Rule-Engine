# Session tracking & belief state

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

## Deck analysis & draw odds

Three read-only queries over the tracked belief (`card_game_engine.games.dominion.inference`); none of
them changes the session, and every result is a frozen dataclass with `to_dict()` (exact values as
`"a/b"` strings next to floats):

| Function | Answers |
|---|---|
| `analyze_deck(session) -> DeckAnalysis` | per zone (`draw_pile`, `discard_pile`, `hand`, `in_play`, `owned`, `trash`), per card and per type: `certain` (set only for a deterministic count), `min`/`max`, exact `expected`, `status` (`observed` / `deduced` / `probabilistic` / `unknown`), `P(count = c)` for probabilistic counts, the number of `?` cards and the known top order of the draw pile |
| `analyze_draw(session, draw_count, target, at="auto") -> DrawAnalysis` | `P(X = k)` for X = number of target cards among the next `draw_count` draws, `P(none)`, `P(at least k)`, `E[X]`, guaranteed min/max, `P(reshuffle)`, `P(fewer cards drawn)`, method, notes |
| `get_guaranteed_cards(session, draw_count=None, at="auto") -> GuaranteedCards` | min/max of every card and type in the next `draw_count` draws (default: the hand size), the guarantees ("at least 2 Copper") and owned cards that cannot be drawn |

**Targets** are a card name or a type (`Treasure`, `Action`, `Victory`, `Curse`, `Attack`, `Reaction`),
case-insensitive; a card name wins over a type name (`Curse` is the card). Typos get suggestions.
The pure versions (`Target`, `parse_target`, `draw_count_distribution`, `guaranteed_range`) live in
`planning/analysis.py` and work on weighted `(weight, DrawPileKnowledge, DiscardKnowledge)` sources.
They relabel every card as match / non-match / `?` and run the usual `enumerate_draws` (so reshuffles
and short draws follow the same rules as everywhere else); a seeded Monte Carlo fallback, labelled
`monte_carlo`, is used only if even the relabelled outcomes exceed the enumeration limit.

**`at`**: `now` draws from the current piles (pending draws are the first draws); `next_turn` draws after
this turn's Clean-up, with the same assumptions as `predict` (pending draws unseen, no further plays,
hand and play to the discard pile); `auto` (default) picks what `predict` picks - the pending
start-of-turn hand, else the next turn. The `draw_count` draws are **continuous draws, no Clean-up
between them** (drawing 7 means 5 + 2 more, reshuffling if needed); use the planner's `hands:N` horizon
for multi-turn projections.

**Unknown cards.** A `?` card never counts as a match, so with `?` cards the values are **bounds**
(`bounded: true`): the lower value counts no `?` as the target, the upper (`upper` in JSON) counts every
drawn `?` as the target, and the truth lies in `[lower, upper]` for `P(X >= k)` and `E[X]` (`max`
also counts `?`). If the draws reach a pile whose **size** is unknown (e.g. a snapshot without
`deck_count`/`owned_cards`), the result is `determinable: false`: no probabilities are given (null in
JSON), only the min/max bounds.
`get_guaranteed_cards` lists only identities known to exist: a card that only a `?` could be is not enumerated.

CLI: `session deck FILE`, `session draws FILE --count N --target T [--at now|next-turn]`,
`session guaranteed FILE [--count N] [--at ...]` (all with `--json`), and the stateless
`cardgame dominion analyze STATE|-e NAME [--count N] [--target T] [--at ...] [--json]` (with `--target`:
the draw distribution; without: guaranteed cards). An unknown target is a usage error (exit 2). As run:

```text
$ cardgame dominion session start game.session.json
$ cardgame dominion session hand game.session.json "4 Copper, Estate"
$ cardgame dominion session buy game.session.json Silver
$ cardgame dominion session guaranteed game.session.json --count 7
Cards in the next 7 card(s) - next turn, after this turn's Clean-up (turn 2), method exact, hidden states 1
Guaranteed: at least 3 Copper, at least 2 Estate, at least 4 Treasure, at least 2 Victory
  Card / type          min  max
  Copper                 3    5
  Estate                 2    3
  Silver                 0    1
  Treasure (type)        4    5
  Victory (type)         2    3
P(reshuffle while drawing): 1    P(fewer than 7 cards drawn): 0
$ cardgame dominion session draws game.session.json --count 7 --target Silver
Silver in the next 7 card(s) - next turn, after this turn's Clean-up (turn 2), method exact, hidden states 1
Guaranteed range: at least 0, at most 1 Silver
  k    P(X = k)
  0    2/3 = 0.667
  1    1/3 = 0.333
P(none): 2/3 = 0.667    P(at least 1): 1/3 = 0.333    E[count]: 1/3 = 0.333
P(reshuffle while drawing): 1    P(fewer than 7 cards drawn): 0
Note: The draws are continuous: no Clean-up happens between them (for multi-turn projections use the planner's --horizon hands:N).
$ cardgame dominion session end-turn game.session.json
$ cardgame dominion session end-turn game.session.json
$ cardgame dominion session draws game.session.json --count 5 --target Treasure --at now
Treasure (type) in the next 5 card(s) - drawn from the current piles (turn 3), method exact, hidden states 1
Guaranteed range: at least 2, at most 5 Treasure
  k    P(X = k)
  2    2/33 = 0.061
  3    4/11 = 0.364
  4    5/11 = 0.455
  5    4/33 = 0.121
P(none): 0    P(at least 1): 1    E[count]: 40/11 = 3.636
P(reshuffle while drawing): 1    P(fewer than 5 cards drawn): 0
$ cardgame dominion analyze -e partial_knowledge --count 3 --target Treasure --at now
Treasure (type) in the next 3 card(s) - drawn from the current piles (turn 10), method exact, hidden states 1
Guaranteed range: at least 1, at most 3 Treasure (the maximum counts '?' cards as matches)
  k    P(X = k)  [lower, upper]
  1    [0, 3/10 = 0.300]
  2    [1/5 = 0.200, 9/10 = 0.900]
  3    [1/10 = 0.100, 3/5 = 0.600]
P(none): 0    P(at least 1): 1    E[count]: [9/5 = 1.800, 13/5 = 2.600]
$ cardgame dominion analyze -e unknown_deck_minimal --target Copper
Copper in the next 5 card(s) - next turn, after this turn's Clean-up (turn ?), method exact, hidden states 1
Not determinable: the draws reach a pile of unknown size. Bounds only: at least 0, at most 5 Copper.
```

(Output trimmed: some `Note:` lines omitted. The 7-card queries are 5 cards from the draw pile, then 2
from the reshuffled 4 Copper, Estate, Silver; turn 3's 5 cards come from the reshuffled 11.)

## Session limitations

- Single-player tracking: only your own deck is tracked. Opponents' effects on you (Militia, Witch,
  Bandit...) must be recorded manually with `discard` / `gain` / `trash`.
- The Supply is never used to infer your own gains; other players' buys are not tracked, so record
  counts you care about with `supply PILE=N`.
- If the number of hidden states would exceed `max_worlds` (default 5000), they are collapsed to what
  is certain and the rest becomes `?` (flagged; predictions become lower bounds).
- A `?` card has no prior, so an observation it may have matched (a drawn or entered card, a trash,
  discard or purchase) has no likelihood: of the pile drawn from only what is certainly left stays
  known (`known - observed` of each card), the rest becomes `?`, and several remaining hidden states are
  merged to what is certain (flagged). This loses information but keeps every range honest.
- The planner works on one state: with a mixture of hidden states, cards whose location is uncertain
  are passed as `?` (a conservative approximation, reported in the plan's notes); it refuses to plan
  while the current hand is undetermined.
- A purchase recorded while your hand is still unknown is used as evidence about that hand: hands
  that could not afford it are ruled out, and the remaining hands keep their relative probabilities
  (this assumes what you chose to buy did not depend on which affordable hand you had). Enter the hand
  first to avoid this assumption.

---

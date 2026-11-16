# Using the Dominion commands

*Back to the [README](../README.md).*

## Commands

All game commands are `cardgame <game> <command>`; `cardgame games` lists games.

| Command | What it does |
|---------|--------------|
| `cardgame dominion recommend STATE.json` | Best move + alternatives with reasons. `--json`, `--top N`, `--brief`, `--example NAME`, `-` for stdin |
| `cardgame dominion legal STATE.json --explain` | Legal moves, and why each unaffordable/empty buy is illegal |
| `cardgame dominion validate STATE.json` | Rule-consistency check (exit code 1 if invalid) |
| `cardgame dominion interactive [--save s.json] [--load s.json]` | Step-by-step snapshot builder, then live play and buy advice, turn after turn |
| `cardgame dominion examples [--show NAME]` | List examples / print one as a template |
| `cardgame dominion cards [NAME] [--kingdom]` | Browse the verified card registry (with sources) |
| `cardgame dominion sources` | Research sources and dates |
| `cardgame dominion plan STATE.json` | Lookahead planner: next action by expected value with exact draw odds ([PLANNER.md](PLANNER.md)) |
| `cardgame dominion compare STATE.json [--actions LABEL ...]` | Every legal first action (or the listed ones) evaluated side by side by the planner |
| `cardgame dominion analyze STATE.json --count N [--target T]` | Exact draw odds and guaranteed counts for a snapshot ([SESSIONS.md](SESSIONS.md#deck-analysis--draw-odds)) |
| `cardgame dominion session <verb> FILE ...` | Track a real game across turns ([SESSIONS.md](SESSIONS.md)) |

Without installing, run any command from the repo root:

```bash
PYTHONPATH=src python -m card_game_engine dominion recommend --example opening_5_2        # macOS/Linux
$env:PYTHONPATH="src"; python -m card_game_engine dominion recommend --example opening_5_2  # PowerShell
```

Example output:

```
>> RECOMMENDED: Play Throne Room on Smithy    [confidence: high]
     [+4.5] Smithy #1: +3 card(s) drawn.
     [+4.5] Smithy #2: +3 card(s) drawn.
     Throne Room: the doubled card does not use Actions.
     Uses your last Action: Moneylender will stay in hand.

Alternatives:
   2. Play Throne Room on Moneylender              score    5.40
   3. Play Smithy                                  score    4.50

Plan for this phase: Play Throne Room on Smithy -> Then: Draw 6 card(s) and add them
to your hand before continuing. Re-run the recommendation with your new hand.
```

The scores and reasons are explained in [HEURISTIC.md](HEURISTIC.md).

## Interactive mode

`cardgame dominion interactive --save game.json` asks about player count, kingdom (the rulebook's
First Game set or your own 10 cards) and your turn number.

- **Turn 1 or 2: quick start.** You still hold the starting 7 Copper + 3 Estate, so it only asks how
  many Coppers are in your hand (and, on turn 2, what you bought on turn 1).
- **Later turns:** changed Supply piles, hand, cards already played, your deck (starting deck +
  changes like `+Silver x2, -Estate`) and opponents' VP. Coins are asked only if you have already
  played cards: Treasures still in your hand are added when the tool plays them.

It then recommends a move. Press Enter to apply it (or type an option's number). If the move draws
cards, it asks what you drew. If a card needs your choices (Chapel, Remodel, Witch's attack...),
it lists the steps and asks for the result. It never guesses.

When your turn ends it asks what each opponent gained or trashed (e.g. `Province`, `-Estate`; blank =
nothing), whether you gained anything on their turns (e.g. `+Curse`), and your new hand, then keeps
going until the game ends. Supply piles and opponents' VP update from what you enter; the kingdom,
your deck and a per-turn list of what you bought carry over. With `--save` the game is written after
every turn, so you can stop and resume with `cardgame dominion interactive --load game.json --save game.json`.

To try a game already in progress:

```bash
cardgame dominion examples --show saved_game_turn_3 > game.json
cardgame dominion interactive --load game.json --save game.json
```

For exact tracking of your hidden cards across turns, use `cardgame dominion session` instead
([SESSIONS.md](SESSIONS.md)).

## Snapshot format (schema v1)

Only `num_players`, `kingdom` and `player.hand` / `player.phase` are required. Anything
omitted is treated as **unknown** and reported as such. It is never made up.

```jsonc
{
  "schema_version": 1,
  "num_players": 2,                        // 2-4
  "turn_number": 9,                        // optional
  "kingdom": ["Cellar", "Chapel", "Village", "Smithy", "Throne Room",
              "Moneylender", "Festival", "Laboratory", "Market", "Witch"],
  "supply": {"Province": 8, "Curse": 6},   // optional; missing piles are assumed full (flagged)
  "trash": ["Copper"],                     // optional
  "player": {
    "phase": "buy",                        // "action" | "buy" ("turn_over" in saved games)
    "hand": ["Gold", "Silver", "Copper", "Copper", "Estate"],   // or {"Copper": 2, ...}
    "in_play": [],                         // cards already played this turn
    "actions": 1, "buys": 1, "coins": 0,   // current remaining resources
    "bought_this_turn": [],
    "owned_cards": {"Copper": 7, "Estate": 3, "Silver": 3, "Gold": 1, "Witch": 1},  // optional: whole deck
    "deck_count": 4, "discard_count": 8     // optional
  },
  "opponents": [{"name": "Opponent", "vp": 0}],  // optional; vp or owned_cards
  "history": [{"turn": 1, "hand": ["Copper", "Copper", "Copper", "Estate", "Estate"], "bought": ["Silver"]}]
                                           // optional; written by `interactive --save`
}
```

Tip: `cardgame dominion examples --show witch_midgame_buy > my_turn.json` gives you a template.
For the lookahead planner you can also describe your draw and discard piles (`player.draw_pile`,
`player.discard_pile`) and a default horizon (`planning`), see [PLANNER.md](PLANNER.md#describing-your-piles-snapshot-fields-all-optional).

## Bundled examples

| Name | Situation |
|------|-----------|
| `opening_5_2` | Turn 1, five Coppers |
| `village_smithy_action` | Action sequencing (Village before Smithy) |
| `throne_room_moneylender` | Which card to Throne |
| `witch_midgame_buy` | $7 with a Witch kingdom: Witch vs Gold (low-confidence close call) |
| `multi_buy_festival` | 3 Buys, $10 to split (3-player) |
| `endgame_penultimate_province` | $8 but behind: Penultimate Province Rule → Duchy |
| `endgame_winning_province` | Last Province wins the game |
| `three_pile_ending` | Ending the game on piles while ahead |
| `unknown_deck_minimal` | Minimal info: shows how uncertainty is reported |
| `saved_game_turn_3` | Saved 3-player game with turn history: continue it in interactive mode |
| `small_deck_village_smithy` | Planner: known 5-card draw pile, Village→Smithy E[$]=5.20 vs Smithy first 4.40 |
| `stranded_draw_vs_money` | Planner: draw pile of Estates, Militia ($5) beats Smithy ($3); the heuristic picks Smithy |
| `ordered_deck_known` | Planner: fully known draw order, Village→Smithy reaches exactly $8 |
| `reshuffle_next_hand` | Planner, `next_hand` horizon: the next hand reshuffles a known discard pile |
| `partial_knowledge` | Planner: known top card + unknown cards, values reported as lower bounds |

## What is supported

**Rules (verified against the publisher's 2E rulebook):** setup pile sizes for 2–4 players;
Action → Buy → Clean-up; Action/Buy accounting; coin pooling across Buys; $0 buys; the 2E rule
that **Treasures cannot be played after buying**; empty piles; game end (Provinces empty or 3
piles empty, checked at end of turn); VP including Gardens.

**Cards:** all 7 basic cards and all 26 Base Set 2E Kingdom cards are in the registry with cost,
types and effects. Support level per card (`cardgame dominion cards`):

- **full**: deterministic effects applied automatically (Village, Smithy, Laboratory, Market,
  Festival, Moat, Merchant's Silver bonus, Moneylender's trash choice, all Treasures/Victory cards)
- **partial**: numeric part automated, the rest is a manual step (Witch, Militia, Throne Room,
  Library, Poacher, Bandit, Bureaucrat, Council Room, Cellar, Harbinger, Vassal, Sentry)
- **manual**: Chapel, Workshop, Remodel, Mine, Artisan (the engine asks you for the result)

Draws are always **pending** until you say what you drew. That's hidden information.

**Not supported:** 5–6 players, expansions, opponents' turns and Moat reactions. The planner works
from the draw/discard piles you describe; cross-turn deck tracking is done by `session`
([SESSIONS.md](SESSIONS.md)). Details: [RULES_RESEARCH.md](RULES_RESEARCH.md).

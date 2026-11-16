# How the Dominion heuristic policy decides

*Reference for `cardgame dominion recommend`. Back to the [README](../README.md).*

Scores are sums of **named components** printed next to each option, e.g.
`[+14.0] Province baseline value in the endgame stage` and
`[-8.0] Penultimate Province Rule: ...`. The components are:

- game **stage** (early → mid → dance → late → endgame) from Provinces left and pile pressure
- basic-card values by stage (Big-Money-style buy rules)
- a 0–10 Kingdom strength rating, discounted late in the game
- **next-shuffle odds** for every buy: how the card changes your chance of hitting $5 / $6 / $8
  (money density and price points, see below)
- deck fit: terminal collision, villages without terminals, duplicates, Curses left for Witch, Moat's
  defence against the attacks in the kingdom, etc.
- endgame safety: winning/losing game-ending buys, Penultimate Province Rule, 3-pile warnings
- Action-phase sequencing: non-terminals before terminals, value of draws/coins/buys/attacks

All numbers live in [`src/card_game_engine/games/dominion/policy/weights.py`](../src/card_game_engine/games/dominion/policy/weights.py).
They are **heuristics, not rules**. Confidence comes from the score gap to the next option. It
drops when the deck or scores are unknown. Every output lists its uncertainties.

The heuristic ranks moves one at a time. To look ahead through the rest of the turn with exact draw
probabilities, use the planner ([PLANNER.md](PLANNER.md)); [BENCHMARKS.md](BENCHMARKS.md) compares the two.

## Next-shuffle odds (money density)

A card's value depends on your deck: Silver turns a 7 Copper + 3 Estate deck's $5 hands from 1 in 12
into 23 in 66; a second terminal in an 11-card deck mostly collides with the first. So for each buy
option (and for buying nothing) the policy computes, **exactly** (fractions), the chance that a
5-card hand from a fresh shuffle of `owned_cards` + the card reaches $5, $6 and $8, plus E[$]:

```
   1. Buy Silver                                   score    6.53
        [+3.5] Silver baseline value in the early stage (opening turns - building economy).
        [+3.0] Next shuffle: P($>=5) 8%->35%, P($>=6) 0%->8%; E[$] 3.50->4.09.
```

The component is `NEXT_SHUFFLE_WEIGHT × Σ weight × (P_after − P_nothing)` over the price points of the
stage (`NEXT_SHUFFLE_PRICE_WEIGHTS`): early $5 (and half for $6), mid $6 and $8, later mostly $8,
endgame little (few shuffles left). Kingdom ratings already include a typical economic contribution,
so Kingdom cards get half of it (`KINGDOM_NEXT_SHUFFLE_SHARE`); for Silver and Gold the stage values
were lowered by the money now counted here (`TYPICAL_MONEY_VALUE`), so nothing is counted twice. Unspent
coins are not penalised as such (Province with $11 is right): opportunity cost shows up as the money
density a cheaper card fails to add. "End turn" shows the odds of the deck as it is. With `--json`,
each option has `next_shuffle` with exact `"a/b"` strings (`buy_nothing`, `after_buy`,
`price_weights`, `score`, `method`).

**The hand model** ([`policy/odds.py`](../src/card_game_engine/games/dominion/policy/odds.py)) is a
steady-state approximation, not your actual draw order: the whole deck is shuffled, 5 cards are drawn,
and the hand is played by a fixed rule built from the rule engine's card data:

1. non-terminal Actions that give something (+Cards, +$, a village's Actions, Merchant) are played
   first; Cellar and Throne Room are never played;
2. then, while you have an Action, the terminal with the highest `+$ + (+Cards) × deck money density`
   (Library draws to 7 cards, setting nothing aside; Moneylender counts as +$2 if you own Copper);
3. all Treasures are played; each Merchant played gives +$1 if a Silver is in hand; each Moneylender
   trashes a Copper from the final hand (if any) for +$3.

Choice effects add no coins (Chapel, Cellar, Remodel, Mine, Workshop, Artisan, Harbinger/Sentry
choices, Vassal's and Poacher's discards, attacks, Bureaucrat's/Bandit's gains). Draws are enumerated
exactly with the planner's grouped hypergeometric helpers (`planning.draws`); cards the rule never
looks at are lumped, which keeps a 25-card deck well under a second for a whole recommendation. If the
enumeration exceeds the planner's limit (`EXACT_LIMIT`), a seeded Monte Carlo estimate is used and the
line says `(Monte Carlo estimate)`. Without `owned_cards` no odds are computed; Silver and Gold then get
their typical money value instead (noted in the output).

**Moat** scores its defence against the strongest attack in the kingdom (`MOAT_DEFENCE`): high
against Witch's Curses, moderate against Bandit/Bureaucrat, small against Militia alone, ×1.5 per
extra opponent. As a card it is a terminal +2 Cards, much weaker than Smithy.

## Sources (paraphrased)

- DominionStrategy wiki, Money strategies (money density, opportunity cost; Province at $8, Gold at
  $6–7, Silver at $3–5): <https://wiki.dominionstrategy.com/index.php/BMU>
- Silver (a Silver/Silver opening reaches $5 on turn 3 or 4 about 91% of the time):
  <https://wiki.dominionstrategy.com/index.php/Silver>
- Moat (+2 Cards terminal draw, clearly weaker than Smithy; most useful against junking attacks):
  <https://wiki.dominionstrategy.com/index.php/Moat>
- Terminal (two terminals in a small deck collide roughly a third or more of the time):
  <https://wiki.dominionstrategy.com/index.php/Terminal>
- "Building the First Game engine" (this kingdom: open Militia/Silver, one Gold, then Markets):
  <https://dominionstrategy.com/2012/07/30/building-the-first-game-engine/>

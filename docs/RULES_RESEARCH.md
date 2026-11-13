# Rule Research — Dominion Second Edition (Base Set)

Research date: **2026-11-13**. All mechanics below are written in our own words; no rulebook text
or card artwork is reproduced. Machine-readable versions live in
`src/card_game_engine/games/dominion/data/rules_base_2e.json` and `src/card_game_engine/games/dominion/data/cards_base_2e.json`.

## Sources

| ID | Source | URL | Used for | Accessed |
|----|--------|-----|----------|----------|
| `rgg-2e-rulebook` | Rio Grande Games — official Dominion 2nd Edition rulebook (PDF, publisher) | https://www.riograndegames.com/wp-content/uploads/2016/09/Dominion2E.pdf | Setup, phases, Buy-phase constraints, end of game, tie-breaks, card clarifications, First Game kingdom | 2026-11-13 |
| `rgg-bigbox-rules` | Rio Grande Games — Big Box rulebook (PDF, publisher) | https://www.riograndegames.com/wp-content/uploads/2017/01/DomBigBox.pdf | Cross-check of the ABC turn structure | 2026-11-13 |
| `card-sheet-2e` | Dominion 2E card sheet (PDF archive, U. Waterloo) | https://cs.uwaterloo.ca/~dtompkin/archive/dtlib/base/Dominion%20(Second%20Edition).pdf | Card costs/types for all 33 cards | 2026-11-13 |
| `ds-card-list` | DominionStrategy — Dominion 2nd Edition Card List | https://dominionstrategy.com/card-lists/dominion-card-list/ | Cross-check of every kingdom card's cost, types and +numbers | 2026-11-13 |
| `ds-wiki-setup` | DominionStrategy Wiki — Gameplay/setup article | https://wiki.dominionstrategy.com/index.php/Buy | Pile-size table by player count (cross-check) | 2026-11-13 |
| `ds-wiki-basic` | DominionStrategy Wiki — Basic cards | https://wiki.dominionstrategy.com/index.php/Basic_card | Basic card costs/values (cross-check) | 2026-11-13 |
| `ds-wiki-curse` | DominionStrategy Wiki — Curse | https://wiki.dominionstrategy.com/index.php/Curse | Curse pile size (10 per player after the first) | 2026-11-13 |
| `ds-wiki-vassal` | DominionStrategy Wiki — Vassal (+ wikiwiki.jp mirror) | https://wiki.dominionstrategy.com/index.php/Vassal | Vassal's +$2 (one summary omitted it; resolved here) | 2026-11-13 |
| `ds-wiki-bureaucrat` | DominionStrategy Wiki — Bureaucrat | https://wiki.dominionstrategy.com/index.php/Bureaucrat | Bureaucrat 2E wording / FAQ | 2026-11-13 |
| `ds-big-money` | DominionStrategy — Big Money / Money strategies (BMU) | https://wiki.dominionstrategy.com/index.php/BMU , https://dominionstrategy.com/big-money | **Strategy only** (policy heuristics, not rules) | 2026-11-13 |
| `ds-ppr` | DominionStrategy — The Penultimate Province Rule | https://dominionstrategy.com/page/41 | **Strategy only** (endgame warning) | 2026-11-13 |

Priority: the publisher rulebook is authoritative. Community sources were used only to cross-check
numbers that were illegible in PDF text extraction (icons for $ amounts do not extract as text) and
for strategy heuristics, which are clearly separated from rules.

## Verified rules (implemented in V1)

### Setup (2–4 players; base box supports 2–4)
- Starting deck: 7 Copper + 3 Estate per player.
- Copper pile: 60 minus 7 per player (46 / 39 / 32).
- Silver 40, Gold 30.
- Estate, Duchy, Province: 8 each with 2 players, 12 each with 3–4.
- Curse: 10 per player beyond the first (10 / 20 / 30).
- Kingdom: 10 piles of 10; a Victory kingdom card (Gardens) uses 8 / 12 like the Victory piles.
- Recommended "First Game" kingdom: Cellar, Market, Merchant, Militia, Mine, Moat, Remodel, Smithy, Village, Workshop.

### Turn structure (Action → Buy → Clean-up)
- You start each turn with 1 Action and 1 Buy and $0 (the rulebook implies one of each).
- Action phase: each Action card played uses one Action. +Actions increase the allowance. A card is fully resolved before the next is played. Playing is always optional.
- Cards that "play" another card (Throne Room, Vassal) do not use up an Action for that play.
- Buy phase: play any number of Treasures, then buy. **Once you have bought a card you may not play more Treasures that turn.**
- Each Buy purchases one card costing ≤ your remaining $. With extra Buys, $ is split across purchases. You may buy a $0 card with $0.
- Bought/gained cards go to your discard pile unless a card says otherwise.
- Clean-up: discard hand and all cards in play, then draw 5. Unspent Actions/Buys/$ are lost.
- Draw/shuffle: when you need to draw and your deck is empty, shuffle your discard pile to form a new deck (not before).

### End of game & scoring
- Checked at the **end of a turn**: the game ends if the Province pile is empty or any **3** Supply piles are empty (4 piles with 5–6 players; not supported in V1).
- VP: Estate 1, Duchy 3, Province 6, Curse −1, Gardens 1 per 10 cards owned (rounded down).
- Most VP wins; ties go to the tied player with fewer turns; still tied → shared victory.

### Card clarifications used (paraphrased)
- **Merchant:** the first Silver you play this turn gives +$1 per Merchant played (multiple Merchants all add to that one Silver).
- **Throne Room:** optional; play an Action from hand twice without spending Actions. Can target another Throne Room.
- **Poacher:** draw first, then discard 1 card per empty Supply pile (as many as you can).
- **Moneylender:** +$3 only if a Copper was actually trashed ("you may" added in 2E).
- **Mine:** "you may" trash a Treasure; the gained Treasure goes to hand and may be played this turn.
- **Library:** draw until 7 in hand; you may set aside Action cards as you draw them; discard those afterwards.
- **Vassal:** +$2; discard top card; if it's an Action you may play it (no Action used).
- **Witch:** Curses are given out in turn order; with no Curses left Witch is still +2 Cards.
- **Remodel / Workshop / Artisan:** gain limits are fixed by the card; extra $ does not raise them. Remodel cannot trash itself.
- **Moat:** reveal from hand when an Attack is played to be unaffected; stays in hand.

## V1 support matrix

The rule engine fully automates what is **deterministic and knowable from the snapshot**; everything
involving hidden information (draws, opponents' hands, top of deck) or player sub-choices is surfaced
as an explicit *manual step* that blocks further legal actions until the user resolves it.

| Level | Cards |
|-------|-------|
| Full | Copper, Silver, Gold, Estate, Duchy, Province, Curse, Gardens (scoring), Village, Smithy, Laboratory, Market, Festival, Moat (action), Merchant (incl. Silver bonus), Moneylender (trash-Copper choice enumerated as two legal actions) |
| Partial (numeric part automated, remainder manual) | Cellar, Harbinger, Poacher, Vassal, Militia, Council Room, Witch, Library, Sentry, Bandit (Gold gain automated), Bureaucrat (Silver gain automated), Throne Room (target enumerated; target's effects applied twice) |
| Manual | Chapel, Workshop, Remodel, Mine, Artisan |

"+N Cards" is always modelled as `pending_draws`: the engine never guesses which cards you draw.

## Unsupported / out of scope in V1
- 5–6 player games (need a second set of basic cards; 4-pile ending).
- Expansions, Events, Landmarks, Duration cards, tokens.
- Opponents' turns and reactions (Moat responses are noted, not simulated).
- Hidden-information inference (deck tracking, shuffle timing).
- Automatic resolution of sub-choices (which cards to trash/discard/gain).

## Assumptions & implementation simplifications
1. If a snapshot omits supply counts, piles are assumed full at their starting size — reported as an assumption in every output.
2. `owned_cards` (your whole deck composition) is optional; without it, VP and deck-fit heuristics are unavailable and confidence drops.
3. Treasure-playing order only matters for Merchant; the engine handles it, and "play all Treasures" is offered as a single convenience action (equivalent to playing each one).
4. Ending the turn from the Action phase is modelled as End Action Phase → End Turn (the Buy phase still happens, possibly with no buy).
5. Throne Room on Moneylender: the trash choice for each play is left as a manual step.
6. Strategy weights in `games/dominion/policy/weights.py` are **heuristics** informed by community strategy writing (Big Money rules, Penultimate Province Rule), not rules.

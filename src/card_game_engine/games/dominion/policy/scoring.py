"""Transparent heuristic scoring of legal actions.

Every score is built from named components with a human-readable reason, so the
policy can always explain *why* an action ranks where it does.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..rules import (
    Action, Buy, CardDef, EndActionPhase, EndTurn, GameState, PlayAction, PlayAllTreasures, PlayTreasure, DominionRuleEngine,
)
from . import weights as W
from .context import GameContext


@dataclass
class Score:
    total: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, value: float, reason: str | None = None) -> None:
        self.total += value
        if reason:
            sign = "+" if value >= 0 else "-"
            self.reasons.append(f"[{sign}{abs(value):.1f}] {reason}" if value else reason)

    def note(self, reason: str) -> None:
        self.reasons.append(reason)

    def warn(self, warning: str) -> None:
        self.warnings.append(warning)


class HeuristicScorer:
    def __init__(self, engine: DominionRuleEngine):
        self.engine = engine

    # ------------------------------------------------------------------
    def score(self, state: GameState, ctx: GameContext, action: Action) -> Score:
        if isinstance(action, PlayAllTreasures):
            return self._score_play_treasures(state)
        if isinstance(action, PlayTreasure):
            s = Score()
            s.add(W.PLAY_TREASURES_SCORE - 1, "Same as playing all Treasures, one at a time.")
            return s
        if isinstance(action, Buy):
            return self._score_buy(state, ctx, action.card)
        if isinstance(action, EndTurn):
            return self._score_end_turn(state, ctx)
        if isinstance(action, EndActionPhase):
            return self._score_end_action_phase(state)
        if isinstance(action, PlayAction):
            return self._score_play_action(state, ctx, action)
        return Score()

    # ------------------------------------------------------------------ Buy phase
    def _score_play_treasures(self, state: GameState) -> Score:
        s = Score()
        reg = self.engine.registry
        total = sum(reg.get(c).treasure_value for c in state.turn.hand if reg.get(c).is_treasure)
        merchant = ""
        if "Silver" in state.turn.hand and not state.turn.silver_played and state.turn.merchants_played:
            total += state.turn.merchants_played
            merchant = f" (incl. +${state.turn.merchants_played} Merchant bonus)"
        s.add(W.PLAY_TREASURES_SCORE, f"Adds ${total}{merchant}; Base-set Treasures have no downside.")
        s.note("Rule: you cannot play Treasures after buying, so play them before any purchase.")
        return s

    def _score_end_turn(self, state: GameState, ctx: GameContext) -> Score:
        s = Score()
        t = state.turn
        if t.buys > 0:
            s.note(f"Pass: spend nothing (${t.coins} and {t.buys} Buy(s) unused). Keeps your deck lean.")
        else:
            s.note("No Buys left - ending the turn is the only option.")
        return s

    def _vp_of(self, card: CardDef, deck_total_after: int | None) -> int:
        if card.special == "gardens":
            return (deck_total_after or 0) // 10
        return card.vp or 0

    def _score_buy(self, state: GameState, ctx: GameContext, name: str) -> Score:
        s = Score()
        card = self.engine.card(name)
        stage = ctx.stage
        deck = ctx.deck
        t = state.turn

        if any(self.engine.card(c).is_treasure for c in t.hand) and not t.bought:
            s.add(-1.0, "You still have unplayed Treasures; play them first (you can't after buying).")

        if not card.kingdom:
            base = W.BASIC_BUY_VALUE[name][stage]
            s.add(base, f"{name} baseline value in the {stage} stage ({ctx.stage_reason}).")
            if name == "Province" and stage == "early" and deck is not None:
                if deck.counts.get("Gold", 0) > 0 or deck.counts.get("Silver", 0) >= 5:
                    s.add(4.0, "Economy already has Gold/5+ Silver, so an early Province is fine (Big Money rule).")
                else:
                    s.note("Big Money rule of thumb: with no Gold and <5 Silver, an early Gold usually beats Province.")
            if name == "Copper" and "Gardens" in state.kingdom:
                s.add(2.0, "Gardens is in the kingdom: every card (even Copper) raises Gardens' value.")
        else:
            self._score_kingdom_buy(state, ctx, card, s)

        self._endgame_adjustments(state, ctx, card, s)
        return s

    def _score_kingdom_buy(self, state: GameState, ctx: GameContext, card: CardDef, s: Score) -> None:
        stage, deck = ctx.stage, ctx.deck
        rating = W.KINGDOM_RATING.get(card.name, 4.0)
        mult = W.KINGDOM_STAGE_MULTIPLIER[stage]

        if card.special == "gardens":
            total_after = (deck.total + 1) if deck else None
            vp_now = (total_after // 10) if total_after else None
            weight = {"early": 0.3, "mid": 0.6, "dance": 1.0, "late": 1.3, "endgame": 1.6}[stage]
            if vp_now is not None:
                s.add(max(rating * mult, 1.6 * vp_now * weight + 1.0),
                      f"Gardens would be worth {vp_now} VP with {total_after} cards (1 VP per 10 cards).")
            else:
                s.add(rating * mult, "Gardens value depends on your deck size (unknown).")
            return

        if mult < 1.0:
            s.add(rating * mult, f"{card.name} strength rating {rating:.1f} x {mult:.2f} ({stage} stage: Kingdom cards matter less as the game nears its end).")
        else:
            s.add(rating, f"{card.name} strength rating {rating:.1f} (early stage: full value).")
        if deck is None:
            s.note("Deck-fit not evaluated (owned_cards unknown).")
            return

        copies = deck.counts.get(card.name, 0) + state.turn.bought.count(card.name)
        bought_terminals = sum(1 for c in state.turn.bought if self.engine.card(c).is_terminal)
        terminals = deck.terminals + bought_terminals
        villages = deck.villages + sum(1 for c in state.turn.bought if self.engine.card(c).is_village)

        if card.is_terminal:
            allowed = max(1, deck.total // 10) + villages
            if terminals >= allowed:
                s.add(-2.5 * (terminals - allowed + 1),
                      f"Terminal collision risk: you have {terminals} terminal Action(s) for {deck.total} cards and {villages} village(s).")
        if card.is_village:
            if terminals == 0:
                s.add(-2.0, "No terminal Actions for a village to support yet.")
            elif villages >= terminals:
                s.add(-1.5, f"Already {villages} village(s) for {terminals} terminal(s).")
        cantrip = card.effects.actions >= 1 and card.effects.cards >= 1
        if copies >= 2 and not cantrip and card.special not in ("gardens",):
            s.add(-1.0 * (copies - 1), f"Diminishing returns: you already have {copies} {card.name}.")

        n = card.name
        if n == "Witch":
            curses = state.supply.get("Curse", 0)
            if curses == 0:
                s.add(-4.0, "No Curses left: Witch is only +2 Cards now.")
            elif copies >= 2:
                s.add(-1.5, "Two Witches usually suffice to win the Curse split.")
            else:
                s.add(1.0, f"{curses} Curses left to hand out.")
        elif n == "Chapel":
            if copies >= 1:
                s.add(-6.0, "One Chapel is enough.")
            elif stage != "early" or deck.total > 14:
                s.add(-4.0, "Trashing pays off mostly in the opening turns.")
        elif n == "Moat":
            if any(self.engine.card(k).is_attack for k in state.kingdom):
                s.add(1.5, "Attacks are in this kingdom; Moat defends.")
        elif n == "Merchant" and deck.counts.get("Silver", 0) >= 2:
            s.add(0.5, "You own several Silvers for Merchant.")
        elif n == "Workshop" and "Gardens" in state.kingdom:
            s.add(2.0, "Workshop/Gardens synergy: gain Gardens or cheap cards.")
        elif n == "Moneylender":
            if copies >= 1:
                s.add(-3.0, "One Moneylender is enough.")
            elif deck.counts.get("Copper", 0) < 5:
                s.add(-2.5, "Few Coppers left to trash.")
        elif n == "Throne Room":
            others = deck.actions - deck.counts.get("Throne Room", 0)
            if others < 3:
                s.add(-2.0, f"Only {others} other Action(s) to Throne.")
        elif n == "Library" and (deck.counts.get("Festival", 0) or "Festival" in state.turn.bought):
            s.add(1.0, "Library + Festival: Festival's +Actions without draw fuel Library.")

    def _endgame_adjustments(self, state: GameState, ctx: GameContext, card: CardDef, s: Score) -> None:
        supply_after = dict(state.supply)
        supply_after[card.name] = supply_after.get(card.name, 0) - 1
        empties_after = [p for p, n in supply_after.items() if n <= 0]
        ends = supply_after.get("Province", 1) <= 0 or len(empties_after) >= self.engine.ruleset.empty_piles_to_end

        deck_total_after = ctx.deck.total + 1 if ctx.deck else None
        if ends:
            if ctx.vp_known:
                my_after = ctx.my_vp + self._vp_of(card, deck_total_after)
                if ctx.deck and card.name != "Gardens" and ctx.deck.counts.get("Gardens"):
                    # adding a card can raise Gardens' value
                    my_after = self.engine.victory_points({**ctx.deck.counts, card.name: ctx.deck.counts.get(card.name, 0) + 1})
                best_opp = ctx.max_opponent_vp
                if my_after > best_opp:
                    s.add(W.WINNING_MOVE_BONUS, f"Ends the game with you ahead ({my_after} vs {best_opp} VP) - a winning buy.")
                elif my_after == best_opp:
                    s.warn(f"Ends the game in a tie ({my_after} VP each); tie-break is fewer turns taken.")
                else:
                    s.add(W.LOSING_MOVE_PENALTY, f"Ends the game while behind ({my_after} vs {best_opp} VP) - you would lose.")
            else:
                s.warn("This buy ends the game at the end of your turn - VP totals are unknown, confirm you are ahead.")
            return

        if card.name == "Province" and state.supply.get("Province", 0) == 2:
            if ctx.vp_known:
                my_after = ctx.my_vp + 6
                best_opp = ctx.max_opponent_vp
                if best_opp + 6 > my_after:
                    s.add(W.PENULTIMATE_PROVINCE_PENALTY,
                          f"Penultimate Province Rule: you'd have {my_after} VP and leave 1 Province; an opponent on {best_opp} VP could buy it, end the game with {best_opp + 6} and win.")
                elif best_opp + 6 == my_after:
                    s.warn("Penultimate Province: an opponent taking the last Province would tie you.")
            else:
                s.warn("Penultimate Province: leaving 1 Province lets an opponent end the game - check the scores.")

        if len(empties_after) == self.engine.ruleset.empty_piles_to_end - 1 and supply_after.get(card.name, 1) == 0:
            s.warn(f"Emptying the {card.name} pile leaves the game one pile away from a 3-pile ending.")

    # ------------------------------------------------------------------ Action phase
    def _score_end_action_phase(self, state: GameState) -> Score:
        s = Score()
        t = state.turn
        playable = [c for c in t.hand if self.engine.card(c).is_action]
        if playable and t.actions > 0:
            s.note("Skip remaining Actions and move to the Buy phase.")
        else:
            s.note("No playable Actions - move to the Buy phase.")
        return s

    def play_value(self, state: GameState, ctx: GameContext, card: CardDef, hand_after: list[str],
                   actions_after: int, s: Score, label: str = "", trash_copper: bool | None = None) -> None:
        """Add the estimated value of resolving ``card`` once to ``s``."""
        e = card.effects
        pre = f"{label}: " if label else ""
        other_actions = [c for c in hand_after if self.engine.card(c).is_action]
        if e.cards:
            s.add(e.cards * W.VALUE_PER_CARD_DRAWN, f"{pre}+{e.cards} card(s) drawn.")
        if e.coins:
            s.add(e.coins * W.VALUE_PER_COIN, f"{pre}+${e.coins}.")
        if e.buys:
            per = W.VALUE_PER_BUY * (1.5 if ctx.stage in ("late", "endgame") else 1.0)
            s.add(e.buys * per, f"{pre}+{e.buys} Buy.")
        if e.actions:
            useful = min(e.actions, len(other_actions) + (1 if e.cards else 0))
            spare = e.actions - useful
            if useful:
                s.add(useful * W.VALUE_PER_USEFUL_ACTION, f"{pre}+Actions that let you play more Action cards.")
            if spare:
                s.add(spare * W.VALUE_PER_SPARE_ACTION, f"{pre}spare +Actions (little to use them on yet).")
        if card.is_attack:
            atk = W.ATTACK_VALUE.get(card.name, 0.5)
            if card.name == "Witch" and state.supply.get("Curse", 0) == 0:
                atk = 0.0
            if atk:
                s.add(atk, f"{pre}attack hurts opponents.")
        if card.gain and state.supply.get(card.gain.card, 0) > 0:
            v = max(0.5, W.BASIC_BUY_VALUE[card.gain.card][ctx.stage] * 0.6)
            s.add(v, f"{pre}gains a {card.gain.card}.")

        n = card.name
        hand = Counter(hand_after)
        late = ctx.stage in ("late", "endgame")
        if n == "Council Room":
            s.add(-0.5, f"{pre}each opponent draws a card.")
        elif n == "Chapel":
            curses = hand["Curse"]
            estates = 0 if late else hand["Estate"]
            coppers = hand["Copper"] if ctx.stage in ("early", "mid") else 0
            k = min(4, curses + estates + coppers)
            per = 1.2 if ctx.stage == "early" else 0.6
            s.add(curses * 1.5 + max(0, k - curses) * per, f"{pre}trash junk in hand ({curses} Curse, {estates} Estate, {coppers} Copper eligible).")
        elif n == "Cellar":
            dead = hand["Estate"] + hand["Duchy"] + hand["Province"] + hand["Curse"] + hand["Gardens"]
            if actions_after == 0:
                dead += len(other_actions)
            s.add(0.8 * dead, f"{pre}cycle {dead} dead card(s).")
        elif n == "Remodel":
            if hand["Gold"] and ctx.stage in ("dance", "late", "endgame"):
                s.add(5.0, f"{pre}Remodel a Gold into a Province.")
            elif hand["Curse"]:
                s.add(2.0, f"{pre}trash a Curse (gain a card up to $2).")
            elif hand["Estate"]:
                s.add(2.0 if not late else 1.0, f"{pre}turn an Estate into a $4 card (or Silver).")
            else:
                s.add(0.8, f"{pre}improve a card in hand.")
        elif n == "Mine":
            if hand["Silver"]:
                s.add(2.0, f"{pre}Silver -> Gold into hand (+$1 this turn).")
            elif hand["Copper"]:
                s.add(1.5, f"{pre}Copper -> Silver into hand (+$1 this turn).")
        elif n == "Workshop":
            s.add(2.0 if not late else 1.2, f"{pre}gain a card costing up to $4.")
        elif n == "Artisan":
            s.add(3.5, f"{pre}gain a $5 card to hand.")
        elif n in ("Harbinger", "Sentry"):
            s.add(0.8, f"{pre}deck control.")
        elif n == "Vassal":
            s.add(0.7, f"{pre}may play a discarded Action for free.")
        elif n == "Library":
            draws = max(0, 7 - len(hand_after))
            s.add(draws * W.VALUE_PER_CARD_DRAWN, f"{pre}draw up to 7 cards ({draws} now).")
        elif n == "Poacher" and ctx.empty_piles:
            s.add(-0.5 * len(ctx.empty_piles), f"{pre}must discard {len(ctx.empty_piles)} card(s).")
        elif n == "Merchant" and hand["Silver"] and not state.turn.silver_played:
            s.add(1.0, f"{pre}+$1 when you play your Silver.")
        elif n == "Moneylender":
            if trash_copper:
                gain = 2.0 + (0.7 if ctx.stage in ("early", "mid") else 0.0)
                s.add(gain, f"{pre}trash a Copper for +$3 (net +$2 this turn{', thins your deck' if gain > 2 else ''}).")
            elif trash_copper is False:
                s.note(f"{pre}no Copper trashed: no +$.")

    def _score_play_action(self, state: GameState, ctx: GameContext, action: PlayAction) -> Score:
        s = Score()
        t = state.turn
        card = self.engine.card(action.card)
        hand_after = list(t.hand)
        hand_after.remove(card.name)

        if card.special == "throne_room":
            if action.throne_target is None:
                s.add(-0.5, "Throne Room with no target does nothing.")
                return s
            target = self.engine.card(action.throne_target)
            hand_after.remove(target.name)
            actions_after = t.actions - 1 + 2 * target.effects.actions
            for i in (1, 2):
                self.play_value(state, ctx, target, hand_after, actions_after, s, label=f"{target.name} #{i}",
                                trash_copper=(target.name == "Moneylender" and hand_after.count("Copper") >= i) or None)
            s.note("Throne Room: the doubled card does not use Actions.")
            gives_actions = target.effects.actions >= 1
        else:
            actions_after = t.actions - 1 + card.effects.actions
            self.play_value(state, ctx, card, hand_after, actions_after, s, trash_copper=action.trash_copper)
            gives_actions = card.effects.actions >= 1

        others = [c for c in hand_after if self.engine.card(c).is_action]
        if gives_actions and others:
            s.add(W.SEQUENCING_BONUS, "Sequencing: play non-terminal Actions first to keep Actions for " + ", ".join(sorted(set(others))) + ".")
        if not gives_actions and t.actions == 1:
            enablers = [c for c in others if self.engine.card(c).effects.actions >= 1]
            if enablers:
                s.add(-W.SEQUENCING_BONUS, f"Sequencing: this uses your last Action; play {sorted(set(enablers))[0]} first.")
            elif others:
                s.note("Uses your last Action: " + ", ".join(sorted(set(others))) + " will stay in hand.")

        if card.manual_steps or (card.special == "throne_room" and self.engine.card(action.throne_target).manual_steps):
            s.note("Part of this card's effect needs your choices; the engine will ask you to resolve it.")
        return s

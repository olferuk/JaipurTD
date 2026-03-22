"""Jaipur game state and logic."""
from __future__ import annotations

import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union

from .cards import (
    BONUS_TOKENS_3, BONUS_TOKENS_4, BONUS_TOKENS_5,
    CAMEL_BONUS, GOODS, PRECIOUS_GOODS, TOKENS,
    GoodType, build_deck,
)

# ---------- Actions ----------

@dataclass(frozen=True)
class TakeOne:
    """Take one good card from the market."""
    good: GoodType

@dataclass(frozen=True)
class TakeCamels:
    """Take all camels from the market."""
    pass

@dataclass(frozen=True)
class TakeMultiple:
    """Exchange cards: take goods from market, give back cards from hand/herd.
    give_goods: goods from hand to return to market
    give_camels: number of camels from herd to return to market
    take_goods: goods to take from market
    """
    take_goods: tuple[GoodType, ...]
    give_goods: tuple[GoodType, ...]
    give_camels: int = 0

@dataclass(frozen=True)
class Sell:
    """Sell cards of one good type."""
    good: GoodType
    count: int

Action = Union[TakeOne, TakeCamels, TakeMultiple, Sell]

# ---------- Player State ----------

@dataclass
class PlayerState:
    hand: Counter = field(default_factory=Counter)  # GoodType -> count (no camels)
    camels: int = 0
    score: int = 0

    @property
    def hand_size(self) -> int:
        return sum(self.hand.values())

    def copy(self) -> PlayerState:
        p = PlayerState()
        p.hand = Counter(self.hand)
        p.camels = self.camels
        p.score = self.score
        return p

# ---------- Game State ----------

@dataclass
class GameState:
    market: list[GoodType] = field(default_factory=list)
    deck: list[GoodType] = field(default_factory=list)
    players: tuple[PlayerState, PlayerState] = field(
        default_factory=lambda: (PlayerState(), PlayerState())
    )
    tokens: dict[GoodType, list[int]] = field(default_factory=dict)
    bonus3: list[int] = field(default_factory=list)
    bonus4: list[int] = field(default_factory=list)
    bonus5: list[int] = field(default_factory=list)
    current_player: int = 0  # 0 or 1
    round_over: bool = False

    @staticmethod
    def new_round(rng: random.Random | None = None) -> GameState:
        """Set up a fresh round."""
        if rng is None:
            rng = random.Random()

        gs = GameState()

        # Token piles (copy so we don't mutate module-level lists)
        gs.tokens = {g: list(TOKENS[g]) for g in GOODS}

        # Bonus tokens (shuffled)
        gs.bonus3 = list(BONUS_TOKENS_3)
        gs.bonus4 = list(BONUS_TOKENS_4)
        gs.bonus5 = list(BONUS_TOKENS_5)
        rng.shuffle(gs.bonus3)
        rng.shuffle(gs.bonus4)
        rng.shuffle(gs.bonus5)

        # Build and shuffle deck
        deck = build_deck()
        rng.shuffle(deck)

        # Remove 3 camels from deck (they start in the market)
        for _ in range(3):
            deck.remove(GoodType.CAMEL)

        # Market: 3 camels + 2 cards from deck
        gs.market = [GoodType.CAMEL] * 3
        for _ in range(2):
            gs.market.append(deck.pop())

        # Deal 5 cards to each player; camels go to herd
        for p in gs.players:
            for _ in range(5):
                card = deck.pop()
                if card == GoodType.CAMEL:
                    p.camels += 1
                else:
                    p.hand[card] += 1

        gs.deck = deck
        gs.current_player = 0
        return gs

    @property
    def cp(self) -> PlayerState:
        """Current player state."""
        return self.players[self.current_player]

    @property
    def op(self) -> PlayerState:
        """Opponent player state."""
        return self.players[1 - self.current_player]

    def _refill_market(self) -> None:
        """Top up market to 5 cards from deck."""
        while len(self.market) < 5 and self.deck:
            self.market.append(self.deck.pop())

    def _empty_piles_count(self) -> int:
        """Count how many good token piles are empty."""
        return sum(1 for g in GOODS if len(self.tokens[g]) == 0)

    def _check_round_over(self) -> None:
        if self._empty_piles_count() >= 3 or len(self.deck) == 0:
            self.round_over = True

    # ---------- Legal actions ----------

    def get_legal_actions(self) -> list[Action]:
        if self.round_over:
            return []

        actions: list[Action] = []
        p = self.cp

        # 1. Take one good from market
        if p.hand_size < 7:
            seen = set()
            for card in self.market:
                if card != GoodType.CAMEL and card not in seen:
                    seen.add(card)
                    actions.append(TakeOne(card))

        # 2. Take all camels from market
        market_camels = self.market.count(GoodType.CAMEL)
        if market_camels > 0:
            actions.append(TakeCamels())

        # 3. Exchange (take multiple goods, give back cards/camels)
        actions.extend(self._exchange_actions())

        # 4. Sell goods
        for good in GOODS:
            count = p.hand[good]
            if good in PRECIOUS_GOODS:
                if count >= 3:
                    for n in range(3, count + 1):
                        actions.append(Sell(good, n))
            else:
                if count >= 2:
                    for n in range(2, count + 1):
                        actions.append(Sell(good, n))
                # Can also sell 1 for non-precious
                if count >= 1:
                    actions.append(Sell(good, 1))

        return actions

    def _exchange_actions(self) -> list[Action]:
        """Generate all valid TakeMultiple actions.
        
        Rules: take N goods from market (non-camel), give back N cards
        (from hand goods + camels). Must take at least 2.
        Can't take and give back the same type (net zero).
        """
        p = self.cp
        market_goods = [c for c in self.market if c != GoodType.CAMEL]
        if len(market_goods) < 2:
            return []

        # Available to give: hand goods + camels
        available_give = sum(p.hand.values()) + p.camels
        if available_give < 2:
            return []

        actions: list[Action] = []

        # Generate subsets of market goods to take (size 2+)
        # For efficiency, work with unique combinations
        market_counter = Counter(market_goods)
        take_combos = self._subsets_from_counter(market_counter, min_size=2)

        for take in take_combos:
            take_count = sum(take.values())
            # Check hand limit: current hand + take - give_from_hand <= 7
            # give_from_hand + give_camels = take_count
            # So: hand_size + take_count - give_from_hand <= 7
            # give_from_hand >= hand_size + take_count - 7 (but >= 0)
            min_from_hand = max(0, p.hand_size + take_count - 7)
            max_from_hand = min(p.hand_size, take_count)
            max_camels = min(p.camels, take_count)

            # Generate give combinations
            give_combos = self._give_combinations(
                p.hand, p.camels, take_count, take, min_from_hand
            )
            for give_goods, give_camels in give_combos:
                actions.append(TakeMultiple(
                    take_goods=tuple(sorted(take.elements(), key=lambda g: g.value)),
                    give_goods=tuple(sorted(give_goods.elements(), key=lambda g: g.value)),
                    give_camels=give_camels,
                ))

        return actions

    def _give_combinations(
        self,
        hand: Counter,
        camels: int,
        need: int,
        taking: Counter,
        min_from_hand: int,
    ) -> list[tuple[Counter, int]]:
        """Generate valid (give_goods, give_camels) pairs.
        
        Can't give back a good type you're also taking (no net-zero swaps).
        """
        # Available hand goods excluding types being taken
        available_hand = Counter()
        for g, c in hand.items():
            if g not in taking:
                available_hand[g] = c

        results: list[tuple[Counter, int]] = []
        available_hand_total = sum(available_hand.values())

        # Try different numbers of camels to give
        for n_camels in range(min(camels, need), -1, -1):
            n_from_hand = need - n_camels
            if n_from_hand < min_from_hand:
                continue
            if n_from_hand > available_hand_total:
                continue
            if n_from_hand == 0:
                results.append((Counter(), n_camels))
                continue
            # Generate subsets of available_hand of size n_from_hand
            for combo in self._subsets_from_counter(available_hand, exact_size=n_from_hand):
                results.append((combo, n_camels))

        return results

    @staticmethod
    def _subsets_from_counter(
        counter: Counter,
        min_size: int = 0,
        exact_size: int | None = None,
    ) -> list[Counter]:
        """Generate unique sub-multisets from a counter."""
        items = [(g, c) for g, c in counter.items() if c > 0]
        results: list[Counter] = []

        def backtrack(idx: int, current: Counter, size: int):
            if exact_size is not None:
                if size == exact_size:
                    results.append(Counter(current))
                    return
                if size > exact_size:
                    return
            elif size >= min_size and size > 0:
                results.append(Counter(current))

            for i in range(idx, len(items)):
                good, max_count = items[i]
                remaining = (exact_size - size) if exact_size else (sum(c for _, c in items[i:]))
                if exact_size and size + remaining < exact_size:
                    break
                for n in range(1, max_count + 1):
                    if exact_size and size + n > exact_size:
                        break
                    current[good] = n
                    backtrack(i + 1, current, size + n)
                del current[good]

        backtrack(0, Counter(), 0)
        return results

    # ---------- Apply action ----------

    def apply_action(self, action: Action) -> GameState:
        """Apply action and return new state (does NOT mutate self)."""
        gs = self.copy()
        p = gs.cp

        if isinstance(action, TakeOne):
            gs.market.remove(action.good)
            p.hand[action.good] += 1
            gs._refill_market()

        elif isinstance(action, TakeCamels):
            n = gs.market.count(GoodType.CAMEL)
            gs.market = [c for c in gs.market if c != GoodType.CAMEL]
            p.camels += n
            gs._refill_market()

        elif isinstance(action, TakeMultiple):
            # Remove taken goods from market
            market_copy = list(gs.market)
            for g in action.take_goods:
                market_copy.remove(g)
            # Add given goods/camels to market
            for g in action.give_goods:
                p.hand[g] -= 1
                if p.hand[g] == 0:
                    del p.hand[g]
                market_copy.append(g)
            p.camels -= action.give_camels
            for _ in range(action.give_camels):
                market_copy.append(GoodType.CAMEL)
            # Take goods into hand
            for g in action.take_goods:
                p.hand[g] += 1
            gs.market = market_copy
            # Market size stays at 5 (exchange is 1:1)

        elif isinstance(action, Sell):
            p.hand[action.good] -= action.count
            if p.hand[action.good] == 0:
                del p.hand[action.good]
            # Take tokens from pile
            pile = gs.tokens[action.good]
            earned = 0
            for _ in range(action.count):
                if pile:
                    earned += pile.pop(0)
            # Bonus tokens
            if action.count >= 5:
                if gs.bonus5:
                    earned += gs.bonus5.pop()
            elif action.count == 4:
                if gs.bonus4:
                    earned += gs.bonus4.pop()
            elif action.count == 3:
                if gs.bonus3:
                    earned += gs.bonus3.pop()
            p.score += earned

        gs._check_round_over()
        if not gs.round_over:
            gs.current_player = 1 - gs.current_player

        return gs

    def score_round(self) -> tuple[int, int]:
        """Return final scores for (player 0, player 1) including camel bonus."""
        s0 = self.players[0].score
        s1 = self.players[1].score
        # Camel majority bonus
        c0 = self.players[0].camels
        c1 = self.players[1].camels
        if c0 > c1:
            s0 += CAMEL_BONUS
        elif c1 > c0:
            s1 += CAMEL_BONUS
        return s0, s1

    def round_winner(self) -> int | None:
        """Return 0, 1, or None (tie)."""
        s0, s1 = self.score_round()
        if s0 > s1:
            return 0
        elif s1 > s0:
            return 1
        # Tiebreaker: most bonus tokens (simplified: most types of tokens sold)
        # Official rule: player with most bonus tokens wins; if still tied, most goods tokens
        # For simplicity, return None (tie → no one wins the round)
        return None

    def copy(self) -> GameState:
        gs = GameState()
        gs.market = list(self.market)
        gs.deck = list(self.deck)
        gs.players = (self.players[0].copy(), self.players[1].copy())
        gs.tokens = {g: list(v) for g, v in self.tokens.items()}
        gs.bonus3 = list(self.bonus3)
        gs.bonus4 = list(self.bonus4)
        gs.bonus5 = list(self.bonus5)
        gs.current_player = self.current_player
        gs.round_over = self.round_over
        return gs


def play_round(agents, rng: random.Random | None = None) -> int | None:
    """Play one round, return winner (0, 1, or None)."""
    gs = GameState.new_round(rng)
    while not gs.round_over:
        actions = gs.get_legal_actions()
        if not actions:
            break
        action = agents[gs.current_player].choose(gs, actions)
        gs = gs.apply_action(action)
    return gs.round_winner()


def play_match(agents, rng: random.Random | None = None) -> int:
    """Play best-of-3 match, return winner (0 or 1)."""
    if rng is None:
        rng = random.Random()
    wins = [0, 0]
    for _ in range(3):
        winner = play_round(agents, rng)
        if winner is not None:
            wins[winner] += 1
        if wins[0] == 2:
            return 0
        if wins[1] == 2:
            return 1
    # If no one got 2 wins (possible with ties), most wins
    if wins[0] > wins[1]:
        return 0
    elif wins[1] > wins[0]:
        return 1
    return 0  # arbitrary tiebreak

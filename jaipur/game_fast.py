"""Fast Jaipur game engine — optimized for self-play training.

Key optimization: exchange actions are simplified to avoid combinatorial explosion.
Instead of enumerating all possible take/give combinations, we generate a practical
subset that covers the strategically important moves.

If the Cython extension is available, all public names are re-exported from it.
Otherwise, falls back to the pure-Python implementation below.
"""
from __future__ import annotations

# ── Try Cython engine first ──────────────────────────────────────
try:
    from ._engine import (
        GameState,
        PlayerState,
        play_round,
        play_match,
        ACT_TAKE_ONE_PY as ACT_TAKE_ONE,
        ACT_TAKE_CAMELS_PY as ACT_TAKE_CAMELS,
        ACT_SELL_PY as ACT_SELL,
        ACT_EXCHANGE_PY as ACT_EXCHANGE,
        _N_GOODS_PY as _N_GOODS,
    )
    _CYTHON_ENGINE = True
except ImportError:
    _CYTHON_ENGINE = False

    import random
    from collections import Counter
    from dataclasses import dataclass, field
    from typing import Union

    from .cards import (
        BONUS_TOKENS_3, BONUS_TOKENS_4, BONUS_TOKENS_5,
        CAMEL_BONUS, GOODS, PRECIOUS_GOODS, TOKENS,
        GoodType, build_deck,
    )

    # ---------- Actions (use ints for speed) ----------

    # Good type indices for fast lookup
    _G2I = {g: i for i, g in enumerate(GOODS)}  # 6 goods
    _N_GOODS = len(GOODS)

    # Action types as simple tuples for speed (no dataclasses overhead)
    ACT_TAKE_ONE = 0
    ACT_TAKE_CAMELS = 1
    ACT_SELL = 2
    ACT_EXCHANGE = 3

    @dataclass
    class PlayerState:
        hand: list[int] = field(default_factory=lambda: [0] * _N_GOODS)
        camels: int = 0
        score: int = 0

        @property
        def hand_size(self) -> int:
            return sum(self.hand)

        def copy(self) -> PlayerState:
            p = PlayerState()
            p.hand = self.hand[:]
            p.camels = self.camels
            p.score = self.score
            return p

    @dataclass
    class GameState:
        market: list[int] = field(default_factory=lambda: [0] * (_N_GOODS + 1))
        deck: list[int] = field(default_factory=list)
        players: tuple[PlayerState, PlayerState] = field(
            default_factory=lambda: (PlayerState(), PlayerState())
        )
        tokens: list[list[int]] = field(default_factory=list)
        bonus3: list[int] = field(default_factory=list)
        bonus4: list[int] = field(default_factory=list)
        bonus5: list[int] = field(default_factory=list)
        current_player: int = 0
        round_over: bool = False
        deck_size: int = 0

        @staticmethod
        def new_round(rng: random.Random | None = None) -> GameState:
            if rng is None:
                rng = random.Random()

            gs = GameState()

            gs.tokens = [list(TOKENS[g]) for g in GOODS]

            gs.bonus3 = list(BONUS_TOKENS_3)
            gs.bonus4 = list(BONUS_TOKENS_4)
            gs.bonus5 = list(BONUS_TOKENS_5)
            rng.shuffle(gs.bonus3)
            rng.shuffle(gs.bonus4)
            rng.shuffle(gs.bonus5)

            deck: list[int] = []
            for g in GOODS:
                from .cards import CARD_COUNTS
                deck.extend([_G2I[g]] * CARD_COUNTS[g])
            from .cards import CARD_COUNTS
            deck.extend([6] * CARD_COUNTS[GoodType.CAMEL])
            rng.shuffle(deck)

            for _ in range(3):
                deck.remove(6)

            gs.market = [0] * (_N_GOODS + 1)
            gs.market[_N_GOODS] = 3
            for _ in range(2):
                card = deck.pop()
                gs.market[card] += 1

            for p in gs.players:
                for _ in range(5):
                    card = deck.pop()
                    if card == 6:
                        p.camels += 1
                    else:
                        p.hand[card] += 1

            gs.deck = deck
            gs.deck_size = len(deck)
            gs.current_player = 0
            return gs

        @property
        def cp(self) -> PlayerState:
            return self.players[self.current_player]

        @property
        def op(self) -> PlayerState:
            return self.players[1 - self.current_player]

        def _refill_market(self) -> None:
            total = sum(self.market)
            while total < 5 and self.deck:
                card = self.deck.pop()
                self.deck_size -= 1
                self.market[card] += 1
                total += 1

        def _check_round_over(self) -> None:
            empty = sum(1 for i in range(_N_GOODS) if not self.tokens[i])
            if empty >= 3 or self.deck_size == 0:
                self.round_over = True

        def get_legal_actions(self) -> list[tuple]:
            if self.round_over:
                return []

            actions: list[tuple] = []
            p = self.cp
            hs = p.hand_size

            if hs < 7:
                for i in range(_N_GOODS):
                    if self.market[i] > 0:
                        actions.append((ACT_TAKE_ONE, i))

            if self.market[_N_GOODS] > 0:
                actions.append((ACT_TAKE_CAMELS,))

            for i in range(_N_GOODS):
                count = p.hand[i]
                if count == 0:
                    continue
                if i < 3:
                    if count >= 3:
                        for n in range(3, count + 1):
                            actions.append((ACT_SELL, i, n))
                else:
                    for n in range(1, count + 1):
                        actions.append((ACT_SELL, i, n))

            self._add_exchange_actions(actions)

            return actions

        def _add_exchange_actions(self, actions: list[tuple]) -> None:
            p = self.cp
            hs = p.hand_size
            market_goods = [(i, self.market[i]) for i in range(_N_GOODS) if self.market[i] > 0]

            if len(market_goods) == 0:
                return

            giveable = [(i, p.hand[i]) for i in range(_N_GOODS) if p.hand[i] > 0]
            total_giveable = sum(c for _, c in giveable) + p.camels

            if total_giveable < 2:
                return

            for ti in range(len(market_goods)):
                gi, gc = market_goods[ti]
                for tj in range(ti, len(market_goods)):
                    gj, gj_c = market_goods[tj]
                    if ti == tj and gc < 2:
                        continue

                    take = (gi, gj) if ti != tj else (gi, gi)

                    take_count = 2
                    new_hand_max = hs + take_count
                    min_from_hand = max(0, hs + take_count - 7)

                    if p.camels >= 2 and min_from_hand == 0:
                        actions.append((ACT_EXCHANGE, take, (), 2))

                    if p.camels >= 1:
                        for gk, gk_c in giveable:
                            if gk not in take:
                                actions.append((ACT_EXCHANGE, take, (gk,), 1))
                                break

                    gave = []
                    remaining = 2
                    for gk, gk_c in giveable:
                        if gk in take:
                            continue
                        n = min(gk_c, remaining)
                        gave.extend([gk] * n)
                        remaining -= n
                        if remaining == 0:
                            break
                    if remaining == 0:
                        actions.append((ACT_EXCHANGE, take, tuple(gave), 0))

            if total_giveable >= 3 and len(market_goods) >= 2:
                all_market = []
                for i, c in market_goods:
                    all_market.extend([i] * min(c, 3))
                all_market.sort()

                if len(all_market) >= 3:
                    take = tuple(all_market[:3])
                    give_goods = []
                    give_camels = min(p.camels, 3)
                    remaining = 3 - give_camels
                    for gk, gk_c in giveable:
                        if gk in take:
                            continue
                        n = min(gk_c, remaining)
                        give_goods.extend([gk] * n)
                        remaining -= n
                        if remaining == 0:
                            break
                    if remaining == 0:
                        hs_after = hs + len(take) - len(give_goods)
                        if hs_after <= 7:
                            actions.append((ACT_EXCHANGE, take, tuple(give_goods), give_camels))

        def apply_action(self, action: tuple) -> GameState:
            gs = self.copy()
            p = gs.cp

            atype = action[0]

            if atype == ACT_TAKE_ONE:
                gi = action[1]
                gs.market[gi] -= 1
                p.hand[gi] += 1
                gs._refill_market()

            elif atype == ACT_TAKE_CAMELS:
                n = gs.market[_N_GOODS]
                gs.market[_N_GOODS] = 0
                p.camels += n
                gs._refill_market()

            elif atype == ACT_SELL:
                gi, count = action[1], action[2]
                p.hand[gi] -= count
                earned = 0
                pile = gs.tokens[gi]
                for _ in range(count):
                    if pile:
                        earned += pile.pop(0)
                if count >= 5:
                    if gs.bonus5:
                        earned += gs.bonus5.pop()
                elif count == 4:
                    if gs.bonus4:
                        earned += gs.bonus4.pop()
                elif count == 3:
                    if gs.bonus3:
                        earned += gs.bonus3.pop()
                p.score += earned

            elif atype == ACT_EXCHANGE:
                take_goods = action[1]
                give_goods = action[2]
                give_camels = action[3]

                for gi in take_goods:
                    gs.market[gi] -= 1
                    p.hand[gi] += 1

                for gi in give_goods:
                    p.hand[gi] -= 1
                    gs.market[gi] += 1
                p.camels -= give_camels
                gs.market[_N_GOODS] += give_camels

            gs._check_round_over()
            if not gs.round_over:
                gs.current_player = 1 - gs.current_player

            return gs

        def score_round(self) -> tuple[int, int]:
            s0 = self.players[0].score
            s1 = self.players[1].score
            c0 = self.players[0].camels
            c1 = self.players[1].camels
            if c0 > c1:
                s0 += CAMEL_BONUS
            elif c1 > c0:
                s1 += CAMEL_BONUS
            return s0, s1

        def round_winner(self) -> int | None:
            s0, s1 = self.score_round()
            if s0 > s1:
                return 0
            elif s1 > s0:
                return 1
            return None

        def copy(self) -> GameState:
            gs = GameState()
            gs.market = self.market[:]
            gs.deck = self.deck[:]
            gs.players = (self.players[0].copy(), self.players[1].copy())
            gs.tokens = [list(t) for t in self.tokens]
            gs.bonus3 = self.bonus3[:]
            gs.bonus4 = self.bonus4[:]
            gs.bonus5 = self.bonus5[:]
            gs.current_player = self.current_player
            gs.round_over = self.round_over
            gs.deck_size = self.deck_size
            return gs

    def play_round(agents, rng: random.Random | None = None) -> int | None:
        gs = GameState.new_round(rng)
        while not gs.round_over:
            actions = gs.get_legal_actions()
            if not actions:
                break
            action = agents[gs.current_player].choose(gs, actions)
            gs = gs.apply_action(action)
        return gs.round_winner()

    def play_match(agents, rng: random.Random | None = None) -> int:
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
        if wins[0] > wins[1]:
            return 0
        elif wins[1] > wins[0]:
            return 1
        return 0

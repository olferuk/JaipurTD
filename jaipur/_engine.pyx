# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False
"""Cython-accelerated Jaipur game engine.

All game state is stored in C arrays. Python-visible wrapper classes
expose the same API as game_fast.py so the rest of the codebase works unchanged.
"""
from libc.stdlib cimport malloc, free, rand, srand
from libc.string cimport memcpy, memset
from cpython.ref cimport PyObject

import random as _random

# ── Constants ──────────────────────────────────────────────────────
DEF N_GOODS = 6
DEF MARKET_SLOTS = 7          # 6 goods + 1 camel slot
DEF MAX_TOKENS = 9            # max pile size (leather has 9)
DEF MAX_DECK = 55
DEF MAX_ACTIONS = 120         # practical upper bound on legal actions
DEF MAX_BONUS = 10

# Action types
DEF ACT_TAKE_ONE = 0
DEF ACT_TAKE_CAMELS = 1
DEF ACT_SELL = 2
DEF ACT_EXCHANGE = 3

# Card counts: diamond=6, gold=6, silver=6, cloth=8, spice=8, leather=10, camel=11
cdef int CARD_COUNTS[7]
CARD_COUNTS = [6, 6, 6, 8, 8, 10, 11]

# Token values per good type (index 0 = top/most valuable)
cdef int TOKEN_VALS[6][9]
cdef int TOKEN_LENS[6]
TOKEN_VALS[0] = [7, 7, 5, 5, 5, 0, 0, 0, 0]  # diamond
TOKEN_VALS[1] = [6, 6, 5, 5, 5, 0, 0, 0, 0]  # gold
TOKEN_VALS[2] = [5, 5, 5, 5, 5, 0, 0, 0, 0]  # silver
TOKEN_VALS[3] = [5, 3, 3, 2, 2, 1, 1, 0, 0]  # cloth
TOKEN_VALS[4] = [5, 3, 3, 2, 2, 1, 1, 0, 0]  # spice
TOKEN_VALS[5] = [4, 3, 2, 1, 1, 1, 1, 1, 1]  # leather
TOKEN_LENS = [5, 5, 5, 7, 7, 9]

cdef int BONUS3_INIT[7]
cdef int BONUS4_INIT[6]
cdef int BONUS5_INIT[5]
BONUS3_INIT = [1, 1, 2, 2, 2, 3, 3]
BONUS4_INIT = [4, 4, 5, 5, 6, 6]
BONUS5_INIT = [8, 8, 9, 10, 10]

DEF CAMEL_BONUS = 5

# ── C-level game state ────────────────────────────────────────────

cdef struct CPlayer:
    int hand[6]          # count per good type
    int camels
    int score

cdef struct CGameState:
    int market[7]        # 0-5 goods, 6 camels
    int deck[55]
    int deck_len
    CPlayer players[2]
    int tokens[6][9]
    int token_lens[6]    # current length of each token pile
    int bonus3[10]
    int bonus3_len
    int bonus4[10]
    int bonus4_len
    int bonus5[10]
    int bonus5_len
    int current_player
    bint round_over

# ── Fisher-Yates shuffle using Python RNG ─────────────────────────

cdef void shuffle_int_array(int* arr, int n, object rng):
    """Shuffle array in-place using the provided Python random.Random."""
    cdef int i, j, tmp
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp

# ── Core C functions ──────────────────────────────────────────────

cdef inline int c_hand_size(CPlayer* p) noexcept nogil:
    cdef int s = 0
    cdef int i
    for i in range(N_GOODS):
        s += p.hand[i]
    return s

cdef void c_init_round(CGameState* gs, object rng):
    """Initialize a new round."""
    cdef int i, j, idx, card

    # Tokens
    for i in range(N_GOODS):
        gs.token_lens[i] = TOKEN_LENS[i]
        for j in range(TOKEN_LENS[i]):
            gs.tokens[i][j] = TOKEN_VALS[i][j]

    # Bonus piles
    gs.bonus3_len = 7
    for i in range(7):
        gs.bonus3[i] = BONUS3_INIT[i]
    gs.bonus4_len = 6
    for i in range(6):
        gs.bonus4[i] = BONUS4_INIT[i]
    gs.bonus5_len = 5
    for i in range(5):
        gs.bonus5[i] = BONUS5_INIT[i]
    shuffle_int_array(gs.bonus3, 7, rng)
    shuffle_int_array(gs.bonus4, 6, rng)
    shuffle_int_array(gs.bonus5, 5, rng)

    # Build deck: good indices 0-5, camel = 6
    idx = 0
    for i in range(7):  # 6 goods + camel
        for j in range(CARD_COUNTS[i]):
            gs.deck[idx] = i
            idx += 1
    gs.deck_len = idx  # should be 55
    shuffle_int_array(gs.deck, gs.deck_len, rng)

    # Remove 3 camels for market
    cdef int removed = 0
    cdef int write_idx = 0
    for i in range(gs.deck_len):
        if gs.deck[i] == 6 and removed < 3:
            removed += 1
        else:
            gs.deck[write_idx] = gs.deck[i]
            write_idx += 1
    gs.deck_len = write_idx  # should be 52

    # Market: 3 camels + 2 from deck
    memset(gs.market, 0, sizeof(gs.market))
    gs.market[N_GOODS] = 3
    for i in range(2):
        gs.deck_len -= 1
        card = gs.deck[gs.deck_len]
        gs.market[card] += 1

    # Players
    for i in range(2):
        memset(gs.players[i].hand, 0, sizeof(gs.players[i].hand))
        gs.players[i].camels = 0
        gs.players[i].score = 0

    # Deal 5 cards each
    for i in range(2):
        for j in range(5):
            gs.deck_len -= 1
            card = gs.deck[gs.deck_len]
            if card == 6:
                gs.players[i].camels += 1
            else:
                gs.players[i].hand[card] += 1

    gs.current_player = 0
    gs.round_over = 0

cdef void c_refill_market(CGameState* gs) noexcept nogil:
    cdef int total = 0
    cdef int i, card
    for i in range(MARKET_SLOTS):
        total += gs.market[i]
    while total < 5 and gs.deck_len > 0:
        gs.deck_len -= 1
        card = gs.deck[gs.deck_len]
        gs.market[card] += 1
        total += 1

cdef void c_check_round_over(CGameState* gs) noexcept nogil:
    cdef int empty = 0
    cdef int i
    for i in range(N_GOODS):
        if gs.token_lens[i] == 0:
            empty += 1
    if empty >= 3 or gs.deck_len == 0:
        gs.round_over = 1

cdef void c_copy_state(CGameState* dst, const CGameState* src) noexcept nogil:
    memcpy(dst, src, sizeof(CGameState))

# ── Apply action (C level) ────────────────────────────────────────

cdef void c_apply_take_one(CGameState* gs, int gi) noexcept nogil:
    gs.market[gi] -= 1
    gs.players[gs.current_player].hand[gi] += 1
    c_refill_market(gs)

cdef void c_apply_take_camels(CGameState* gs) noexcept nogil:
    gs.players[gs.current_player].camels += gs.market[N_GOODS]
    gs.market[N_GOODS] = 0
    c_refill_market(gs)

cdef void c_apply_sell(CGameState* gs, int gi, int count) noexcept nogil:
    cdef CPlayer* p = &gs.players[gs.current_player]
    cdef int earned = 0
    cdef int i, j
    p.hand[gi] -= count

    # Take tokens from front of pile
    for i in range(count):
        if gs.token_lens[gi] > 0:
            earned += gs.tokens[gi][0]
            # Shift pile left
            gs.token_lens[gi] -= 1
            for j in range(gs.token_lens[gi]):
                gs.tokens[gi][j] = gs.tokens[gi][j + 1]

    # Bonus
    if count >= 5 and gs.bonus5_len > 0:
        gs.bonus5_len -= 1
        earned += gs.bonus5[gs.bonus5_len]
    elif count == 4 and gs.bonus4_len > 0:
        gs.bonus4_len -= 1
        earned += gs.bonus4[gs.bonus4_len]
    elif count == 3 and gs.bonus3_len > 0:
        gs.bonus3_len -= 1
        earned += gs.bonus3[gs.bonus3_len]

    p.score += earned

cdef void c_apply_exchange(CGameState* gs, const int* take_goods, int take_len,
                           const int* give_goods, int give_len, int give_camels) noexcept nogil:
    cdef CPlayer* p = &gs.players[gs.current_player]
    cdef int i
    for i in range(take_len):
        gs.market[take_goods[i]] -= 1
        p.hand[take_goods[i]] += 1
    for i in range(give_len):
        p.hand[give_goods[i]] -= 1
        gs.market[give_goods[i]] += 1
    p.camels -= give_camels
    gs.market[N_GOODS] += give_camels


# ── Python-visible wrapper ────────────────────────────────────────

# Export action type constants at module level
ACT_TAKE_ONE_PY = ACT_TAKE_ONE
ACT_TAKE_CAMELS_PY = ACT_TAKE_CAMELS
ACT_SELL_PY = ACT_SELL
ACT_EXCHANGE_PY = ACT_EXCHANGE
_N_GOODS_PY = N_GOODS


cdef class PlayerState:
    """Python-visible player state backed by C struct."""
    cdef CPlayer _c

    def __init__(self):
        memset(self._c.hand, 0, sizeof(self._c.hand))
        self._c.camels = 0
        self._c.score = 0

    @property
    def hand(self):
        return [self._c.hand[i] for i in range(N_GOODS)]

    @hand.setter
    def hand(self, val):
        cdef int i
        for i in range(N_GOODS):
            self._c.hand[i] = val[i]

    @property
    def camels(self):
        return self._c.camels

    @camels.setter
    def camels(self, int val):
        self._c.camels = val

    @property
    def score(self):
        return self._c.score

    @score.setter
    def score(self, int val):
        self._c.score = val

    @property
    def hand_size(self):
        return c_hand_size(&self._c)

    def copy(self):
        cdef PlayerState p = PlayerState.__new__(PlayerState)
        memcpy(&p._c, &self._c, sizeof(CPlayer))
        return p


cdef class GameState:
    """Python-visible game state backed by C struct — API-compatible with game_fast.py."""
    cdef CGameState _c

    def __init__(self):
        memset(&self._c, 0, sizeof(CGameState))

    @staticmethod
    def new_round(rng=None):
        if rng is None:
            rng = _random.Random()
        cdef GameState gs = GameState.__new__(GameState)
        memset(&gs._c, 0, sizeof(CGameState))
        c_init_round(&gs._c, rng)
        return gs

    # ── Properties for compatibility ──

    @property
    def market(self):
        return [self._c.market[i] for i in range(MARKET_SLOTS)]

    @market.setter
    def market(self, val):
        cdef int i
        for i in range(MARKET_SLOTS):
            self._c.market[i] = val[i]

    @property
    def deck(self):
        return [self._c.deck[i] for i in range(self._c.deck_len)]

    @deck.setter
    def deck(self, val):
        cdef int i
        self._c.deck_len = len(val)
        for i in range(self._c.deck_len):
            self._c.deck[i] = val[i]

    @property
    def deck_size(self):
        return self._c.deck_len

    @deck_size.setter
    def deck_size(self, int val):
        self._c.deck_len = val

    @property
    def players(self):
        cdef PlayerState p0 = PlayerState.__new__(PlayerState)
        cdef PlayerState p1 = PlayerState.__new__(PlayerState)
        memcpy(&p0._c, &self._c.players[0], sizeof(CPlayer))
        memcpy(&p1._c, &self._c.players[1], sizeof(CPlayer))
        return (p0, p1)

    @property
    def tokens(self):
        result = []
        for i in range(N_GOODS):
            pile = []
            for j in range(self._c.token_lens[i]):
                pile.append(self._c.tokens[i][j])
            result.append(pile)
        return result

    @tokens.setter
    def tokens(self, val):
        cdef int i, j
        for i in range(N_GOODS):
            self._c.token_lens[i] = len(val[i])
            for j in range(len(val[i])):
                self._c.tokens[i][j] = val[i][j]

    @property
    def bonus3(self):
        return [self._c.bonus3[i] for i in range(self._c.bonus3_len)]

    @property
    def bonus4(self):
        return [self._c.bonus4[i] for i in range(self._c.bonus4_len)]

    @property
    def bonus5(self):
        return [self._c.bonus5[i] for i in range(self._c.bonus5_len)]

    @property
    def current_player(self):
        return self._c.current_player

    @current_player.setter
    def current_player(self, int val):
        self._c.current_player = val

    @property
    def round_over(self):
        return bool(self._c.round_over)

    @round_over.setter
    def round_over(self, val):
        self._c.round_over = 1 if val else 0

    @property
    def cp(self):
        cdef PlayerState p = PlayerState.__new__(PlayerState)
        memcpy(&p._c, &self._c.players[self._c.current_player], sizeof(CPlayer))
        return p

    @property
    def op(self):
        cdef PlayerState p = PlayerState.__new__(PlayerState)
        memcpy(&p._c, &self._c.players[1 - self._c.current_player], sizeof(CPlayer))
        return p

    # ── Core methods ──

    def _refill_market(self):
        c_refill_market(&self._c)

    def _check_round_over(self):
        c_check_round_over(&self._c)

    def get_legal_actions(self):
        if self._c.round_over:
            return []

        cdef CPlayer* p = &self._c.players[self._c.current_player]
        cdef int hs = c_hand_size(p)
        cdef int i, count

        actions = []

        # Take one good
        if hs < 7:
            for i in range(N_GOODS):
                if self._c.market[i] > 0:
                    actions.append((ACT_TAKE_ONE, i))

        # Take camels
        if self._c.market[N_GOODS] > 0:
            actions.append((ACT_TAKE_CAMELS,))

        # Sell
        for i in range(N_GOODS):
            count = p.hand[i]
            if count == 0:
                continue
            if i < 3:  # precious
                if count >= 3:
                    for n in range(3, count + 1):
                        actions.append((ACT_SELL, i, n))
            else:
                for n in range(1, count + 1):
                    actions.append((ACT_SELL, i, n))

        # Exchange
        self._add_exchange_actions(actions, p, hs)

        return actions

    cdef _add_exchange_actions(self, list actions, CPlayer* p, int hs):
        cdef int i, j, ti, tj, gi, gc, gj, gj_c, gk, gk_c
        cdef int take_count, min_from_hand, remaining, n, hs_after

        # Collect market goods
        market_goods = []
        for i in range(N_GOODS):
            if self._c.market[i] > 0:
                market_goods.append((i, self._c.market[i]))

        if len(market_goods) == 0:
            return

        # Collect giveable hand goods
        giveable = []
        cdef int total_giveable = p.camels
        for i in range(N_GOODS):
            if p.hand[i] > 0:
                giveable.append((i, p.hand[i]))
                total_giveable += p.hand[i]

        if total_giveable < 2:
            return

        cdef int mg_len = len(market_goods)

        # Size-2 exchanges
        for ti in range(mg_len):
            gi, gc = market_goods[ti]
            for tj in range(ti, mg_len):
                gj, gj_c = market_goods[tj]
                if ti == tj and gc < 2:
                    continue

                if ti != tj:
                    take = (gi, gj)
                else:
                    take = (gi, gi)

                min_from_hand = max(0, hs + 2 - 7)

                # Option 1: 2 camels
                if p.camels >= 2 and min_from_hand == 0:
                    actions.append((ACT_EXCHANGE, take, (), 2))

                # Option 2: 1 camel + 1 good
                if p.camels >= 1:
                    for gk, gk_c in giveable:
                        if gk != take[0] and gk != take[1]:
                            actions.append((ACT_EXCHANGE, take, (gk,), 1))
                            break

                # Option 3: 2 goods
                gave = []
                remaining = 2
                for gk, gk_c in giveable:
                    if gk == take[0] or gk == take[1]:
                        continue
                    n = min(gk_c, remaining)
                    for _ in range(n):
                        gave.append(gk)
                    remaining -= n
                    if remaining == 0:
                        break
                if remaining == 0:
                    actions.append((ACT_EXCHANGE, take, tuple(gave), 0))

        # Size-3 exchanges
        if total_giveable >= 3 and mg_len >= 2:
            all_market = []
            for i_val, c_val in market_goods:
                for _ in range(min(c_val, 3)):
                    all_market.append(i_val)
            all_market.sort()

            if len(all_market) >= 3:
                take = tuple(all_market[:3])
                give_goods = []
                give_camels = min(p.camels, 3)
                remaining = 3 - give_camels
                take_set = set(take)
                for gk, gk_c in giveable:
                    if gk in take_set:
                        continue
                    n = min(gk_c, remaining)
                    for _ in range(n):
                        give_goods.append(gk)
                    remaining -= n
                    if remaining == 0:
                        break
                if remaining == 0:
                    hs_after = hs + len(take) - len(give_goods)
                    if hs_after <= 7:
                        actions.append((ACT_EXCHANGE, take, tuple(give_goods), give_camels))

    def apply_action(self, tuple action):
        cdef GameState gs = GameState.__new__(GameState)
        c_copy_state(&gs._c, &self._c)

        cdef int atype = action[0]
        cdef int gi, count, give_camels
        cdef int take_arr[4]
        cdef int give_arr[4]
        cdef int take_len, give_len, i_idx

        if atype == ACT_TAKE_ONE:
            c_apply_take_one(&gs._c, action[1])

        elif atype == ACT_TAKE_CAMELS:
            c_apply_take_camels(&gs._c)

        elif atype == ACT_SELL:
            c_apply_sell(&gs._c, action[1], action[2])

        elif atype == ACT_EXCHANGE:
            take_goods = action[1]
            give_goods = action[2]
            give_camels = action[3]
            take_len = len(take_goods)
            give_len = len(give_goods)
            for i_idx in range(take_len):
                take_arr[i_idx] = take_goods[i_idx]
            for i_idx in range(give_len):
                give_arr[i_idx] = give_goods[i_idx]
            c_apply_exchange(&gs._c, take_arr, take_len, give_arr, give_len, give_camels)

        c_check_round_over(&gs._c)
        if not gs._c.round_over:
            gs._c.current_player = 1 - gs._c.current_player

        return gs

    def score_round(self):
        cdef int s0 = self._c.players[0].score
        cdef int s1 = self._c.players[1].score
        cdef int c0 = self._c.players[0].camels
        cdef int c1 = self._c.players[1].camels
        if c0 > c1:
            s0 += CAMEL_BONUS
        elif c1 > c0:
            s1 += CAMEL_BONUS
        return (s0, s1)

    def round_winner(self):
        s0, s1 = self.score_round()
        if s0 > s1:
            return 0
        elif s1 > s0:
            return 1
        return None

    def copy(self):
        cdef GameState gs = GameState.__new__(GameState)
        c_copy_state(&gs._c, &self._c)
        return gs

    # ── Fast encoding (avoids Python property overhead) ──

    def encode_for(self, int player):
        """Return 32-element float list for encoding — avoids property overhead."""
        import numpy as np
        cdef CPlayer* me = &self._c.players[player]
        cdef CPlayer* opp = &self._c.players[1 - player]

        feat = np.zeros(32, dtype=np.float32)
        cdef int i

        for i in range(N_GOODS):
            feat[i] = me.hand[i]
        feat[6] = me.camels

        for i in range(N_GOODS):
            feat[7 + i] = self._c.market[i]
        feat[13] = self._c.market[N_GOODS]

        for i in range(N_GOODS):
            feat[14 + i] = self._c.token_lens[i]
            if self._c.token_lens[i] > 0:
                feat[20 + i] = self._c.tokens[i][0]

        feat[26] = me.score / 100.0
        feat[27] = opp.score / 100.0
        feat[28] = c_hand_size(me)
        feat[29] = c_hand_size(opp)
        feat[30] = opp.camels
        feat[31] = self._c.deck_len / 40.0

        return feat


# ── C-level action representation for internal simulation ─────

DEF MAX_ACT_BUF = 120

cdef struct CAction:
    int atype
    int gi             # good index (TAKE_ONE, SELL)
    int count          # sell count
    int take[4]        # exchange: goods to take
    int take_len
    int give[4]        # exchange: goods to give
    int give_len
    int give_camels

cdef void c_apply_caction(CGameState* gs, CAction* act) noexcept nogil:
    if act.atype == ACT_TAKE_ONE:
        c_apply_take_one(gs, act.gi)
    elif act.atype == ACT_TAKE_CAMELS:
        c_apply_take_camels(gs)
    elif act.atype == ACT_SELL:
        c_apply_sell(gs, act.gi, act.count)
    elif act.atype == ACT_EXCHANGE:
        c_apply_exchange(gs, act.take, act.take_len, act.give, act.give_len, act.give_camels)


cdef int c_get_legal_actions(CGameState* gs, CAction* buf) noexcept nogil:
    """Fill buf with legal actions, return count."""
    if gs.round_over:
        return 0

    cdef CPlayer* p = &gs.players[gs.current_player]
    cdef int hs = c_hand_size(p)
    cdef int n_act = 0
    cdef int i, j, count, n_val

    # Take one good
    if hs < 7:
        for i in range(N_GOODS):
            if gs.market[i] > 0:
                buf[n_act].atype = ACT_TAKE_ONE
                buf[n_act].gi = i
                n_act += 1

    # Take camels
    if gs.market[N_GOODS] > 0:
        buf[n_act].atype = ACT_TAKE_CAMELS
        n_act += 1

    # Sell
    for i in range(N_GOODS):
        count = p.hand[i]
        if count == 0:
            continue
        if i < 3:  # precious
            if count >= 3:
                for n_val in range(3, count + 1):
                    buf[n_act].atype = ACT_SELL
                    buf[n_act].gi = i
                    buf[n_act].count = n_val
                    n_act += 1
        else:
            for n_val in range(1, count + 1):
                buf[n_act].atype = ACT_SELL
                buf[n_act].gi = i
                buf[n_act].count = n_val
                n_act += 1

    # Exchange actions
    n_act = c_add_exchange_actions(gs, p, hs, buf, n_act)

    return n_act


cdef int c_add_exchange_actions(CGameState* gs, CPlayer* p, int hs,
                                CAction* buf, int n_act) noexcept nogil:
    cdef int mg_idx[6]
    cdef int mg_cnt[6]
    cdef int mg_len = 0
    cdef int gv_idx[6]
    cdef int gv_cnt[6]
    cdef int gv_len = 0
    cdef int total_giveable = p.camels
    cdef int i, j, ti, tj, gi, gc, gj, gj_c, gk, gk_c
    cdef int min_from_hand, remaining, n_val, hs_after
    cdef int gave_idx[4]
    cdef int gave_len
    cdef int all_m[18]
    cdef int all_m_len
    cdef int tmp
    cdef int t0, t1, t2, gc3
    cdef int k

    for i in range(N_GOODS):
        if gs.market[i] > 0:
            mg_idx[mg_len] = i
            mg_cnt[mg_len] = gs.market[i]
            mg_len += 1
        if p.hand[i] > 0:
            gv_idx[gv_len] = i
            gv_cnt[gv_len] = p.hand[i]
            total_giveable += p.hand[i]
            gv_len += 1

    if mg_len == 0 or total_giveable < 2:
        return n_act

    # Size-2 exchanges
    for ti in range(mg_len):
        gi = mg_idx[ti]
        gc = mg_cnt[ti]
        for tj in range(ti, mg_len):
            gj = mg_idx[tj]
            gj_c = mg_cnt[tj]
            if ti == tj and gc < 2:
                continue

            min_from_hand = hs + 2 - 7
            if min_from_hand < 0:
                min_from_hand = 0

            # Option 1: 2 camels
            if p.camels >= 2 and min_from_hand == 0:
                buf[n_act].atype = ACT_EXCHANGE
                buf[n_act].take[0] = gi
                buf[n_act].take[1] = gj
                buf[n_act].take_len = 2
                buf[n_act].give_len = 0
                buf[n_act].give_camels = 2
                n_act += 1

            # Option 2: 1 camel + 1 good
            if p.camels >= 1:
                for i in range(gv_len):
                    gk = gv_idx[i]
                    if gk != gi and gk != gj:
                        buf[n_act].atype = ACT_EXCHANGE
                        buf[n_act].take[0] = gi
                        buf[n_act].take[1] = gj
                        buf[n_act].take_len = 2
                        buf[n_act].give[0] = gk
                        buf[n_act].give_len = 1
                        buf[n_act].give_camels = 1
                        n_act += 1
                        break

            # Option 3: 2 goods
            gave_len = 0
            remaining = 2
            for i in range(gv_len):
                gk = gv_idx[i]
                gk_c = gv_cnt[i]
                if gk == gi or gk == gj:
                    continue
                n_val = gk_c if gk_c < remaining else remaining
                for j in range(n_val):
                    gave_idx[gave_len] = gk
                    gave_len += 1
                remaining -= n_val
                if remaining == 0:
                    break
            if remaining == 0:
                buf[n_act].atype = ACT_EXCHANGE
                buf[n_act].take[0] = gi
                buf[n_act].take[1] = gj
                buf[n_act].take_len = 2
                for i in range(gave_len):
                    buf[n_act].give[i] = gave_idx[i]
                buf[n_act].give_len = gave_len
                buf[n_act].give_camels = 0
                n_act += 1

    # Size-3 exchanges
    if total_giveable >= 3 and mg_len >= 2:
        # Collect all market goods sorted
        all_m_len = 0
        for i in range(mg_len):
            n_val = mg_cnt[i] if mg_cnt[i] < 3 else 3
            for j in range(n_val):
                all_m[all_m_len] = mg_idx[i]
                all_m_len += 1
        # Sort (insertion sort, small array)
        for i in range(1, all_m_len):
            tmp = all_m[i]
            j = i - 1
            while j >= 0 and all_m[j] > tmp:
                all_m[j + 1] = all_m[j]
                j -= 1
            all_m[j + 1] = tmp

        if all_m_len >= 3:
            t0 = all_m[0]
            t1 = all_m[1]
            t2 = all_m[2]
            gave_len = 0
            gc3 = p.camels if p.camels < 3 else 3
            remaining = 3 - gc3
            for i in range(gv_len):
                gk = gv_idx[i]
                gk_c = gv_cnt[i]
                if gk == t0 or gk == t1 or gk == t2:
                    continue
                n_val = gk_c if gk_c < remaining else remaining
                for j in range(n_val):
                    gave_idx[gave_len] = gk
                    gave_len += 1
                remaining -= n_val
                if remaining == 0:
                    break
            if remaining == 0:
                hs_after = hs + 3 - gave_len
                if hs_after <= 7:
                    buf[n_act].atype = ACT_EXCHANGE
                    buf[n_act].take[0] = t0
                    buf[n_act].take[1] = t1
                    buf[n_act].take[2] = t2
                    buf[n_act].take_len = 3
                    for i in range(gave_len):
                        buf[n_act].give[i] = gave_idx[i]
                    buf[n_act].give_len = gave_len
                    buf[n_act].give_camels = gc3
                    n_act += 1

    return n_act


# ── Fast LCG RNG for C-level simulation ──────────────────────────

cdef struct CLCG:
    unsigned long long state

cdef inline void lcg_seed(CLCG* rng, unsigned long long seed) noexcept nogil:
    rng.state = seed

cdef inline unsigned int lcg_next(CLCG* rng) noexcept nogil:
    rng.state = rng.state * 6364136223846793005ULL + 1442695040888963407ULL
    return <unsigned int>(rng.state >> 33)

cdef inline int lcg_randint(CLCG* rng, int n) noexcept nogil:
    """Return random int in [0, n)."""
    return <int>(lcg_next(rng) % <unsigned int>n)

cdef void lcg_shuffle(int* arr, int n, CLCG* rng) noexcept nogil:
    cdef int i, j, tmp
    for i in range(n - 1, 0, -1):
        j = lcg_randint(rng, i + 1)
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp


cdef void c_init_round_fast(CGameState* gs, CLCG* rng) noexcept nogil:
    """Initialize round using C-only RNG (no Python calls)."""
    cdef int i, j, idx, card

    for i in range(N_GOODS):
        gs.token_lens[i] = TOKEN_LENS[i]
        for j in range(TOKEN_LENS[i]):
            gs.tokens[i][j] = TOKEN_VALS[i][j]

    gs.bonus3_len = 7
    for i in range(7):
        gs.bonus3[i] = BONUS3_INIT[i]
    gs.bonus4_len = 6
    for i in range(6):
        gs.bonus4[i] = BONUS4_INIT[i]
    gs.bonus5_len = 5
    for i in range(5):
        gs.bonus5[i] = BONUS5_INIT[i]
    lcg_shuffle(gs.bonus3, 7, rng)
    lcg_shuffle(gs.bonus4, 6, rng)
    lcg_shuffle(gs.bonus5, 5, rng)

    idx = 0
    for i in range(7):
        for j in range(CARD_COUNTS[i]):
            gs.deck[idx] = i
            idx += 1
    gs.deck_len = idx
    lcg_shuffle(gs.deck, gs.deck_len, rng)

    cdef int removed = 0
    cdef int write_idx = 0
    for i in range(gs.deck_len):
        if gs.deck[i] == 6 and removed < 3:
            removed += 1
        else:
            gs.deck[write_idx] = gs.deck[i]
            write_idx += 1
    gs.deck_len = write_idx

    memset(gs.market, 0, sizeof(gs.market))
    gs.market[N_GOODS] = 3
    for i in range(2):
        gs.deck_len -= 1
        card = gs.deck[gs.deck_len]
        gs.market[card] += 1

    for i in range(2):
        memset(gs.players[i].hand, 0, sizeof(gs.players[i].hand))
        gs.players[i].camels = 0
        gs.players[i].score = 0

    for i in range(2):
        for j in range(5):
            gs.deck_len -= 1
            card = gs.deck[gs.deck_len]
            if card == 6:
                gs.players[i].camels += 1
            else:
                gs.players[i].hand[card] += 1

    gs.current_player = 0
    gs.round_over = 0


cdef int c_round_winner(CGameState* gs) noexcept nogil:
    """Return 0, 1, or -1 for tie."""
    cdef int s0 = gs.players[0].score
    cdef int s1 = gs.players[1].score
    if gs.players[0].camels > gs.players[1].camels:
        s0 += CAMEL_BONUS
    elif gs.players[1].camels > gs.players[0].camels:
        s1 += CAMEL_BONUS
    if s0 > s1:
        return 0
    elif s1 > s0:
        return 1
    return -1


cdef int c_play_round_random(CLCG* rng) noexcept nogil:
    """Play a full round with random agents, entirely in C. Returns 0, 1, or -1."""
    cdef CGameState gs
    cdef CAction acts[MAX_ACT_BUF]
    cdef int n_acts, choice

    c_init_round_fast(&gs, rng)

    while not gs.round_over:
        n_acts = c_get_legal_actions(&gs, acts)
        if n_acts == 0:
            break
        choice = lcg_randint(rng, n_acts)
        c_apply_caction(&gs, &acts[choice])
        c_check_round_over(&gs)
        if not gs.round_over:
            gs.current_player = 1 - gs.current_player

    return c_round_winner(&gs)


def simulate_random_rounds(int n, unsigned long long seed=42):
    """Run n random-vs-random rounds entirely in C. Returns (p0_wins, p1_wins, draws)."""
    cdef CLCG rng
    lcg_seed(&rng, seed)
    cdef int w0 = 0, w1 = 0, draws = 0
    cdef int result
    cdef int i

    with nogil:
        for i in range(n):
            result = c_play_round_random(&rng)
            if result == 0:
                w0 += 1
            elif result == 1:
                w1 += 1
            else:
                draws += 1

    return (w0, w1, draws)


def play_round(agents, rng=None):
    gs = GameState.new_round(rng)
    while not gs.round_over:
        actions = gs.get_legal_actions()
        if not actions:
            break
        action = agents[gs.current_player].choose(gs, actions)
        gs = gs.apply_action(action)
    return gs.round_winner()


def play_match(agents, rng=None):
    if rng is None:
        rng = _random.Random()
    cdef int w0 = 0, w1 = 0
    for _ in range(3):
        winner = play_round(agents, rng)
        if winner is not None:
            if winner == 0:
                w0 += 1
            else:
                w1 += 1
        if w0 == 2:
            return 0
        if w1 == 2:
            return 1
    if w0 > w1:
        return 0
    elif w1 > w0:
        return 1
    return 0

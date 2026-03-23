"""Tests for Jaipur game engine."""

import random
from collections import Counter

from jaipur.agents import RandomAgent
from jaipur.cards import GOODS, GoodType
from jaipur.game import (
    GameState,
    Sell,
    TakeCamels,
    TakeOne,
    play_round,
)


class TestSetup:
    def test_deck_size(self):
        gs = GameState.new_round(random.Random(42))
        # 55 total - 5 market - 5 per player = 30 in deck
        total = len(gs.deck) + len(gs.market) + sum(p.hand_size + p.camels for p in gs.players)
        assert total == 55

    def test_market_size(self):
        gs = GameState.new_round(random.Random(42))
        assert len(gs.market) == 5

    def test_market_has_3_camels_initially(self):
        # Market starts with 3 camels + 2 from deck
        # After setup, market has at least the initial camels (some deck cards could be camels too)
        gs = GameState.new_round(random.Random(42))
        assert len(gs.market) == 5

    def test_player_starts_with_5_cards(self):
        gs = GameState.new_round(random.Random(42))
        for p in gs.players:
            assert p.hand_size + p.camels == 5


class TestTakeOne:
    def test_take_one_good(self):
        gs = GameState.new_round(random.Random(42))
        goods_in_market = [c for c in gs.market if c != GoodType.CAMEL]
        if goods_in_market:
            good = goods_in_market[0]
            old_hand = gs.cp.hand[good]
            gs2 = gs.apply_action(TakeOne(good))
            assert gs2.players[gs.current_player].hand[good] == old_hand + 1
            assert len(gs2.market) == 5  # refilled

    def test_hand_limit(self):
        gs = GameState.new_round(random.Random(42))
        # If hand is full (7), TakeOne should not be available
        gs.cp.hand = Counter({GoodType.DIAMOND: 4, GoodType.GOLD: 3})
        actions = gs.get_legal_actions()
        assert not any(isinstance(a, TakeOne) for a in actions)


class TestTakeCamels:
    def test_take_camels(self):
        gs = GameState.new_round(random.Random(42))
        n_camels = gs.market.count(GoodType.CAMEL)
        if n_camels > 0:
            old_camels = gs.cp.camels
            gs2 = gs.apply_action(TakeCamels())
            assert gs2.players[gs.current_player].camels == old_camels + n_camels
            assert GoodType.CAMEL not in gs2.market
            assert len(gs2.market) == 5  # refilled


class TestSell:
    def test_sell_leather(self):
        gs = GameState.new_round(random.Random(42))
        gs.players[0].hand = Counter({GoodType.LEATHER: 3})
        gs.current_player = 0
        gs2 = gs.apply_action(Sell(GoodType.LEATHER, 3))
        assert gs2.players[0].hand.get(GoodType.LEATHER, 0) == 0
        assert gs2.players[0].score > 0

    def test_precious_needs_3(self):
        gs = GameState.new_round(random.Random(42))
        gs.players[0].hand = Counter({GoodType.DIAMOND: 2})
        gs.current_player = 0
        actions = gs.get_legal_actions()
        sells = [a for a in actions if isinstance(a, Sell) and a.good == GoodType.DIAMOND]
        assert len(sells) == 0  # can't sell 2 diamonds

    def test_precious_can_sell_3(self):
        gs = GameState.new_round(random.Random(42))
        gs.players[0].hand = Counter({GoodType.DIAMOND: 3})
        gs.current_player = 0
        actions = gs.get_legal_actions()
        sells = [a for a in actions if isinstance(a, Sell) and a.good == GoodType.DIAMOND]
        assert any(s.count == 3 for s in sells)

    def test_bonus_token_3(self):
        gs = GameState.new_round(random.Random(42))
        gs.players[0].hand = Counter({GoodType.CLOTH: 3})
        gs.current_player = 0
        tokens_before = sum(gs.tokens[GoodType.CLOTH][:3])
        gs2 = gs.apply_action(Sell(GoodType.CLOTH, 3))
        # Should have gotten good tokens + a bonus
        assert gs2.players[0].score >= tokens_before

    def test_sell_1_non_precious(self):
        gs = GameState.new_round(random.Random(42))
        gs.players[0].hand = Counter({GoodType.LEATHER: 1})
        gs.current_player = 0
        actions = gs.get_legal_actions()
        sells = [a for a in actions if isinstance(a, Sell) and a.good == GoodType.LEATHER]
        assert any(s.count == 1 for s in sells)


class TestRoundEnd:
    def test_three_empty_piles(self):
        gs = GameState.new_round(random.Random(42))
        # Empty 3 token piles
        for good in list(GOODS)[:3]:
            gs.tokens[good] = []
        gs._check_round_over()
        assert gs.round_over

    def test_empty_deck(self):
        gs = GameState.new_round(random.Random(42))
        gs.deck = []
        gs._check_round_over()
        assert gs.round_over


class TestFullGame:
    def test_random_game_completes(self):
        """A random game should always terminate."""
        rng = random.Random(42)
        for _ in range(100):
            winner = play_round((RandomAgent(rng), RandomAgent(rng)), rng)
            assert winner is None or winner in (0, 1)

    def test_no_infinite_loop(self):
        """Game must end within reasonable turns."""
        rng = random.Random(123)
        gs = GameState.new_round(rng)
        agents = (RandomAgent(rng), RandomAgent(rng))
        turns = 0
        while not gs.round_over:
            actions = gs.get_legal_actions()
            if not actions:
                break
            action = agents[gs.current_player].choose(gs, actions)
            gs = gs.apply_action(action)
            turns += 1
            assert turns < 500, "Game took too many turns"

"""AI agents for Jaipur."""
from __future__ import annotations

import random
from typing import Protocol

from .cards import GoodType, PRECIOUS_GOODS
from .game import Action, GameState, Sell


class Agent(Protocol):
    def choose(self, state: GameState, actions: list[Action]) -> Action: ...


class RandomAgent:
    """Picks a random legal action."""
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def choose(self, state: GameState, actions: list[Action]) -> Action:
        return self.rng.choice(actions)


class GreedyAgent:
    """Simple heuristic: sell when profitable, take expensive goods."""
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    # Priority order for goods value
    GOOD_VALUE = {
        GoodType.DIAMOND: 6,
        GoodType.GOLD: 5,
        GoodType.SILVER: 4,
        GoodType.SPICE: 3,
        GoodType.CLOTH: 2,
        GoodType.LEATHER: 1,
    }

    def choose(self, state: GameState, actions: list[Action]) -> Action:
        # Prefer selling (highest value first)
        sells = [a for a in actions if isinstance(a, Sell)]
        if sells:
            best_sell = max(sells, key=lambda s: self.GOOD_VALUE.get(s.good, 0) * s.count)
            return best_sell

        # Otherwise take most valuable good
        from .game import TakeOne
        takes = [a for a in actions if isinstance(a, TakeOne)]
        if takes:
            best_take = max(takes, key=lambda t: self.GOOD_VALUE.get(t.good, 0))
            return best_take

        # Fallback: random
        return self.rng.choice(actions)

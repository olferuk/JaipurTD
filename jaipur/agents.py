"""AI agents for Jaipur (fast engine)."""

import random
from typing import Protocol

from .game_fast import ACT_SELL, ACT_TAKE_ONE, GameState


class Agent(Protocol):
    def choose(self, state: GameState, actions: list[tuple]) -> tuple: ...


class RandomAgent:
    """Picks a random legal action."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def choose(self, state: GameState, actions: list[tuple]) -> tuple:
        return self.rng.choice(actions)


class GreedyAgent:
    """Simple heuristic: sell when profitable, take expensive goods."""

    # Priority: diamond > gold > silver > spice > cloth > leather
    GOOD_VALUE = [6, 5, 4, 3, 2, 1]

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def choose(self, state: GameState, actions: list[tuple]) -> tuple:
        # Prefer selling (highest value first)
        sells = [a for a in actions if a[0] == ACT_SELL]
        if sells:
            return max(sells, key=lambda s: self.GOOD_VALUE[s[1]] * s[2])

        # Take most valuable good
        takes = [a for a in actions if a[0] == ACT_TAKE_ONE]
        if takes:
            return max(takes, key=lambda t: self.GOOD_VALUE[t[1]])

        return self.rng.choice(actions)

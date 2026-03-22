"""Neural network agent for Jaipur."""
import random

import numpy as np
import torch

from jaipur.encoding import encode_state
from jaipur.game import Action, GameState
from ai.network import ValueNetwork


class NeuralAgent:
    """Chooses actions by evaluating resulting states with a value network.
    
    Uses epsilon-greedy exploration during training.
    """

    def __init__(
        self,
        network: ValueNetwork,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
    ):
        self.network = network
        self.epsilon = epsilon
        self.rng = rng or random.Random()

    @torch.no_grad()
    def choose(self, state: GameState, actions: list[Action]) -> Action:
        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)

        player = state.current_player
        best_action = None
        best_value = -1.0

        for action in actions:
            next_state = state.apply_action(action)
            feat = encode_state(next_state, player)
            tensor = torch.from_numpy(feat).unsqueeze(0)
            value = self.network(tensor).item()
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

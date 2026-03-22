"""Neural network agent for Jaipur."""
import random

import numpy as np
import torch

from jaipur.encoding import encode_state
from jaipur.game_fast import GameState
from ai.network import ValueNetwork


class NeuralAgent:
    """Chooses actions by evaluating resulting states with a value network."""

    def __init__(
        self,
        network: ValueNetwork,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
        device: torch.device | None = None,
    ):
        self.network = network
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def choose(self, state: GameState, actions: list[tuple]) -> tuple:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)

        player = state.current_player

        # Batch evaluate all actions at once
        feats = []
        for action in actions:
            next_state = state.apply_action(action)
            feats.append(encode_state(next_state, player))

        batch = torch.from_numpy(np.stack(feats)).to(self.device)
        values = self.network(batch)
        best_idx = values.argmax().item()
        return actions[best_idx]

"""Value network for Jaipur TD-learning."""
import torch
import torch.nn as nn

from jaipur.encoding import FEATURE_SIZE


class ValueNetwork(nn.Module):
    """MLP that estimates P(win) from a game state encoding.
    
    Input: feature vector of size FEATURE_SIZE
    Output: scalar in [0, 1] (win probability)
    """

    def __init__(self, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_SIZE, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

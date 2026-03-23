"""Dual-head (policy + value) network for AlphaZero-style Jaipur agent.

Architecture:
    Shared backbone (state encoding → hidden layers) splits into:
    - Policy head: logits over fixed action space (168 slots)
    - Value head: scalar in [-1, 1] (expected game outcome)

Action Space Design (168 total slots):
    [0-5]     TAKE_ONE(good_idx)           — 6 slots, one per good type
    [6]       TAKE_CAMELS                  — 1 slot
    [7-48]    SELL(good_idx, count)         — 42 slots (6 goods x 7 max count)
              Index = 7 + good_idx * 7 + (count - 1)
    [49-111]  EXCHANGE_2(pair_idx, strat)   — 63 slots (21 take-pairs x 3 give-strategies)
              take-pair: (i,j) with i<=j from {0..5} → 21 combos
              strategy: 0=all_camels, 1=camel+good, 2=all_goods
              Index = 49 + pair_idx * 3 + strategy
    [112-167] EXCHANGE_3(triple_idx)        — 56 slots (56 take-triples x 1 strategy)
              take-triple: (i,j,k) with i<=j<=k from {0..5} → C(8,3)=56
              Index = 112 + triple_idx

    Why this design:
    - The game engine already simplifies exchanges to a practical subset
      (see game_fast._add_exchange_actions), so the mapping is 1:1.
    - Exchange give-strategy is determined by give_camels count:
      0 camels → all_goods, 1 camel → mix, 2 camels → all_camels.
    - Size-3 exchanges use a single strategy (camels-first, then cheapest goods),
      matching the engine's generation logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from jaipur.encoding import FEATURE_SIZE  # 32
from jaipur.game_fast import ACT_TAKE_ONE, ACT_TAKE_CAMELS, ACT_SELL, ACT_EXCHANGE

# ---------------------------------------------------------------------------
# Action space constants
# ---------------------------------------------------------------------------

ACTION_SPACE_SIZE = 168

# Precompute pair/triple index lookups
_PAIR_TO_IDX: dict[tuple[int, int], int] = {}
_IDX_TO_PAIR: list[tuple[int, int]] = []
_idx = 0
for _i in range(6):
    for _j in range(_i, 6):
        _PAIR_TO_IDX[(_i, _j)] = _idx
        _IDX_TO_PAIR.append((_i, _j))
        _idx += 1
assert _idx == 21

_TRIPLE_TO_IDX: dict[tuple[int, int, int], int] = {}
_IDX_TO_TRIPLE: list[tuple[int, int, int]] = []
_idx = 0
for _i in range(6):
    for _j in range(_i, 6):
        for _k in range(_j, 6):
            _TRIPLE_TO_IDX[(_i, _j, _k)] = _idx
            _IDX_TO_TRIPLE.append((_i, _j, _k))
            _idx += 1
assert _idx == 56


# ---------------------------------------------------------------------------
# Action ↔ index mapping
# ---------------------------------------------------------------------------

def action_to_index(action: tuple) -> int:
    """Map a game action tuple to a fixed action-space index (0..167)."""
    atype = action[0]

    if atype == ACT_TAKE_ONE:
        return action[1]  # 0-5

    if atype == ACT_TAKE_CAMELS:
        return 6

    if atype == ACT_SELL:
        g, n = action[1], action[2]
        return 7 + g * 7 + (n - 1)  # 7-48

    if atype == ACT_EXCHANGE:
        take = tuple(sorted(action[1]))
        give_camels = action[3]

        if len(take) == 2:
            pair_idx = _PAIR_TO_IDX[take]
            # Determine give strategy from camel count
            if give_camels >= 2:
                strategy = 0  # all camels
            elif give_camels == 1:
                strategy = 1  # mix
            else:
                strategy = 2  # all goods
            return 49 + pair_idx * 3 + strategy  # 49-111

        if len(take) == 3:
            triple_idx = _TRIPLE_TO_IDX[take]
            return 112 + triple_idx  # 112-167

    raise ValueError(f"Cannot encode action: {action}")


def actions_to_mask(actions: list[tuple], device: torch.device | None = None) -> torch.Tensor:
    """Create a boolean mask (True = legal) over the action space."""
    mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool, device=device)
    for a in actions:
        try:
            mask[action_to_index(a)] = True
        except (KeyError, ValueError):
            pass  # skip actions that don't fit the encoding
    return mask


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class AlphaNetwork(nn.Module):
    """Dual-head network: shared backbone → (policy logits, value scalar).

    Input:  state feature vector of size FEATURE_SIZE (32)
    Output: (policy_logits [ACTION_SPACE_SIZE], value [-1, 1])
    """

    def __init__(self, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(FEATURE_SIZE, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, ACTION_SPACE_SIZE),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: state features, shape (batch, FEATURE_SIZE)

        Returns:
            policy_logits: shape (batch, ACTION_SPACE_SIZE)
            value: shape (batch,) in [-1, 1]
        """
        h = self.backbone(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def predict(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get policy probabilities (masked + softmax) and value.

        Args:
            x: state features, shape (batch, FEATURE_SIZE)
            mask: boolean mask of legal actions, shape (batch, ACTION_SPACE_SIZE)

        Returns:
            policy: probability distribution, shape (batch, ACTION_SPACE_SIZE)
            value: shape (batch,) in [-1, 1]
        """
        logits, value = self.forward(x)
        if mask is not None:
            # Set illegal actions to -inf before softmax
            logits = logits.masked_fill(~mask, float("-inf"))
        policy = F.softmax(logits, dim=-1)
        return policy, value

"""Encode Jaipur game state as a feature vector for the neural network."""
import numpy as np

from .cards import GOODS
from .game_fast import GameState, _N_GOODS


# Feature vector layout:
# [0:6]   current player hand (one per good type)
# [6]     current player camels
# [7:13]  market goods (one per good type)
# [13]    market camels
# [14:20] tokens remaining per good type
# [20:26] max token value per good type
# [26]    current player score (normalized)
# [27]    opponent score (normalized)
# [28]    current player hand size
# [29]    opponent hand size
# [30]    opponent camels
# [31]    deck size (normalized)
# Total: 32 features

FEATURE_SIZE = 32
_MAX_SCORE = 100.0
_MAX_DECK = 40.0


def encode_state(gs: GameState, player: int) -> np.ndarray:
    """Encode game state from perspective of `player` (0 or 1).

    Uses the fast C-level encoder when the Cython engine is active.
    """
    # Fast path: Cython GameState has encode_for() that avoids property overhead
    if hasattr(gs, 'encode_for'):
        return gs.encode_for(player)

    feat = np.zeros(FEATURE_SIZE, dtype=np.float32)

    me = gs.players[player]
    opp = gs.players[1 - player]

    # My hand
    for i in range(_N_GOODS):
        feat[i] = me.hand[i]

    feat[6] = me.camels

    # Market
    for i in range(_N_GOODS):
        feat[7 + i] = gs.market[i]
    feat[13] = gs.market[_N_GOODS]

    # Tokens remaining + next value
    for i in range(_N_GOODS):
        pile = gs.tokens[i]
        feat[14 + i] = len(pile)
        feat[20 + i] = pile[0] if pile else 0

    feat[26] = me.score / _MAX_SCORE
    feat[27] = opp.score / _MAX_SCORE
    feat[28] = me.hand_size
    feat[29] = opp.hand_size
    feat[30] = opp.camels
    feat[31] = gs.deck_size / _MAX_DECK

    return feat

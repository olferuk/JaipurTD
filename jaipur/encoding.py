"""Encode Jaipur game state as a feature vector for the neural network."""
import numpy as np

from .cards import GOODS, TOKENS, GoodType
from .game import GameState


# Feature vector layout:
# [0:6]   current player hand (one per good type)
# [6]     current player camels
# [7:13]  market goods (one per good type)
# [13]    market camels
# [14:20] tokens remaining per good type
# [20:26] max token value per good type (next token you'd get)
# [26]    current player score (normalized)
# [27]    opponent score (normalized)
# [28]    current player hand size
# [29]    opponent hand size
# [30]    opponent camels
# [31]    deck size (normalized)
# [32:38] opponent hand (one per good type) — HIDDEN, always 0 for fair play
# Total: 38 features

FEATURE_SIZE = 38
_GOOD_INDEX = {g: i for i, g in enumerate(GOODS)}
_MAX_SCORE = 100.0  # rough normalization
_MAX_DECK = 40.0


def encode_state(gs: GameState, player: int) -> np.ndarray:
    """Encode game state from perspective of `player` (0 or 1)."""
    feat = np.zeros(FEATURE_SIZE, dtype=np.float32)

    me = gs.players[player]
    opp = gs.players[1 - player]

    # My hand: cards per good type
    for g in GOODS:
        feat[_GOOD_INDEX[g]] = me.hand.get(g, 0)

    # My camels
    feat[6] = me.camels

    # Market goods
    for card in gs.market:
        if card == GoodType.CAMEL:
            feat[13] += 1
        else:
            feat[7 + _GOOD_INDEX[card]] += 1

    # Tokens remaining + next token value
    for g in GOODS:
        i = _GOOD_INDEX[g]
        pile = gs.tokens[g]
        feat[14 + i] = len(pile)
        feat[20 + i] = pile[0] if pile else 0

    # Scores (normalized)
    feat[26] = me.score / _MAX_SCORE
    feat[27] = opp.score / _MAX_SCORE

    # Hand sizes
    feat[28] = me.hand_size
    feat[29] = opp.hand_size

    # Opponent camels
    feat[30] = opp.camels

    # Deck size
    feat[31] = len(gs.deck) / _MAX_DECK

    # feat[32:38] reserved for opponent hand (hidden info, stays 0)

    return feat

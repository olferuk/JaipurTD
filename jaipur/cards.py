"""Card types, tokens, and deck definition for Jaipur."""

from enum import StrEnum


class GoodType(StrEnum):
    DIAMOND = "diamond"
    GOLD = "gold"
    SILVER = "silver"
    CLOTH = "cloth"
    SPICE = "spice"
    LEATHER = "leather"
    CAMEL = "camel"


# How many of each card in the full 55-card deck
CARD_COUNTS: dict[GoodType, int] = {
    GoodType.DIAMOND: 6,
    GoodType.GOLD: 6,
    GoodType.SILVER: 6,
    GoodType.CLOTH: 8,
    GoodType.SPICE: 8,
    GoodType.LEATHER: 10,
    GoodType.CAMEL: 11,
}

# Goods that can be traded (everything except camels)
GOODS = [g for g in GoodType if g != GoodType.CAMEL]

# Precious goods require selling 3+ at a time
PRECIOUS_GOODS = {GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER}

# Token values for each good type (taken from top = index 0, descending value)
TOKENS: dict[GoodType, list[int]] = {
    GoodType.DIAMOND: [7, 7, 5, 5, 5],
    GoodType.GOLD: [6, 6, 5, 5, 5],
    GoodType.SILVER: [5, 5, 5, 5, 5],
    GoodType.CLOTH: [5, 3, 3, 2, 2, 1, 1],
    GoodType.SPICE: [5, 3, 3, 2, 2, 1, 1],
    GoodType.LEATHER: [4, 3, 2, 1, 1, 1, 1, 1, 1],
}

# Bonus token pools (shuffled, drawn randomly)
BONUS_TOKENS_3 = [1, 1, 2, 2, 2, 3, 3]
BONUS_TOKENS_4 = [4, 4, 5, 5, 6, 6]
BONUS_TOKENS_5 = [8, 8, 9, 10, 10]

# Camel majority bonus
CAMEL_BONUS = 5


def build_deck() -> list[GoodType]:
    """Return a full 55-card deck as a list."""
    deck: list[GoodType] = []
    for good, count in CARD_COUNTS.items():
        deck.extend([good] * count)
    return deck

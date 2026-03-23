"""Play 1000 random games and print statistics."""

import random

from jaipur.agents import GreedyAgent, RandomAgent
from jaipur.game_fast import play_match


def main():
    rng = random.Random(42)
    n_games = 1000

    # Random vs Random
    print("=" * 50)
    print("Random vs Random")
    print("=" * 50)
    wins = [0, 0]
    agents = (RandomAgent(rng), RandomAgent(rng))
    for _ in range(n_games):
        w = play_match(agents, rng)
        wins[w] += 1
    print(f"Player 0 wins: {wins[0]} ({wins[0] / n_games * 100:.1f}%)")
    print(f"Player 1 wins: {wins[1]} ({wins[1] / n_games * 100:.1f}%)")

    # Greedy vs Random
    print()
    print("=" * 50)
    print("Greedy vs Random")
    print("=" * 50)
    wins = [0, 0]
    agents = (GreedyAgent(rng), RandomAgent(rng))
    for _ in range(n_games):
        w = play_match(agents, rng)
        wins[w] += 1
    print(f"Greedy wins:  {wins[0]} ({wins[0] / n_games * 100:.1f}%)")
    print(f"Random wins:  {wins[1]} ({wins[1] / n_games * 100:.1f}%)")

    # Greedy vs Greedy
    print()
    print("=" * 50)
    print("Greedy vs Greedy")
    print("=" * 50)
    wins = [0, 0]
    agents = (GreedyAgent(rng), GreedyAgent(rng))
    for _ in range(n_games):
        w = play_match(agents, rng)
        wins[w] += 1
    print(f"Player 0 wins: {wins[0]} ({wins[0] / n_games * 100:.1f}%)")
    print(f"Player 1 wins: {wins[1]} ({wins[1] / n_games * 100:.1f}%)")


if __name__ == "__main__":
    main()

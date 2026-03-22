"""Evaluate neural agent vs baselines."""
import random
from pathlib import Path

import torch

from jaipur.game_fast import play_match
from jaipur.agents import RandomAgent, GreedyAgent
from ai.network import ValueNetwork
from ai.agents import NeuralAgent


def load_network(path: str = "models/value_net.pt") -> ValueNetwork:
    checkpoint = torch.load(path, weights_only=True)
    net = ValueNetwork(checkpoint["hidden1"], checkpoint["hidden2"])
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return net


def evaluate(net: ValueNetwork, n_games: int = 500, seed: int = 99):
    rng = random.Random(seed)

    neural = NeuralAgent(net, epsilon=0.0, rng=rng)
    greedy = GreedyAgent(rng)
    rand = RandomAgent(rng)

    for name, opponent in [("Random", rand), ("Greedy", greedy)]:
        # Neural as player 0
        wins_as_p0 = [0, 0]
        for _ in range(n_games):
            w = play_match((neural, opponent), rng)
            wins_as_p0[w] += 1

        # Neural as player 1
        wins_as_p1 = [0, 0]
        for _ in range(n_games):
            w = play_match((opponent, neural), rng)
            wins_as_p1[w] += 1

        neural_wins = wins_as_p0[0] + wins_as_p1[1]
        total = n_games * 2
        print(f"Neural vs {name:>6}: {neural_wins}/{total} ({neural_wins/total*100:.1f}%)")
        print(f"  As P0: {wins_as_p0[0]}/{n_games} | As P1: {wins_as_p1[1]}/{n_games}")
        print()


if __name__ == "__main__":
    net = load_network()
    evaluate(net)

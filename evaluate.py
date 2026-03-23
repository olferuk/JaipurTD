"""Evaluate neural agent vs baselines + Elo tournament."""

import random
from pathlib import Path

import torch

from ai.agents import NeuralAgent
from ai.network import ValueNetwork
from jaipur.agents import GreedyAgent, RandomAgent
from jaipur.game_fast import play_match


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
        wins_as_p0 = [0, 0]
        for _ in range(n_games):
            w = play_match((neural, opponent), rng)
            wins_as_p0[w] += 1

        wins_as_p1 = [0, 0]
        for _ in range(n_games):
            w = play_match((opponent, neural), rng)
            wins_as_p1[w] += 1

        neural_wins = wins_as_p0[0] + wins_as_p1[1]
        total = n_games * 2
        print(f"Neural vs {name:>6}: {neural_wins}/{total} ({neural_wins / total * 100:.1f}%)")
        print(f"  As P0: {wins_as_p0[0]}/{n_games} | As P1: {wins_as_p1[1]}/{n_games}")
        print()


# ---------- Elo Tournament ----------


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def elo_tournament(agents: dict[str, object], n_rounds: int = 200, seed: int = 42):
    """Round-robin Elo tournament between named agents."""
    rng = random.Random(seed)
    names = list(agents.keys())
    elo = {name: 1500.0 for name in names}
    k = 32.0  # K-factor
    wins = {name: {other: 0 for other in names} for name in names}

    print(f"Elo Tournament: {len(names)} agents, {n_rounds} rounds each pair")
    print("=" * 60)

    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            for _ in range(n_rounds):
                # Alternate who goes first
                if rng.random() < 0.5:
                    pair = (agents[name_a], agents[name_b])
                    w = play_match(pair, rng)
                    winner = name_a if w == 0 else name_b
                else:
                    pair = (agents[name_b], agents[name_a])
                    w = play_match(pair, rng)
                    winner = name_b if w == 0 else name_a

                loser = name_b if winner == name_a else name_a
                wins[winner][loser] += 1

                # Update Elo
                ea = expected_score(elo[name_a], elo[name_b])
                sa = 1.0 if winner == name_a else 0.0
                elo[name_a] += k * (sa - ea)
                elo[name_b] += k * ((1 - sa) - (1 - ea))

    # Print results
    ranked = sorted(elo.items(), key=lambda x: -x[1])
    print(f"\n{'Rank':<6}{'Agent':<20}{'Elo':>8}")
    print("-" * 34)
    for rank, (name, rating) in enumerate(ranked, 1):
        print(f"{rank:<6}{name:<20}{rating:>8.0f}")

    # Win matrix
    print(f"\nWin matrix ({n_rounds} games per pair):")
    header = f"{'':>12}" + "".join(f"{n:>12}" for n in names)
    print(header)
    for name_a in names:
        row = f"{name_a:>12}"
        for name_b in names:
            if name_a == name_b:
                row += f"{'—':>12}"
            else:
                row += f"{wins[name_a][name_b]:>12}"
        print(row)

    return elo


if __name__ == "__main__":
    import sys

    # Load all checkpoints + final model
    models_dir = Path("models")
    agents: dict[str, object] = {}

    rng = random.Random(99)
    agents["Random"] = RandomAgent(rng)
    agents["Greedy"] = GreedyAgent(rng)

    # Load neural agents
    for pt_file in sorted(models_dir.glob("*.pt")):
        net = load_network(str(pt_file))
        name = pt_file.stem
        if name == "value_net":
            name = "Neural (final)"
        else:
            name = name.replace("checkpoint_", "Neural @")
        agents[name] = NeuralAgent(net, epsilon=0.0, rng=rng)

    if len(agents) <= 2:
        print("No trained models found in models/. Run training first.")
        print("  python -m ai.trainer")
        sys.exit(1)

    print("=== Head-to-head vs baselines ===\n")
    if "Neural (final)" in agents:
        final_net = load_network("models/value_net.pt")
        evaluate(final_net)

    print("\n=== Elo Tournament ===\n")
    elo_tournament(agents, n_rounds=200)

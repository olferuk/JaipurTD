"""TD(0) self-play training loop for Jaipur — v2 with improvements."""
import random
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from jaipur.encoding import encode_state
from jaipur.game_fast import GameState, play_match
from ai.network import ValueNetwork
from ai.agents import NeuralAgent


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_self_play(
    n_episodes: int = 100_000,
    lr_max: float = 0.001,
    lr_min: float = 0.00001,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,
    hidden1: int = 128,
    hidden2: int = 64,
    batch_size: int = 512,
    greedy_mix: float = 0.3,  # fraction of games vs Greedy
    save_path: str = "models/value_net.pt",
    checkpoint_every: int = 10_000,
    eval_every: int = 5_000,
    eval_games: int = 200,
    seed: int = 42,
) -> ValueNetwork:
    """Train with TD(0): self-play + Greedy opponents, batched updates, cosine LR."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    network = ValueNetwork(hidden1, hidden2).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr_max)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_episodes, eta_min=lr_min
    )

    # Greedy opponent for mixed training
    from jaipur.agents import GreedyAgent, RandomAgent
    greedy = GreedyAgent(rng)

    # Batch buffers for TD updates
    state_buf: list[np.ndarray] = []
    target_buf: list[float] = []

    best_vs_greedy = 0.0
    best_path = Path(save_path).parent / "best.pt"

    pbar = tqdm(range(1, n_episodes + 1), desc="Training", unit="ep")
    for episode in pbar:
        progress = episode / n_episodes
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

        neural_agent = NeuralAgent(network, epsilon=epsilon, rng=rng, device=device)

        # Mix: some games vs Greedy, rest self-play
        if rng.random() < greedy_mix:
            # vs Greedy: randomly assign sides
            if rng.random() < 0.5:
                agents = (neural_agent, greedy)
                neural_player = 0
            else:
                agents = (greedy, neural_agent)
                neural_player = 1
        else:
            agents = (neural_agent, neural_agent)
            neural_player = -1  # both are neural

        gs = GameState.new_round(rng)
        # Track states per player for TD chain
        prev_feat: dict[int, np.ndarray | None] = {0: None, 1: None}

        while not gs.round_over:
            actions = gs.get_legal_actions()
            if not actions:
                break

            player = gs.current_player
            feat = encode_state(gs, player)

            # TD target: previous state -> current value
            if prev_feat[player] is not None:
                # Only train on neural player's states (or both in self-play)
                if neural_player == -1 or neural_player == player:
                    with torch.no_grad():
                        t = torch.from_numpy(feat).unsqueeze(0).to(device)
                        current_value = network(t).item()
                    state_buf.append(prev_feat[player])
                    target_buf.append(current_value)

            prev_feat[player] = feat

            action = agents[player].choose(gs, actions)
            gs = gs.apply_action(action)

        # Terminal signals
        winner = gs.round_winner()
        for p in (0, 1):
            if prev_feat[p] is not None:
                if neural_player == -1 or neural_player == p:
                    final_value = 1.0 if winner == p else (0.5 if winner is None else 0.0)
                    state_buf.append(prev_feat[p])
                    target_buf.append(final_value)

        # Batch update — drain buffer in chunks
        last_loss = None
        while len(state_buf) >= batch_size:
            states_t = torch.from_numpy(np.stack(state_buf[:batch_size])).to(device)
            targets_t = torch.tensor(target_buf[:batch_size], device=device)

            values = network(states_t)
            last_loss = F.mse_loss(values, targets_t)

            optimizer.zero_grad()
            last_loss.backward()
            optimizer.step()
            scheduler.step()

            state_buf = state_buf[batch_size:]
            target_buf = target_buf[batch_size:]

        if episode % 100 == 0 and last_loss is not None:
            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{last_loss.item():.4f}",
                ε=f"{epsilon:.3f}",
                lr=f"{lr_now:.1e}",
            )

        # Checkpoint
        if episode % checkpoint_every == 0:
            cp_path = Path(save_path).parent / f"checkpoint_{episode}.pt"
            cp_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hidden1": hidden1,
                "hidden2": hidden2,
                "episodes": episode,
            }, cp_path)
            tqdm.write(f"  💾 Checkpoint: {cp_path}")

        # Periodic eval vs Greedy (early stopping)
        if episode % eval_every == 0:
            win_rate = _eval_vs_greedy(network, eval_games, rng, device)
            tqdm.write(f"  📊 vs Greedy: {win_rate:.1%} (best: {best_vs_greedy:.1%})")
            if win_rate > best_vs_greedy:
                best_vs_greedy = win_rate
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": network.state_dict(),
                    "hidden1": hidden1,
                    "hidden2": hidden2,
                    "episodes": episode,
                    "win_rate_vs_greedy": win_rate,
                }, best_path)
                tqdm.write(f"  🏆 New best! Saved to {best_path}")

    pbar.close()

    # Save final
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": network.state_dict(),
        "hidden1": hidden1,
        "hidden2": hidden2,
        "episodes": n_episodes,
    }, save_path)
    print(f"\nFinal model: {save_path}")
    print(f"Best model:  {best_path} ({best_vs_greedy:.1%} vs Greedy)")

    return network


def _eval_vs_greedy(
    network: ValueNetwork, n_games: int, rng: random.Random, device: torch.device,
) -> float:
    """Quick eval: return win rate vs Greedy."""
    from jaipur.agents import GreedyAgent

    network.eval()
    neural = NeuralAgent(network, epsilon=0.0, rng=rng, device=device)
    greedy = GreedyAgent(rng)
    wins = 0

    for i in range(n_games):
        if i % 2 == 0:
            w = play_match((neural, greedy), rng)
            if w == 0:
                wins += 1
        else:
            w = play_match((greedy, neural), rng)
            if w == 1:
                wins += 1

    network.train()
    return wins / n_games


if __name__ == "__main__":
    train_self_play()

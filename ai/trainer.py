"""TD(0) self-play training loop for Jaipur."""
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from jaipur.encoding import encode_state
from jaipur.game_fast import GameState, play_round
from ai.network import ValueNetwork
from ai.agents import NeuralAgent


def td_update(
    network: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    state_feat: torch.Tensor,
    next_value: float,
) -> float:
    """Single TD(0) update. Returns the TD error."""
    v = network(state_feat.unsqueeze(0))
    target = torch.tensor([next_value])
    loss = F.mse_loss(v, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_self_play(
    n_episodes: int = 50_000,
    lr: float = 0.001,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,
    hidden1: int = 128,
    hidden2: int = 64,
    save_path: str = "models/value_net.pt",
    checkpoint_every: int = 10_000,
    seed: int = 42,
) -> ValueNetwork:
    """Train a value network via TD(0) self-play.
    
    Both players share the same network. After each move, we do a TD update
    from the current player's perspective.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    network = ValueNetwork(hidden1, hidden2)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    wins = [0, 0]
    recent_errors: list[float] = []

    pbar = tqdm(range(1, n_episodes + 1), desc="Training", unit="ep")
    for episode in pbar:
        # Decay epsilon linearly
        progress = episode / n_episodes
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

        agent0 = NeuralAgent(network, epsilon=epsilon, rng=rng)
        agent1 = NeuralAgent(network, epsilon=epsilon, rng=rng)
        agents = (agent0, agent1)

        gs = GameState.new_round(rng)
        prev_feat: dict[int, torch.Tensor | None] = {0: None, 1: None}

        while not gs.round_over:
            actions = gs.get_legal_actions()
            if not actions:
                break

            player = gs.current_player
            feat = torch.from_numpy(encode_state(gs, player))

            # TD update for this player's previous state
            if prev_feat[player] is not None:
                with torch.no_grad():
                    current_value = network(feat.unsqueeze(0)).item()
                td_err = td_update(network, optimizer, prev_feat[player], current_value)
                recent_errors.append(td_err)

            prev_feat[player] = feat

            # Choose and apply action
            action = agents[player].choose(gs, actions)
            gs = gs.apply_action(action)

        # Terminal update: round is over
        winner = gs.round_winner()
        for p in (0, 1):
            if prev_feat[p] is not None:
                final_value = 1.0 if winner == p else (0.5 if winner is None else 0.0)
                td_update(network, optimizer, prev_feat[p], final_value)

        if winner is not None:
            wins[winner] += 1

        # Update progress bar
        if episode % 100 == 0:
            avg_err = sum(recent_errors[-500:]) / max(len(recent_errors[-500:]), 1)
            total_w = wins[0] + wins[1]
            p0 = wins[0] / total_w * 100 if total_w else 50
            pbar.set_postfix(
                ε=f"{epsilon:.3f}",
                td_err=f"{avg_err:.4f}",
                p0_win=f"{p0:.1f}%",
            )

        # Periodic checkpoint
        if episode % checkpoint_every == 0:
            cp_path = Path(save_path).parent / f"checkpoint_{episode}.pt"
            cp_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hidden1": hidden1,
                "hidden2": hidden2,
                "episodes": episode,
                "epsilon": epsilon,
            }, cp_path)
            tqdm.write(f"  💾 Checkpoint saved: {cp_path}")

    pbar.close()

    # Save final model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": network.state_dict(),
        "hidden1": hidden1,
        "hidden2": hidden2,
        "episodes": n_episodes,
    }, save_path)
    print(f"\nModel saved to {save_path}")

    return network


if __name__ == "__main__":
    train_self_play()

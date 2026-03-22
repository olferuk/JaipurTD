"""Self-play training loop for AlphaZero-style Jaipur agent.

Training pipeline:
    1. Generate games via MCTS self-play → (state, policy_target, value_target)
    2. Train dual-head network on collected data
    3. Periodically evaluate vs GreedyAgent and NeuralAgent baselines

Usage:
    python -m ai.alpha_trainer                    # default settings
    python -m ai.alpha_trainer --iterations 50    # custom iteration count
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from jaipur.encoding import encode_state
from jaipur.game_fast import GameState, play_match
from jaipur.agents import GreedyAgent

from ai.alpha_network import (
    AlphaNetwork, ACTION_SPACE_SIZE,
    action_to_index, actions_to_mask,
)
from ai.alpha_mcts import MCTS, AlphaZeroAgent


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-capacity ring buffer for training samples."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.states: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []
        self.values: list[float] = []
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def push(self, state: np.ndarray, policy: np.ndarray, value: float) -> None:
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.policies.append(policy)
            self.values.append(value)
        else:
            self.states[self.pos] = state
            self.policies[self.pos] = policy
            self.values[self.pos] = value
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int, rng: random.Random) -> tuple:
        """Return (states, policies, values) tensors."""
        n = len(self)
        indices = [rng.randint(0, n - 1) for _ in range(batch_size)]
        states = np.stack([self.states[i] for i in indices])
        policies = np.stack([self.policies[i] for i in indices])
        values = np.array([self.values[i] for i in indices], dtype=np.float32)
        return (
            torch.from_numpy(states),
            torch.from_numpy(policies),
            torch.from_numpy(values),
        )


# ---------------------------------------------------------------------------
# Self-play game generation
# ---------------------------------------------------------------------------


def play_self_play_game(
    mcts: MCTS,
    rng: random.Random,
    temperature_moves: int = 15,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Play one self-play game, returning training data.

    Args:
        mcts: MCTS search engine (shared network)
        rng: random number generator
        temperature_moves: apply temperature=1.0 for this many moves,
            then switch to near-greedy (temperature=0.1)

    Returns:
        List of (state_features, mcts_policy, player) tuples.
        Value targets are filled in after the game ends.
    """
    gs = GameState.new_round(rng)
    trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_count = 0

    while not gs.round_over:
        actions = gs.get_legal_actions()
        if not actions:
            break

        player = gs.current_player
        temp = 1.0 if move_count < temperature_moves else 0.1

        # MCTS search
        policy, _ = mcts.search(gs, player, rng, temperature=temp)

        # Record training sample
        feat = encode_state(gs, player)
        trajectory.append((feat, policy, player))

        # Select action from policy
        legal_indices = []
        action_map: dict[int, tuple] = {}
        for a in actions:
            try:
                idx = action_to_index(a)
                legal_indices.append(idx)
                action_map[idx] = a
            except (KeyError, ValueError):
                continue

        if not legal_indices:
            action = rng.choice(actions)
        else:
            probs = np.array([policy[i] for i in legal_indices], dtype=np.float32)
            total = probs.sum()
            if total < 1e-8:
                action = rng.choice(actions)
            else:
                probs /= total
                chosen = rng.choices(legal_indices, weights=probs, k=1)[0]
                action = action_map[chosen]

        gs = gs.apply_action(action)
        move_count += 1

    # Determine outcome
    winner = gs.round_winner()

    # Build training samples with value targets
    samples = []
    for feat, policy, player in trajectory:
        if winner is None:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        samples.append((feat, policy, value))

    return samples


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_batch(
    network: AlphaNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
    value_weight: float = 1.0,
) -> tuple[float, float, float]:
    """One training step on a batch from replay buffer.

    Returns:
        (total_loss, policy_loss, value_loss) as floats
    """
    states, policies, values = replay.sample(batch_size, rng)
    states = states.to(device)
    policies = policies.to(device)
    values = values.to(device)

    logits, pred_values = network(states)

    # Policy loss: cross-entropy between MCTS policy and network output
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(policies * log_probs).sum(dim=-1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(pred_values, values)

    total_loss = policy_loss + value_weight * value_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_vs_baseline(
    network: AlphaNetwork,
    opponent,
    n_games: int,
    simulations: int,
    rng: random.Random,
    device: torch.device,
) -> float:
    """Win rate of AlphaZero agent vs a baseline opponent."""
    network.eval()
    agent = AlphaZeroAgent(
        network,
        simulations=simulations,
        determinizations=4,
        temperature=0.1,
        rng=rng,
        device=device,
    )
    wins = 0
    for i in range(n_games):
        if i % 2 == 0:
            w = play_match((agent, opponent), rng)
            if w == 0:
                wins += 1
        else:
            w = play_match((opponent, agent), rng)
            if w == 1:
                wins += 1
    network.train()
    return wins / n_games


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train_alpha(
    iterations: int = 30,
    games_per_iter: int = 100,
    train_steps_per_iter: int = 200,
    batch_size: int = 256,
    simulations: int = 50,
    determinizations: int = 4,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    hidden1: int = 128,
    hidden2: int = 64,
    buffer_capacity: int = 100_000,
    eval_games: int = 40,
    eval_simulations: int = 30,
    save_dir: str = "models/alpha",
    seed: int = 42,
) -> AlphaNetwork:
    """Full AlphaZero training loop.

    Each iteration:
        1. Self-play: generate training games via MCTS
        2. Train: optimize network on replay buffer
        3. Evaluate: test vs GreedyAgent baseline
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    network = AlphaNetwork(hidden1, hidden2).to(device)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=lr / 100,
    )

    replay = ReplayBuffer(buffer_capacity)
    greedy = GreedyAgent(rng)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_vs_greedy = 0.0

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{iterations}")
        print(f"{'='*60}")

        # --- Self-play ---
        network.eval()
        mcts = MCTS(
            network=network,
            simulations=simulations,
            determinizations=determinizations,
            device=device,
        )

        total_samples = 0
        pbar = tqdm(range(games_per_iter), desc="Self-play", unit="game")
        for _ in pbar:
            samples = play_self_play_game(mcts, rng)
            for feat, policy, value in samples:
                replay.push(feat, policy, value)
                total_samples += 1
            pbar.set_postfix(samples=total_samples, buffer=len(replay))

        # --- Training ---
        if len(replay) < batch_size:
            print("Not enough samples, skipping training")
            continue

        network.train()
        total_ploss = 0.0
        total_vloss = 0.0

        pbar = tqdm(range(train_steps_per_iter), desc="Training", unit="step")
        for step in pbar:
            loss, ploss, vloss = train_batch(
                network, optimizer, replay, batch_size, device, rng,
            )
            total_ploss += ploss
            total_vloss += vloss
            if (step + 1) % 50 == 0:
                pbar.set_postfix(
                    p_loss=f"{total_ploss / (step + 1):.4f}",
                    v_loss=f"{total_vloss / (step + 1):.4f}",
                )

        scheduler.step()
        avg_ploss = total_ploss / train_steps_per_iter
        avg_vloss = total_vloss / train_steps_per_iter
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Avg policy_loss={avg_ploss:.4f}  value_loss={avg_vloss:.4f}  lr={lr_now:.1e}")

        # --- Evaluation ---
        print("  Evaluating vs Greedy...")
        win_rate = evaluate_vs_baseline(
            network, greedy, eval_games, eval_simulations, rng, device,
        )
        print(f"  vs Greedy: {win_rate:.1%} (best: {best_vs_greedy:.1%})")

        # Save checkpoint
        checkpoint = {
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hidden1": hidden1,
            "hidden2": hidden2,
            "iteration": iteration,
            "win_rate_vs_greedy": win_rate,
        }
        torch.save(checkpoint, save_path / f"checkpoint_{iteration:03d}.pt")

        if win_rate > best_vs_greedy:
            best_vs_greedy = win_rate
            torch.save(checkpoint, save_path / "best.pt")
            print(f"  New best! Saved to {save_path / 'best.pt'}")

    # Save final model
    torch.save({
        "model_state_dict": network.state_dict(),
        "hidden1": hidden1,
        "hidden2": hidden2,
        "iteration": iterations,
        "win_rate_vs_greedy": best_vs_greedy,
    }, save_path / "final.pt")
    print(f"\nTraining complete. Final model: {save_path / 'final.pt'}")
    print(f"Best vs Greedy: {best_vs_greedy:.1%}")

    return network


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero training for Jaipur")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games-per-iter", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models/alpha")
    args = parser.parse_args()

    train_alpha(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        train_steps_per_iter=args.train_steps,
        simulations=args.simulations,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.save_dir,
    )

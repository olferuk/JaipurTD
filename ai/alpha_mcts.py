"""Neural MCTS (Monte Carlo Tree Search) for AlphaZero-style Jaipur agent.

Implements PUCT-based tree search with neural network evaluation at leaf nodes
instead of random rollouts. Handles Jaipur's hidden information (opponent hand,
deck order) via determinization: at each MCTS simulation, unknown cards are
randomly sampled into a consistent assignment before tree traversal.

Usage:
    network = AlphaNetwork()
    agent = AlphaZeroAgent(network, simulations=100)
    action = agent.choose(game_state, legal_actions)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from jaipur.cards import CARD_COUNTS, GOODS, GoodType
from jaipur.encoding import encode_state
from jaipur.game_fast import GameState, _N_GOODS, ACT_SELL

from ai.alpha_network import (
    AlphaNetwork, ACTION_SPACE_SIZE,
    action_to_index, actions_to_mask,
)

# ---------------------------------------------------------------------------
# MCTS node
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    """A node in the MCTS tree, storing visit counts and value estimates."""
    state: GameState
    player: int  # player who just moved (parent's current_player)
    parent: MCTSNode | None = None
    action: tuple | None = None  # action that led here
    children: dict[int, MCTSNode] = field(default_factory=dict)  # action_idx → child
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0  # P(a) from network policy
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Mean value Q(s,a) from this node's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


# ---------------------------------------------------------------------------
# Determinization — sample hidden information
# ---------------------------------------------------------------------------


def determinize(gs: GameState, observer: int, rng: random.Random) -> GameState:
    """Create a plausible game state by sampling hidden information.

    The observer knows: their own hand, market, both camel counts, hand sizes,
    deck size, and token piles. Unknown: opponent's hand composition and deck
    contents/order.

    Approach:
        1. Compute all cards not directly visible to observer.
        2. Subtract known camel assignments (opponent camels are public info).
        3. Randomly assign cards to opponent's hand (matching known hand_size)
           and deck (matching known deck_size).
    """
    gs = gs.copy()
    me = gs.players[observer]
    opp = gs.players[1 - observer]

    # Save opponent hand size before we reset it (it's a computed property)
    opp_hand_size = opp.hand_size

    # Count cards visible to observer (per type: 0-5 goods, 6 camel)
    seen = [0] * 7
    for i in range(_N_GOODS):
        seen[i] = me.hand[i] + gs.market[i]
    seen[6] = me.camels + gs.market[_N_GOODS] + opp.camels

    # Build pool of unknown cards
    card_counts = [CARD_COUNTS[g] for g in GOODS] + [CARD_COUNTS[GoodType.CAMEL]]
    unknown: list[int] = []
    for i in range(7):
        n = card_counts[i] - seen[i]
        if n > 0:
            unknown.extend([i] * n)

    # We need exactly (opp_hand_size + deck_size) cards from unknown pool.
    # If pool is larger (due to cards discarded via selling), randomly drop excess.
    needed = opp_hand_size + gs.deck_size
    rng.shuffle(unknown)
    if len(unknown) > needed:
        unknown = unknown[:needed]
    elif len(unknown) < needed:
        # Shouldn't happen in a valid state; fall back gracefully
        return gs

    # Separate into goods and camels, then assign
    goods_pool = [c for c in unknown if c < _N_GOODS]
    camel_pool = [c for c in unknown if c == _N_GOODS]
    rng.shuffle(goods_pool)

    # Assign opponent hand (exactly opp_hand_size good cards)
    opp.hand = [0] * _N_GOODS
    for i in range(min(opp_hand_size, len(goods_pool))):
        opp.hand[goods_pool[i]] += 1
    remaining_goods = goods_pool[opp_hand_size:]

    # Remaining cards form the deck
    deck_cards = remaining_goods + camel_pool
    rng.shuffle(deck_cards)
    gs.deck = deck_cards
    gs.deck_size = len(deck_cards)

    return gs


# ---------------------------------------------------------------------------
# MCTS engine
# ---------------------------------------------------------------------------


class MCTS:
    """Neural MCTS with PUCT selection and determinization.

    Args:
        network: dual-head AlphaNetwork for policy/value evaluation
        c_puct: exploration constant for PUCT formula
        simulations: number of MCTS simulations per move
        determinizations: number of random card samplings per move
        device: torch device for network inference
    """

    def __init__(
        self,
        network: AlphaNetwork,
        c_puct: float = 1.5,
        simulations: int = 100,
        determinizations: int = 8,
        device: torch.device | None = None,
    ):
        self.network = network
        self.c_puct = c_puct
        self.simulations = simulations
        self.determinizations = determinizations
        self.device = device or torch.device("cpu")

    def search(
        self,
        root_state: GameState,
        player: int,
        rng: random.Random,
        temperature: float = 1.0,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """Run MCTS from root_state and return action visit counts.

        Args:
            root_state: current game state
            player: perspective player (for value interpretation)
            rng: random number generator
            temperature: controls exploration in final action selection

        Returns:
            policy: probability distribution over ACTION_SPACE_SIZE
            visit_counts: dict mapping action_index → visit count
        """
        # Aggregate visit counts across determinizations
        total_visits: dict[int, int] = {}
        sims_per_det = max(1, self.simulations // self.determinizations)

        for _ in range(self.determinizations):
            det_state = determinize(root_state, player, rng)
            root = MCTSNode(state=det_state, player=1 - det_state.current_player)

            for _ in range(sims_per_det):
                self._simulate(root, player, rng)

            for action_idx, child in root.children.items():
                total_visits[action_idx] = (
                    total_visits.get(action_idx, 0) + child.visit_count
                )

        # Convert to policy distribution
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if not total_visits:
            return policy, total_visits

        if temperature < 1e-6:
            # Greedy: pick the most-visited action
            best = max(total_visits, key=total_visits.get)
            policy[best] = 1.0
        else:
            # Temperature-scaled visit counts
            indices = list(total_visits.keys())
            counts = np.array([total_visits[i] for i in indices], dtype=np.float32)
            counts = counts ** (1.0 / temperature)
            total = counts.sum()
            if total > 0:
                counts /= total
            for idx, prob in zip(indices, counts):
                policy[idx] = prob

        return policy, total_visits

    def _simulate(self, root: MCTSNode, perspective: int, rng: random.Random) -> float:
        """Run one MCTS simulation: select → expand → evaluate → backprop."""
        node = root
        path: list[MCTSNode] = [node]

        # SELECT: traverse tree using PUCT until we hit an unexpanded node
        while node.is_expanded and node.children:
            node = self._select_child(node)
            path.append(node)

        # Check terminal
        if node.state.round_over:
            winner = node.state.round_winner()
            if winner is None:
                value = 0.0
            elif winner == perspective:
                value = 1.0
            else:
                value = -1.0
        else:
            # EXPAND + EVALUATE
            value = self._expand_and_evaluate(node, perspective, rng)

        # BACKPROPAGATE
        for n in reversed(path):
            n.visit_count += 1
            # Value is from perspective player's view; flip for opponent nodes
            if n.player == perspective:
                n.total_value += value
            else:
                n.total_value -= value

        return value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest PUCT score."""
        sqrt_parent = math.sqrt(node.visit_count)
        best_score = float("-inf")
        best_child = None

        for child in node.children.values():
            q = child.q_value
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    @torch.no_grad()
    def _expand_and_evaluate(
        self, node: MCTSNode, perspective: int, rng: random.Random,
    ) -> float:
        """Expand node: add children for legal actions, evaluate with network."""
        actions = node.state.get_legal_actions()
        if not actions:
            node.is_expanded = True
            return 0.0

        # Network evaluation
        feat = encode_state(node.state, node.state.current_player)
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        mask = actions_to_mask(actions, self.device).unsqueeze(0)

        policy, value = self.network.predict(feat_t, mask)
        policy = policy.squeeze(0).cpu().numpy()
        value_scalar = value.item()

        # Convert value to perspective player's viewpoint
        if node.state.current_player != perspective:
            value_scalar = -value_scalar

        # Create child nodes
        for action in actions:
            try:
                action_idx = action_to_index(action)
            except (KeyError, ValueError):
                continue

            child_state = node.state.apply_action(action)
            child = MCTSNode(
                state=child_state,
                player=node.state.current_player,
                parent=node,
                action=action,
                prior=float(policy[action_idx]),
            )
            node.children[action_idx] = child

        node.is_expanded = True
        return value_scalar


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AlphaZeroAgent:
    """AlphaZero-style agent using MCTS + neural network.

    Implements the Agent protocol: choose(state, actions) → action.
    """

    def __init__(
        self,
        network: AlphaNetwork,
        simulations: int = 100,
        determinizations: int = 8,
        c_puct: float = 1.5,
        temperature: float = 0.1,
        rng: random.Random | None = None,
        device: torch.device | None = None,
    ):
        self.network = network
        self.temperature = temperature
        self.rng = rng or random.Random()
        self.device = device or torch.device("cpu")
        self.mcts = MCTS(
            network=network,
            c_puct=c_puct,
            simulations=simulations,
            determinizations=determinizations,
            device=self.device,
        )

    def choose(self, state: GameState, actions: list[tuple]) -> tuple:
        """Choose an action via MCTS search."""
        if len(actions) == 1:
            return actions[0]

        player = state.current_player
        policy, visit_counts = self.mcts.search(
            state, player, self.rng, temperature=self.temperature,
        )

        # Map policy back to legal actions
        action_map: dict[int, tuple] = {}
        for a in actions:
            try:
                action_map[action_to_index(a)] = a
            except (KeyError, ValueError):
                continue

        if not action_map:
            return self.rng.choice(actions)

        # Select action based on policy
        legal_indices = list(action_map.keys())
        probs = np.array([policy[i] for i in legal_indices], dtype=np.float32)
        total = probs.sum()
        if total < 1e-8:
            # Uniform fallback
            return self.rng.choice(actions)
        probs /= total

        chosen_idx = self.rng.choices(legal_indices, weights=probs, k=1)[0]
        return action_map[chosen_idx]

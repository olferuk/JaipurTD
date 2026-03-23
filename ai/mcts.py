"""Monte Carlo Tree Search agent for Jaipur with information set sampling.

Jaipur has hidden information (opponent's hand, deck order), so we use
determinized MCTS: at the root we sample possible hidden states, run MCTS
on each, and aggregate action visit counts to pick the best move.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from jaipur.game_fast import GameState, PlayerState, _N_GOODS
from jaipur.cards import CARD_COUNTS, GoodType, GOODS


def _determinize(state: GameState, player: int, rng: random.Random) -> GameState:
    """Create a determinized copy of the game state.

    Randomizes all hidden information from the perspective of `player`:
    - Opponent's hand cards
    - Remaining deck cards and their order

    The market, current player's hand, camels, scores, and tokens are kept.
    """
    gs = state.copy()
    opp = 1 - player
    opp_state = gs.players[opp]

    # Pool all unknown cards: opponent hand + deck
    unknown: list[int] = list(gs.deck)
    for i in range(_N_GOODS):
        unknown.extend([i] * opp_state.hand[i])

    rng.shuffle(unknown)

    # Re-deal opponent hand (same total count, random cards)
    opp_hand_size = opp_state.hand_size
    new_hand = [0] * _N_GOODS
    for _ in range(opp_hand_size):
        card = unknown.pop()
        # If we draw a camel (index 6), put it back and try again
        # (opponent hand doesn't hold camels; they go to herd automatically)
        # But the deck can have camels. We need to handle this.
        if card == 6:
            # Camel can't go in hand — swap with a non-camel from remaining pool
            swapped = False
            for j in range(len(unknown)):
                if unknown[j] != 6:
                    unknown[j], card = card, unknown[j]
                    swapped = True
                    break
            if not swapped:
                # Only camels left — just keep fewer hand cards
                unknown.append(card)
                break
        new_hand[card] += 1

    opp_state.hand = new_hand
    gs.deck = unknown
    gs.deck_size = len(unknown)
    return gs


class _MCTSNode:
    """A node in the MCTS search tree."""

    __slots__ = ("state", "action", "parent", "children",
                 "visits", "total_value", "untried_actions")

    def __init__(self, state: GameState, action: Optional[tuple] = None,
                 parent: Optional[_MCTSNode] = None):
        self.state = state
        self.action = action  # action that led here
        self.parent = parent
        self.children: list[_MCTSNode] = []
        self.visits: int = 0
        self.total_value: float = 0.0
        self.untried_actions: Optional[list[tuple]] = None

    def is_fully_expanded(self) -> bool:
        if self.untried_actions is None:
            self.untried_actions = self.state.get_legal_actions()
        return len(self.untried_actions) == 0

    def ucb1(self, exploration: float, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.total_value / self.visits
        explore = exploration * math.sqrt(math.log(parent_visits) / self.visits)
        return exploit + explore

    def best_child(self, exploration: float) -> _MCTSNode:
        pv = self.visits
        return max(self.children, key=lambda c: c.ucb1(exploration, pv))


_GOOD_VALUE = [6, 5, 4, 3, 2, 1]  # diamond > gold > silver > spice > cloth > leather


def _rollout(state: GameState, player: int, rng: random.Random) -> float:
    """Semi-greedy playout to terminal state. Returns reward for `player` in [0, 1].

    Uses an epsilon-greedy rollout: 70% of the time pick a heuristically good move
    (sell highest value, take most valuable good), 30% random for diversity.
    """
    from jaipur.game_fast import ACT_SELL, ACT_TAKE_ONE

    gs = state
    depth = 0
    max_depth = 100

    while not gs.round_over and depth < max_depth:
        actions = gs.get_legal_actions()
        if not actions:
            break

        if rng.random() < 0.3:
            action = rng.choice(actions)
        else:
            # Heuristic: prefer sells (highest value), then take valuable goods
            sells = [a for a in actions if a[0] == ACT_SELL]
            if sells:
                action = max(sells, key=lambda s: _GOOD_VALUE[s[1]] * s[2])
            else:
                takes = [a for a in actions if a[0] == ACT_TAKE_ONE]
                if takes:
                    action = max(takes, key=lambda t: _GOOD_VALUE[t[1]])
                else:
                    action = rng.choice(actions)

        gs = gs.apply_action(action)
        depth += 1

    winner = gs.round_winner()
    if winner is None:
        return 0.5
    return 1.0 if winner == player else 0.0


def _mcts_search(root_state: GameState, player: int, num_simulations: int,
                 exploration: float, rng: random.Random) -> dict[tuple, int]:
    """Run MCTS from a single determinized root state.

    Returns a dict mapping action -> visit count.
    """
    root = _MCTSNode(root_state)
    root.untried_actions = root_state.get_legal_actions()

    if not root.untried_actions:
        return {}

    for _ in range(num_simulations):
        node = root

        # Selection: walk tree using UCB1
        while not node.state.round_over and node.is_fully_expanded() and node.children:
            node = node.best_child(exploration)

        # Expansion: add one child if not terminal
        if not node.state.round_over and node.untried_actions:
            action = node.untried_actions.pop()
            child_state = node.state.apply_action(action)
            child = _MCTSNode(child_state, action=action, parent=node)
            child.untried_actions = child_state.get_legal_actions()
            node.children.append(child)
            node = child

        # Simulation (rollout)
        reward = _rollout(node.state, player, rng)

        # Backpropagation — each node stores value from the PARENT's
        # perspective (the player who chose the action leading here).
        # node.state.current_player is the NEXT mover, i.e. the opponent
        # of whoever made the move. So if next mover == player, the parent
        # was the opponent and we store opponent's reward (1 - reward).
        while node is not None:
            node.visits += 1
            if node.parent is None:
                # Root node — value not used for selection
                node.total_value += reward
            elif node.state.current_player == player or node.state.round_over:
                # Parent was opponent → store opponent's value
                node.total_value += 1.0 - reward
            else:
                # Parent was player → store player's value
                node.total_value += reward
            node = node.parent

    # Collect visit counts for root children
    action_visits: dict[tuple, int] = {}
    for child in root.children:
        action_visits[child.action] = child.visits
    return action_visits


class MCTSAgent:
    """MCTS agent with information set sampling (determinization).

    For each decision, samples multiple determinizations of hidden information,
    runs MCTS on each, and aggregates visit counts to select the best action.

    Args:
        num_simulations: MCTS iterations per determinization.
        exploration_constant: UCB1 exploration weight (default sqrt(2)).
        num_determinizations: Number of hidden-state samples to aggregate.
        rng: Random number generator for reproducibility.
    """

    def __init__(
        self,
        num_simulations: int = 100,
        exploration_constant: float = 1.414,
        num_determinizations: int = 8,
        rng: random.Random | None = None,
    ):
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.num_determinizations = num_determinizations
        self.rng = rng or random.Random()

    def choose(self, state: GameState, actions: list[tuple]) -> tuple:
        """Choose the best action using determinized MCTS."""
        if len(actions) == 1:
            return actions[0]

        player = state.current_player

        # Aggregate visit counts across determinizations
        total_visits: dict[tuple, int] = {}
        for action in actions:
            total_visits[action] = 0

        sims_per_det = max(1, self.num_simulations // self.num_determinizations)

        for _ in range(self.num_determinizations):
            det_state = _determinize(state, player, self.rng)
            visits = _mcts_search(
                det_state, player, sims_per_det,
                self.exploration_constant, self.rng,
            )
            for action, count in visits.items():
                if action in total_visits:
                    total_visits[action] += count

        # Pick action with most aggregate visits
        best_action = max(total_visits, key=lambda a: total_visits[a])

        # Fallback if MCTS found nothing (shouldn't happen)
        if total_visits[best_action] == 0:
            return self.rng.choice(actions)

        return best_action

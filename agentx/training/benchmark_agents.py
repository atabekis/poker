# rl_agent/training/benchmark_agents.py

"""
This module defines the non-learning benchmark agents to train against.
In our training logic we don't use the agents from the game client to decouple the benchmark agents from the game logic.
This way we can train/benchmark the RL agent in a massively parallel way.
"""

# Python imports
import os
import json
import random
from abc import ABC, abstractmethod

# Local imports
from .kuhn_env import KuhnPokerEnv


class BaseAgent(ABC):
    """abstract base class for all simple agents."""
    @abstractmethod
    def act(self, env: KuhnPokerEnv) -> int:
        pass


class RandomAgent(BaseAgent):
    """an agent that chooses actions randomly"""
    def act(self, env: KuhnPokerEnv) -> int:
        return random.choice(env.get_legal_actions())


class CFRAgent(BaseAgent):
    """
    An agent that plays according to a pre-computed CFR strategy.

    Dynamically selects the correct policy (3-card or 4-card) based on the environment it is playing in.
    It loads the official tournament strategy files.
    """

    def __init__(self, base_path: str = ''):
        """
        Initializes the agent by loading both 3-card and 4-card policies.
        :param base_path: the relative path from the project root to the strategies directory.
        """
        # paths to the strategy files
        path_3_card = os.path.join(base_path, 'strategies', 'MVP_3_card_strategy.json')
        path_4_card = os.path.join(base_path, 'strategies', 'MVP_4_card_strategy.json')

        try:
            with open(path_3_card, 'r') as f:
                self.policy_3_card = json.load(f)
            with open(path_4_card, 'r') as f:
                self.policy_4_card = json.load(f)
        except FileNotFoundError as e:
            print(f"error: could not find strategy file. ensure you are running from the project root.")
            print(f"details: {e}")
            raise

        # card mappings
        self.card_map_3 = {0: 'J', 1: 'Q', 2: 'K'}
        self.card_map_4 = {0: 'J', 1: 'Q', 2: 'K', 3: 'A'}

    def act(self, env: KuhnPokerEnv) -> int:
        """
        Chooses an action by looking up the info set in the correct policy.

        :param env: the poker env instance, used to get game state.
        :returns: the chosen action (bet or pass).
        """
        # select policy and card map based on the environment
        if env.num_cards == 3:
            policy = self.policy_3_card
            card_map = self.card_map_3

        elif env.num_cards == 4:
            policy = self.policy_4_card
            card_map = self.card_map_4
        else:
            # fallback for unsupported number of cards, should never happen
            return random.choice(env.get_legal_actions())

        # get current player card and map it to its character representation
        card_int = env.cards[env.current_player]
        card_char = card_map.get(card_int)

        # construct the info set key in the format CARD_HISTORY (e.g., K_pb)
        info_set_key = f"{card_char}_{env.history}"

        # look up the action probabilities from the policy
        action_probs = policy.get(info_set_key, {'p': 1.0, 'b': 0.0})
        bet_prob = action_probs.get('b', 0.0)

        # choose an action based on the probability
        return KuhnPokerEnv.BET if random.random() < bet_prob else KuhnPokerEnv.PASS
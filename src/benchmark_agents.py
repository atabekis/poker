# src/benchmark_agents.py

import random
from abc import ABC, abstractmethod

# import KuhnPokerEnv from the poker_env directory
from poker_env.kuhn_env import KuhnPokerEnv

# --- agent base class ---
class BaseAgent(ABC):
    """abstract base class for all simple agents."""
    @abstractmethod
    def act(self, env: KuhnPokerEnv) -> int:
        pass

# --- agent implementations ---
class RandomAgent(BaseAgent):
    """an agent that chooses actions uniformly at random."""
    def act(self, env: KuhnPokerEnv) -> int:
        return random.choice(env.get_legal_actions())

class CFRAgent(BaseAgent):
    """an agent that plays according to a pre-computed cfr strategy."""
    CFR_STRATEGY = {
        "J": 0.22471, "Jb": 0.00003, "Jp": 0.33811, "Jpb": 0.00002,
        "K": 0.65545, "Kb": 0.99997, "Kp": 0.99988, "Kpb": 0.99996,
        "Q": 0.00014, "Qb": 0.33643, "Qp": 0.00023, "Qpb": 0.56420,
    }

    def __init__(self):
        self.strategy = self.CFR_STRATEGY
        self.card_map = {0: 'J', 1: 'Q', 2: 'K'}

    def act(self, env: KuhnPokerEnv) -> int:
        card_int = env.cards[env.current_player]
        card_char = self.card_map.get(card_int)
        if not card_char: return random.choice(env.get_legal_actions())
        key = card_char + env.history
        bet_prob = self.strategy.get(key, 0.0)
        return KuhnPokerEnv.BET if random.random() < bet_prob else KuhnPokerEnv.PASS


class MCCFRAgent(BaseAgent):
    """an agent that plays according to a pre-computed mccfr strategy."""
    MCCFR_STRATEGY = {
        "J": 0.195, "Jp": 0.32, "Jb": 0.03, "Jpb": 0.03,
        "Q": 0.03, "Qp": 0.05, "Qb": 0.44, "Qpb": 0.53,
        "K": 0.578, "Kp": 0.97, "Kb": 0.97, "Kpb": 0.97,
    }
    def __init__(self):
        self.strategy = self.MCCFR_STRATEGY
        self.card_map = {0: 'J', 1: 'Q', 2: 'K'}
    def act(self, env: KuhnPokerEnv) -> int:
        card_int = env.cards[env.current_player]
        card_char = self.card_map.get(card_int)
        if not card_char: return random.choice(env.get_legal_actions())
        key = card_char + env.history
        bet_prob = self.strategy.get(key, 0.0)
        return KuhnPokerEnv.BET if random.random() < bet_prob else KuhnPokerEnv.PASS
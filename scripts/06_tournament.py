# scripts/06_eval_with_game.py

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from itertools import combinations
from tqdm import tqdm

# add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poker_env.kuhn_env import KuhnPokerEnv
from src.models import AverageStrategyNet
from src.utils import load_checkpoint


# --- agent classes and implementations ---

class BaseAgent(ABC):
    """abstract base class for all agents."""

    @abstractmethod
    def act(self, env: KuhnPokerEnv) -> int:
        pass


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
    # keys have been standardized from the source to match the cfr agent's format
    MCCFR_STRATEGY = {
        # jack (card "1")
        "J": 0.195, "Jp": 0.32, "Jb": 0.03, "Jpb": 0.03,
        # queen (card "2")
        "Q": 0.03, "Qp": 0.05, "Qb": 0.44, "Qpb": 0.53,
        # king (card "3")
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


class NFSPInferenceAgent(BaseAgent):
    """a wrapper for our trained nfsp agents for inference."""

    def __init__(self, agent_data: dict, state_dim: int, num_actions: int, device):
        avg_strategy_state_dict = agent_data['weights']['as_net']
        hidden_size = avg_strategy_state_dict['network.0.weight'].shape[0]
        self.avg_strategy_net = AverageStrategyNet(state_dim, num_actions, hidden_size).to(device)
        self.avg_strategy_net.load_state_dict(avg_strategy_state_dict)
        self.avg_strategy_net.eval()
        self.device = device

    def act(self, env: KuhnPokerEnv) -> int:
        obs = env._get_observation()
        state_tensor = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            logits = self.avg_strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)
            action = torch.multinomial(probs, num_samples=1).item()
        return action


def run_hand_evaluation_match(agent1, agent2, num_matches, env):
    """runs a head-to-head match based on hand win rate (for champion selection)."""
    agent1_wins = 0
    for i in range(num_matches):
        players = [agent1, agent2] if i % 2 == 0 else [agent2, agent1]
        agent1_player_id = 0 if i % 2 == 0 else 1
        env.reset()
        done = False
        while not done:
            action = players[env.current_player].act(env)
            _, reward, done, _ = env.step(action)
        final_rewards = [0, 0]
        final_rewards[env.current_player] = reward
        final_rewards[1 - env.current_player] = -reward
        if final_rewards[agent1_player_id] > 0:
            agent1_wins += 1
    return agent1_wins / num_matches


# --- tournament match simulation class ---

class TournamentMatch:
    """
    manages a full tournament match between two agents, playing hands
    until one's bankroll is depleted.
    """

    def __init__(self, agent1: BaseAgent, agent2: BaseAgent, starting_bankroll: int, env: KuhnPokerEnv):
        """
        :param agent1: the first agent object.
        :param agent2: the second agent object.
        :param starting_bankroll: the amount of money each agent starts with.
        :param env: the kuhn poker environment instance.
        """
        self.players = [agent1, agent2]
        self.bankrolls = [starting_bankroll, starting_bankroll]
        self.env = env
        self.dealer_button = random.choice([0, 1])

    def _play_hand(self):
        """simulates one hand of poker and updates bankrolls."""
        self.dealer_button = 1 - self.dealer_button
        player0 = self.players[self.dealer_button]
        player1 = self.players[1 - self.dealer_button]
        hand_players = [player0, player1]

        self.bankrolls[self.dealer_button] -= 1
        self.bankrolls[1 - self.dealer_button] -= 1

        self.env.reset()
        done = False
        while not done:
            current_player_id = self.env.current_player
            acting_agent = hand_players[current_player_id]
            action = acting_agent.act(self.env)
            _, reward, done, _ = self.env.step(action)

        last_player_id = self.env.current_player

        hand_rewards = [0, 0]
        hand_rewards[last_player_id] = reward
        hand_rewards[1 - last_player_id] = -reward

        self.bankrolls[self.dealer_button] += hand_rewards[0]
        self.bankrolls[1 - self.dealer_button] += hand_rewards[1]

    def run_match(self) -> int:
        """
        runs the match until one player is bankrupt.

        :returns: the index of the winning agent (0 for agent1, 1 for agent2).
        """
        while True:
            if self.bankrolls[0] <= 0:
                return 1
            if self.bankrolls[1] <= 0:
                return 0

            self._play_hand()


# --- main orchestration logic ---

def main(args):
    """main function to load a population, find a champion, and run tournament simulations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_data = load_checkpoint(args.checkpoint_path)
    if not checkpoint_data: return
    population, _ = checkpoint_data

    env_cards = 3
    env = KuhnPokerEnv(num_cards=env_cards)

    nfsp_agents = [NFSPInferenceAgent(p, env.state_dim, env.NUM_ACTIONS, device) for p in population]

    print("\n" + "=" * 50)
    print("starting internal tournament to find champion...")
    total_wins = np.zeros(len(nfsp_agents))
    for i, j in tqdm(list(combinations(range(len(nfsp_agents)), 2)), desc="internal tournament"):
        win_rate_i = run_hand_evaluation_match(nfsp_agents[i], nfsp_agents[j], args.internal_matches, env)
        total_wins[i] += win_rate_i * args.internal_matches
        total_wins[j] += (1 - win_rate_i) * args.internal_matches

    champion_index = np.argmax(total_wins)
    champion_agent = nfsp_agents[champion_index]
    champion_id = population[champion_index]['id']
    print(f"internal tournament complete. champion is agent id: {champion_id}")

    benchmarks = {
        "RandomAgent": RandomAgent(),
        "CFRAgent": CFRAgent(),
        "MCCFRAgent": MCCFRAgent(),  # <-- added mccfr agent to the benchmarks
    }
    results = {}

    print("\n" + "=" * 50)
    print(f"evaluating champion (id: {champion_id}) in tournament simulations...")
    print(f"running {args.num_matches} full matches per benchmark with a ${args.bankroll} bankroll.")

    for name, benchmark_agent in benchmarks.items():
        champion_match_wins = 0
        for _ in tqdm(range(args.num_matches), desc=f"vs {name}"):
            match = TournamentMatch(champion_agent, benchmark_agent, args.bankroll, env)
            winner_idx = match.run_match()
            if winner_idx == 0:
                champion_match_wins += 1

        results[name] = champion_match_wins / args.num_matches

    print("\n" + "=" * 55)
    print(f"final tournament simulation results for champion (id: {champion_id})")
    print("=" * 55)
    print(f"{'opponent':<20} | {'champion match win rate':>25}")
    print("-" * 55)
    for name, wr in results.items():
        print(f"{name:<20} | {wr:>24.2%}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run tournament simulations for a trained pbt population.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="path to the .pth checkpoint file.")
    parser.add_argument("--num-matches", type=int, default=1000,
                        help="number of full tournament matches to run against each benchmark.")
    parser.add_argument("--internal-matches", type=int, default=1000,
                        help="number of hands for the internal champion-selection tournament.")
    parser.add_argument("--bankroll", type=int, default=5, help="starting bankroll for each tournament match.")

    parsed_args = parser.parse_args()
    main(parsed_args)
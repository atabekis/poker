# src/evaluation.py

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
from tqdm import tqdm

# add project root to python path if this file is run standalone
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poker_env.kuhn_env import KuhnPokerEnv
from src.models import AverageStrategyNet
from src.utils import load_checkpoint
# modified: import all benchmark agents, including mccfr
from src.benchmark_agents import BaseAgent, RandomAgent, CFRAgent, MCCFRAgent


# --- agent and match classes (no changes needed here) ---

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
            return torch.multinomial(probs, num_samples=1).item()


class TournamentMatch:
    """manages a full tournament match between two agents."""

    def __init__(self, agent1: BaseAgent, agent2: BaseAgent, env: KuhnPokerEnv):
        self.players = [agent1, agent2]
        self.env = env

    def _play_hand(self):
        self.env.reset_hand()
        hand_done = False
        while not hand_done:
            acting_agent = self.players[self.env.current_player]
            action = acting_agent.act(self.env)
            _, _, hand_done, _ = self.env.step(action)

    def run_match(self) -> int:
        self.env.reset_match()
        while True:
            if self.env.bankrolls[1] <= 0: return 0
            if self.env.bankrolls[0] <= 0: return 1
            self._play_hand()


def run_hand_evaluation_match(agent1, agent2, num_matches, env):
    """runs a head-to-head match based on hand win rate."""
    agent1_wins = 0
    for i in range(num_matches):
        players = [agent1, agent2] if i % 2 == 0 else [agent2, agent1]
        agent1_player_id = 0 if i % 2 == 0 else 1
        env.reset_hand()
        hand_done = False
        while not hand_done:
            action = players[env.current_player].act(env)
            _, reward, hand_done, _ = env.step(action)
        final_rewards = [0, 0];
        final_rewards[env.current_player] = reward;
        final_rewards[1 - env.current_player] = -reward
        if final_rewards[agent1_player_id] > 0: agent1_wins += 1
    return agent1_wins / num_matches


# --- main reusable evaluation function ---

def run_tournament_evaluation(checkpoint_path: str, eval_config: dict):
    """
    loads a population, finds a champion, and runs tournament simulations against benchmarks.

    :param checkpoint_path: path to the .pth checkpoint file.
    :param eval_config: a dictionary with evaluation parameters (num_matches, bankroll, etc.).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_data = load_checkpoint(checkpoint_path)
    if not checkpoint_data: return
    population, gen = checkpoint_data

    env = KuhnPokerEnv(num_cards=3, starting_bankroll=eval_config['bankroll'])
    nfsp_agents = [NFSPInferenceAgent(p, env.state_dim, env.NUM_ACTIONS, device) for p in population]

    print(f"\n--- starting evaluation for checkpoint: gen {gen} ---")
    total_wins = np.zeros(len(nfsp_agents))
    for i, j in tqdm(list(combinations(range(len(nfsp_agents)), 2)), desc="  internal tournament", leave=False):
        win_rate_i = run_hand_evaluation_match(nfsp_agents[i], nfsp_agents[j], eval_config['internal_matches'], env)
        total_wins[i] += win_rate_i * eval_config['internal_matches']
        total_wins[j] += (1 - win_rate_i) * eval_config['internal_matches']

    champion_index = np.argmax(total_wins)
    champion_agent = nfsp_agents[champion_index]
    champion_id = population[champion_index]['id']
    print(f"  -> internal tournament complete. champion is agent id: {champion_id}")

    # modified: add mccfragent to the dictionary of benchmarks
    benchmarks = {
        "RandomAgent": RandomAgent(),
        "CFRAgent": CFRAgent(),
        "MCCFRAgent": MCCFRAgent()
    }
    results = {}

    for name, benchmark_agent in benchmarks.items():
        champion_match_wins = 0
        for _ in tqdm(range(eval_config['num_matches']), desc=f"  vs {name}", leave=False):
            match = TournamentMatch(champion_agent, benchmark_agent, env)
            winner_idx = match.run_match()
            if winner_idx == 0: champion_match_wins += 1
        results[name] = champion_match_wins / eval_config['num_matches']

    print("\n" + "=" * 60)
    print(f"  evaluation results for champion (id: {champion_id}) from gen {gen}")
    print("=" * 60)
    print(f"{'opponent':<20} | {'champion match win rate':>25}")
    print("-" * 60)
    for name, wr in results.items():
        print(f"{name:<20} | {wr:>24.2%}")
    print("=" * 60)
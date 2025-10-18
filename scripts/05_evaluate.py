# scripts/05_evaluate.py

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

# --- strategy dictionaries for benchmark agents ---

CFR_STRATEGY = {
    "J": 0.22471, "Jb": 0.00003, "Jp": 0.33811, "Jpb": 0.00002,
    "K": 0.65545, "Kb": 0.99997, "Kp": 0.99988, "Kpb": 0.99996,
    "Q": 0.00014, "Qb": 0.33643, "Qp": 0.00023, "Qpb": 0.56420,
}

MCCFR_STRATEGY = {
    "1": 0.195, "1C": 0.32, "1B": 0.03, "1CB": 0.03,
    "2": 0.03, "2C": 0.05, "2B": 0.44, "2CB": 0.53,
    "3": 0.578, "3C": 0.97, "3B": 0.97, "3CB": 0.97,
}


# --- agent base classes and implementations ---

class BaseAgent(ABC):
    """abstract base class for all agents."""

    @abstractmethod
    def act(self, env: KuhnPokerEnv) -> int:
        pass


class RandomAgent(BaseAgent):
    """an agent that chooses actions uniformly at random."""

    def act(self, env: KuhnPokerEnv) -> int:
        return random.choice(env.get_legal_actions())


class PassAgent(BaseAgent):
    """an agent that always passes (or folds)."""

    def act(self, env: KuhnPokerEnv) -> int:
        return KuhnPokerEnv.PASS


class CFRAgent(BaseAgent):
    """an agent that plays according to a pre-computed cfr strategy."""

    def __init__(self, strategy_dict: dict):
        self.strategy = strategy_dict
        self.card_map = {0: 'J', 1: 'Q', 2: 'K'}

    def act(self, env: KuhnPokerEnv) -> int:
        card_int = env.cards[env.current_player]
        card_char = self.card_map.get(card_int)

        if not card_char:
            raise ValueError(f"card '{card_int}' not in map. this cfr agent only supports 3-card poker.")

        key = card_char + env.history
        bet_prob = self.strategy.get(key, 0.0)

        return KuhnPokerEnv.BET if random.random() < bet_prob else KuhnPokerEnv.PASS


class MCCFRAgent(BaseAgent):
    """an agent that plays according to the transcribed mccfr strategy."""

    def __init__(self, strategy_dict: dict):
        self.strategy = strategy_dict

    def act(self, env: KuhnPokerEnv) -> int:
        card_int = env.cards[env.current_player]
        history = env.history
        card_key = str(card_int + 1)

        history_suffix = ""
        if history == "p":
            history_suffix = "C"
        elif history == "b":
            history_suffix = "B"
        elif history == "pb":
            history_suffix = "CB"

        key = card_key + history_suffix
        bet_prob = self.strategy.get(key, 0.0)

        return KuhnPokerEnv.BET if random.random() < bet_prob else KuhnPokerEnv.PASS


class NFSPInferenceAgent(BaseAgent):
    """a wrapper for our trained nfsp agents for inference."""

    def __init__(self, agent_data: dict, state_dim: int, num_actions: int, device):
        """
        initializes the inference agent from a population data dictionary.

        :param agent_data: the dictionary for one agent from the loaded population list.
        :param state_dim: the dimension of the state space.
        :param num_actions: the number of possible actions.
        :param device: the torch device to use (cpu or cuda).
        """
        avg_strategy_state_dict = agent_data['weights']['as_net']

        # definitive fix: deduce hidden_size by inspecting the weight tensor shapes.
        # the shape of the first layer's weight is [hidden_size, input_size].
        # this is robust and does not depend on the config file.
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


# --- evaluation function ---

def run_evaluation_match(agent1: BaseAgent, agent2: BaseAgent, num_matches: int, env: KuhnPokerEnv) -> float:
    """
    runs a head-to-head match between two agents.

    :param agent1: the agent for whom we want the win rate.
    :param agent2: the opponent agent.
    :param num_matches: the number of hands to simulate.
    :param env: the game environment instance.
    :returns: the win rate of agent1 against agent2.
    """
    agent1_wins = 0
    for i in range(num_matches):
        if i % 2 == 0:
            players = [agent1, agent2]
            agent1_player_id = 0
        else:
            players = [agent2, agent1]
            agent1_player_id = 1

        env.reset()
        done = False
        while not done:
            current_player_id = env.current_player
            acting_agent = players[current_player_id]
            action = acting_agent.act(env)
            _, reward, done, _ = env.step(action)

        last_player_id = env.current_player
        final_rewards = [0, 0]
        final_rewards[last_player_id] = reward
        final_rewards[1 - last_player_id] = -reward

        if final_rewards[agent1_player_id] > 0:
            agent1_wins += 1

    return agent1_wins / num_matches


# --- main orchestration logic ---

def main(args):
    """main function to load a population, find a champion, and evaluate it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_data = load_checkpoint(args.checkpoint_path)
    if not checkpoint_data:
        print(f"could not load checkpoint at {args.checkpoint_path}")
        return
    population, _ = checkpoint_data

    env = KuhnPokerEnv(num_cards=3)

    nfsp_agents = []
    for p_data in population:
        agent = NFSPInferenceAgent(
            p_data,  # pass the entire agent data dictionary
            env.state_dim,
            env.NUM_ACTIONS,
            device
        )
        nfsp_agents.append(agent)

    print("\n" + "=" * 50)
    print("starting internal tournament to find champion...")
    print(f"evaluating {len(nfsp_agents)} agents over {args.internal_matches} matches per pairing.")

    total_wins = np.zeros(len(nfsp_agents))
    agent_indices = list(range(len(nfsp_agents)))

    for i, j in tqdm(list(combinations(agent_indices, 2)), desc="internal tournament"):
        agent_i = nfsp_agents[i]
        agent_j = nfsp_agents[j]
        win_rate_i = run_evaluation_match(agent_i, agent_j, args.internal_matches, env)
        total_wins[i] += win_rate_i * args.internal_matches
        total_wins[j] += (1 - win_rate_i) * args.internal_matches

    champion_index = np.argmax(total_wins)
    champion_agent = nfsp_agents[champion_index]
    champion_id = population[champion_index]['id']
    champion_config = population[champion_index]['config']

    print("internal tournament complete.")
    print(f"  -> champion is agent id: {champion_id} with config: {champion_config}")

    benchmarks = {
        "RandomAgent": RandomAgent(),
        "PassAgent": PassAgent(),
        "CFRAgent": CFRAgent(CFR_STRATEGY),
        "MCCFRAgent": MCCFRAgent(MCCFR_STRATEGY)
    }

    results = {}

    print("\n" + "=" * 50)
    print(f"evaluating champion (id: {champion_id}) against benchmarks...")
    print(f"running {args.num_matches} matches per benchmark.")

    for name, benchmark_agent in tqdm(benchmarks.items(), desc="benchmark evaluation"):
        win_rate = run_evaluation_match(champion_agent, benchmark_agent, args.num_matches, env)
        results[name] = win_rate

    print("\n" + "=" * 55)
    print(f"final evaluation results for champion (id: {champion_id})")
    print("=" * 55)
    print(f"{'opponent':<20} | {'champion win rate':>20}")
    print("-" * 55)
    for name, wr in results.items():
        print(f"{name:<20} | {wr:>19.2%}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate a trained pbt population.")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="path to the .pth checkpoint file from pbt training.")
    parser.add_argument("--num-matches", type=int, default=10000,
                        help="number of matches to run for each benchmark evaluation.")
    parser.add_argument("--internal-matches", type=int, default=1000,
                        help="number of matches for the internal round-robin tournament.")

    parsed_args = parser.parse_args()
    main(parsed_args)
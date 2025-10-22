# rl_agent/training/evaluation.py

"""
Evaluation script for trained NFSP agents.

This module provides the necessary tools to evaluate a population of trained agents.
It includes:
1.  An inference-specific agent wrapper that uses the learned average strategy for decision-making
2.  Functions to run head-to-head matches between two agents
3.  A main tournament function that:
    - Loads a population of agents from a training checkpoint
    - Conducts a round-robin tournament to identify the strongest agent (the champion)
    - Pits the champion against a set of predefined benchmark agents (e.g., Random, CFR).

The evaluation is based on hand win rate in the game.
"""

# Python imports
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from itertools import combinations

# Local imports
from .kuhn_env import KuhnPokerEnv
from ..agent_utils import AverageStrategyNet
from .utils import load_checkpoint
from .benchmark_agents import BaseAgent, RandomAgent, CFRAgent


class NFSPInferenceAgent(BaseAgent):
    """
    A wrapper for a trained NFSP agent for inference during evaluation. This class loads the weights from a training
    checkpoint and uses it to make decisions. The agent acts stochastically by sampling from the probability
    distribution output by the average strategy network.
    """
    def __init__(self, agent_data: dict, state_dim: int, num_actions: int, device):
        avg_strategy_state_dict = agent_data['weights']['as_net']
        hidden_size = agent_data['config']['hidden_size']
        self.avg_strategy_net = AverageStrategyNet(state_dim, num_actions, hidden_size).to(device)
        self.avg_strategy_net.load_state_dict(avg_strategy_state_dict)
        self.avg_strategy_net.eval()
        self.device = device

    def act(self, env: KuhnPokerEnv) -> int:
        """Select an action based on the learned average strategy"""
        obs = env._get_observation()
        state_tensor = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            logits = self.avg_strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)
            return torch.multinomial(probs, num_samples=1).item()


def run_hand_evaluation_match(agent1, agent2, num_hands, env):
    """
    Runs a match btw. two agents and returns the win rate for agent1.
    :param agent1: the first agent
    :param agent2: the second agent
    :param num_hands: total number of hands to play
    :param env: game environment
    :return: fraction of hands won by agent1
    """

    agent1_wins = 0
    for i in range(num_hands):
        players = [agent1, agent2] if i % 2 == 0 else [agent2, agent1] # alternate who starts for each hand
        agent1_player_id = 0 if i % 2 == 0 else 1

        env.reset_hand()
        hand_done = False
        reward = 0
        while not hand_done: # poker playing starts here
            action = players[env.current_player].act(env)
            _, reward, hand_done, _ = env.step(action)

        final_rewards = [0, 0]
        final_rewards[env.current_player] = reward
        final_rewards[1 - env.current_player] = -reward

        if final_rewards[agent1_player_id] > 0: # add win if agent1 has positive reward
            agent1_wins += 1

    return agent1_wins / num_hands


def run_tournament_evaluation(checkpoint_path: str, eval_config: dict, env_params: dict):
    """
    Loads a population, finds a champion, and runs tournament simulations.

    :param checkpoint_path: path to the .pth checkpoint file.
    :param eval_config: dictionary with evaluation parameters.
    :param env_params: dictionary with environment parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_data = load_checkpoint(checkpoint_path)
    if not checkpoint_data: return
    population, gen = checkpoint_data

    env = KuhnPokerEnv(**env_params) # pass all of the config params to the env
    nfsp_agents = [NFSPInferenceAgent(p, env.state_dim, env.NUM_ACTIONS, device) for p in population] # create agents

    print(f"--- starting evaluation for checkpoint: gen {gen} ---")
    total_wins = np.zeros(len(nfsp_agents))

    pairs = list(combinations(range(len(nfsp_agents)), 2))
    for i, j in tqdm(pairs, desc="  internal tournament", leave=False): # tqdm is buggy for HPC, but we will keep it
        win_rate_i = run_hand_evaluation_match( # run the previously defined function
            nfsp_agents[i],
            nfsp_agents[j],
            eval_config['internal_matches_per_pair'],
            env
        )
        total_wins[i] += win_rate_i * eval_config['internal_matches_per_pair'] # add the win rate to the total
        total_wins[j] += (1 - win_rate_i) * eval_config['internal_matches_per_pair']

    champion_index = np.argmax(total_wins)
    champion_agent = nfsp_agents[champion_index]
    champion_id = population[champion_index]['id']
    print(f"  -> internal tournament complete. champion is agent id: {champion_id}")

    benchmarks = {"RandomAgent": RandomAgent(), "CFRAgent": CFRAgent()}  # CRF agent knows whether to play 3 or 4 cards
    results = {}

    for name, benchmark_agent in benchmarks.items():
        win_rate = run_hand_evaluation_match(champion_agent, benchmark_agent, eval_config['benchmark_matches'], env)
        results[name] = win_rate

    # some ugly printing code...
    print('\n')
    print("=" * 60)
    print(f"  evaluation results for champion (id: {champion_id}) from gen {gen}")
    print("=" * 60)
    print(f"{'opponent':<20} | {'champion hand win rate':>25}")
    print("-" * 60)
    for name, wr in results.items():
        print(f"{name:<20} | {wr:>24.2%}")
    print("=" * 60)
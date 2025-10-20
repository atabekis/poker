# # scripts/06_tournament.py
#
# import os
# import sys
# import argparse
# import random
# import numpy as np
# import torch
# import torch.nn.functional as F
# from abc import ABC, abstractmethod
# from itertools import combinations
# from tqdm import tqdm
#
# # add project root to python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#
# from poker_env.kuhn_env import KuhnPokerEnv
# from src.models import AverageStrategyNet
# from src.utils import load_checkpoint
#
#
# # --- agent classes and implementations (no changes needed here) ---
#
# class BaseAgent(ABC):
#     """abstract base class for all agents."""
#
#     @abstractmethod
#     def act(self, env: KuhnPokerEnv) -> int:
#         pass
#
#
# class RandomAgent(BaseAgent):
#     """an agent that chooses actions uniformly at random."""
#
#     def act(self, env: KuhnPokerEnv) -> int:
#         return random.choice(env.get_legal_actions())
#
#
# class CFRAgent(BaseAgent):
#     """an agent that plays according to a pre-computed cfr strategy."""
#     CFR_STRATEGY = {
#         "J": 0.22471, "Jb": 0.00003, "Jp": 0.33811, "Jpb": 0.00002,
#         "K": 0.65545, "Kb": 0.99997, "Kp": 0.99988, "Kpb": 0.99996,
#         "Q": 0.00014, "Qb": 0.33643, "Qp": 0.00023, "Qpb": 0.56420,
#     }
#
#     def __init__(self):
#         self.strategy = self.CFR_STRATEGY
#         self.card_map = {0: 'J', 1: 'Q', 2: 'K'}
#
#     def act(self, env: KuhnPokerEnv) -> int:
#         card_int = env.cards[env.current_player]
#         card_char = self.card_map.get(card_int)
#         if not card_char: return random.choice(env.get_legal_actions())
#         key = card_char + env.history
#         bet_prob = self.strategy.get(key, 0.0)
#         return KuhnPokerEnv.BET if random.random() < bet_prob else KuhnPokerEnv.PASS
#
#
# class MCCFRAgent(BaseAgent):
#     """an agent that plays according to a pre-computed mccfr strategy."""
#     MCCFR_STRATEGY = {
#         "J": 0.195, "Jp": 0.32, "Jb": 0.03, "Jpb": 0.03,
#         "Q": 0.03, "Qp": 0.05, "Qb": 0.44, "Qpb": 0.53,
#         "K": 0.578, "Kp": 0.97, "Kb": 0.97, "Kpb": 0.97,
#     }
#
#     def __init__(self):
#         self.strategy = self.MCCFR_STRATEGY
#         self.card_map = {0: 'J', 1: 'Q', 2: 'K'}
#
#     def act(self, env: KuhnPokerEnv) -> int:
#         card_int = env.cards[env.current_player]
#         card_char = self.card_map.get(card_int)
#         if not card_char: return random.choice(env.get_legal_actions())
#         key = card_char + env.history
#         bet_prob = self.strategy.get(key, 0.0)
#         return KuhnPokerEnv.BET if random.random() < bet_prob else KuhnPokerEnv.PASS
#
#
# class NFSPInferenceAgent(BaseAgent):
#     """a wrapper for our trained nfsp agents for inference."""
#
#     def __init__(self, agent_data: dict, state_dim: int, num_actions: int, device):
#         avg_strategy_state_dict = agent_data['weights']['as_net']
#         hidden_size = avg_strategy_state_dict['network.0.weight'].shape[0]
#         self.avg_strategy_net = AverageStrategyNet(state_dim, num_actions, hidden_size).to(device)
#         self.avg_strategy_net.load_state_dict(avg_strategy_state_dict)
#         self.avg_strategy_net.eval()
#         self.device = device
#
#     def act(self, env: KuhnPokerEnv) -> int:
#         obs = env._get_observation()
#         state_tensor = torch.from_numpy(obs).float().to(self.device)
#         with torch.no_grad():
#             logits = self.avg_strategy_net(state_tensor)
#             probs = F.softmax(logits, dim=1)
#             action = torch.multinomial(probs, num_samples=1).item()
#         return action
#
#
# # --- evaluation functions (updated for new env api) ---
#
# def run_hand_evaluation_match(agent1, agent2, num_matches, env):
#     """runs a head-to-head match based on hand win rate (for champion selection)."""
#     agent1_wins = 0
#     for i in range(num_matches):
#         players = [agent1, agent2] if i % 2 == 0 else [agent2, agent1]
#         agent1_player_id = 0 if i % 2 == 0 else 1
#
#         # fix: use reset_hand for single-hand evaluations
#         env.reset_hand()
#         hand_done = False
#
#         while not hand_done:
#             action = players[env.current_player].act(env)
#             _, reward, hand_done, _ = env.step(action)
#
#         # reward is now delta_bankroll, so checking for > 0 is a valid way to count wins
#         final_rewards = [0, 0]
#         final_rewards[env.current_player] = reward
#         final_rewards[1 - env.current_player] = -reward
#
#         if final_rewards[agent1_player_id] > 0:
#             agent1_wins += 1
#
#     return agent1_wins / num_matches
#
#
# class TournamentMatch:
#     """
#     manages a full tournament match between two agents, now relying on the
#     environment to manage bankrolls and state.
#     """
#
#     def __init__(self, agent1: BaseAgent, agent2: BaseAgent, env: KuhnPokerEnv):
#         """
#         :param agent1: the first agent object.
#         :param agent2: the second agent object.
#         :param env: the stateful, match-aware kuhn poker environment instance.
#         """
#         self.players = [agent1, agent2]
#         self.env = env
#
#     def _play_hand(self):
#         """simulates one hand of poker. the env handles bankroll updates."""
#         self.env.reset_hand()
#         hand_done = False
#         while not hand_done:
#             # players are fixed as player 0 and player 1 for the match
#             acting_agent = self.players[self.env.current_player]
#             action = acting_agent.act(self.env)
#             _, _, hand_done, _ = self.env.step(action)
#
#     def run_match(self) -> int:
#         """
#         runs the match until one player is bankrupt.
#         :returns: the index of the winning agent (0 for agent1, 1 for agent2).
#         """
#         # reset the match state, including bankrolls, at the beginning
#         self.env.reset_match()
#
#         while True:
#             # check for a winner based on the environment's bankrolls
#             if self.env.bankrolls[1] <= 0:
#                 return 0  # agent1 wins
#             if self.env.bankrolls[0] <= 0:
#                 return 1  # agent2 wins
#
#             self._play_hand()
#
#
# # --- main orchestration logic (updated for new env api) ---
#
# def main(args):
#     """main function to load a population, find a champion, and run tournament simulations."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     checkpoint_data = load_checkpoint(args.checkpoint_path)
#     if not checkpoint_data: return
#     population, _ = checkpoint_data
#
#     # fix: initialize the environment with the tournament bankroll
#     env = KuhnPokerEnv(num_cards=3, starting_bankroll=args.bankroll)
#
#     nfsp_agents = [NFSPInferenceAgent(p, env.state_dim, env.NUM_ACTIONS, device) for p in population]
#
#     print("\n" + "=" * 50)
#     print("starting internal tournament to find champion...")
#     total_wins = np.zeros(len(nfsp_agents))
#     for i, j in tqdm(list(combinations(range(len(nfsp_agents)), 2)), desc="internal tournament"):
#         win_rate_i = run_hand_evaluation_match(nfsp_agents[i], nfsp_agents[j], args.internal_matches, env)
#         total_wins[i] += win_rate_i * args.internal_matches
#         total_wins[j] += (1 - win_rate_i) * args.internal_matches
#
#     champion_index = np.argmax(total_wins)
#     champion_agent = nfsp_agents[champion_index]
#     champion_id = population[champion_index]['id']
#     print(f"internal tournament complete. champion is agent id: {champion_id}")
#
#     benchmarks = {
#         "RandomAgent": RandomAgent(),
#         "CFRAgent": CFRAgent(),
#         "MCCFRAgent": MCCFRAgent(),
#     }
#     results = {}
#
#     print("\n" + "=" * 50)
#     print(f"evaluating champion (id: {champion_id}) in tournament simulations...")
#     print(f"running {args.num_matches} full matches per benchmark with a ${args.bankroll} bankroll.")
#
#     for name, benchmark_agent in benchmarks.items():
#         champion_match_wins = 0
#         for _ in tqdm(range(args.num_matches), desc=f"vs {name}"):
#             # fix: pass the already initialized env to the match
#             match = TournamentMatch(champion_agent, benchmark_agent, env)
#             winner_idx = match.run_match()
#             if winner_idx == 0:
#                 champion_match_wins += 1
#         results[name] = champion_match_wins / args.num_matches
#
#     print("\n" + "=" * 55)
#     print(f"final tournament simulation results for champion (id: {champion_id})")
#     print("=" * 55)
#     print(f"{'opponent':<20} | {'champion match win rate':>25}")
#     print("-" * 55)
#     for name, wr in results.items():
#         print(f"{name:<20} | {wr:>24.2%}")
#     print("=" * 55)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="run tournament simulations for a trained pbt population.")
#     parser.add_argument("--checkpoint-path", type=str, required=True, help="path to the .pth checkpoint file.")
#     parser.add_argument("--num-matches", type=int, default=1000,
#                         help="number of full tournament matches to run against each benchmark.")
#     parser.add_argument("--internal-matches", type=int, default=1000,
#                         help="number of hands for the internal champion-selection tournament.")
#     parser.add_argument("--bankroll", type=int, default=5, help="starting bankroll for each tournament match.")
#     parsed_args = parser.parse_args()
#     main(parsed_args)


# scripts/06_tournament.py

import os
import sys
import argparse
import yaml

# add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation import run_tournament_evaluation


def main(args):
    """main function to run a standalone tournament evaluation."""

    # create an eval_config dict from the command-line arguments
    eval_config = {
        'bankroll': args.bankroll,
        'num_matches': args.num_matches,
        'internal_matches': args.internal_matches
    }

    run_tournament_evaluation(args.checkpoint_path, eval_config)


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
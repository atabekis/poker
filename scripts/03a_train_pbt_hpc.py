# scripts/03_train_pbt_hpc.py

import os
import sys
import argparse
import random
import time
import yaml
import copy
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool, set_start_method

# add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poker_env.kuhn_env import KuhnPokerEnv
from src.agent import NFSPAgent
from src.models import AverageStrategyNet
from src.utils import sample_hyperparameters, mutate_hyperparameters, save_checkpoint, load_checkpoint
from src.benchmark_agents import RandomAgent, CFRAgent, BaseAgent
from src.evaluation import run_tournament_evaluation


class InferenceAgent:
    """lightweight opponent agent for nfsp population members."""

    def __init__(self, opponent_config: dict, avg_strategy_state_dict, state_dim, num_actions, device):
        hidden_size = avg_strategy_state_dict['network.0.weight'].shape[0]
        self.avg_strategy_net = AverageStrategyNet(state_dim, num_actions, hidden_size).to(device)
        self.avg_strategy_net.load_state_dict(avg_strategy_state_dict)
        self.avg_strategy_net.eval()
        self.device = device

    def act(self, obs: np.ndarray) -> int:
        """chooses an action based on a given observation numpy array."""
        state_tensor = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            logits = self.avg_strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)
            return torch.multinomial(probs, num_samples=1).item()


def worker(worker_id, agent_config, agent_weights, policy_store_path, results_store_path, pbt_params, env_params):
    """
    hpc-optimized worker with lazy loading of opponent policies.
    """
    random.seed(os.getpid() * int(time.time()) % (2 ** 32 - 1))
    np.random.seed(os.getpid() * int(time.time()) % (2 ** 32 - 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = KuhnPokerEnv(**env_params)

    full_config = agent_config.copy()
    full_config['state_dim'] = env.state_dim
    full_config['num_actions'] = env.NUM_ACTIONS
    full_config.setdefault('rl_buffer_size', 200000)
    full_config.setdefault('sl_buffer_size', 200000)
    agent = NFSPAgent(full_config)
    if agent_weights:
        agent.load_state_dicts(agent_weights)

    # --- lazy loading optimization: setup opponent cache and id list ---
    opponent_cache = {}  # cache for instantiated inferenceagent objects
    opponent_ids = []  # list of potential opponent ids
    if os.path.exists(policy_store_path):
        for policy_file in os.listdir(policy_store_path):
            if policy_file.startswith('policy_agent_') and policy_file.endswith('.pth'):
                agent_id_str = policy_file.replace('policy_agent_', '').replace('.pth', '')
                opponent_ids.append(int(agent_id_str))

    benchmark_opponents = [RandomAgent(), CFRAgent()]

    match_wins = 0
    matches_played = 0
    for _ in range(pbt_params['matches_per_generation']):
        # select opponent for the entire match
        if random.random() < pbt_params['benchmark_match_prob'] and benchmark_opponents:
            opponent = random.choice(benchmark_opponents)
        else:
            # --- lazy loading optimization: load opponent on demand ---
            if not opponent_ids:
                opponent = agent  # fallback to self-play if no opponents exist
            else:
                chosen_opponent_id = random.choice(opponent_ids)
                if chosen_opponent_id in opponent_cache:
                    opponent = opponent_cache[chosen_opponent_id]
                else:
                    # opponent not in cache, load it from disk
                    policy_path = os.path.join(policy_store_path, f"policy_agent_{chosen_opponent_id}.pth")
                    opponent_data = torch.load(policy_path, map_location=device)
                    new_opponent = InferenceAgent(opponent_data['config'], opponent_data['weights'], env.state_dim,
                                                  env.NUM_ACTIONS, device)
                    opponent_cache[chosen_opponent_id] = new_opponent
                    opponent = new_opponent

        # the rest of the training loop is identical
        env.reset_match()
        is_agent_player_0 = random.choice([True, False])
        match_over = False
        while not match_over:
            obs = env.reset_hand()
            hand_done = False
            hand_transitions = []
            while not hand_done:
                current_player_id = env.current_player
                acting_agent_is_main = (is_agent_player_0 and current_player_id == 0) or (
                            not is_agent_player_0 and current_player_id == 1)
                if acting_agent_is_main:
                    action = agent.act(obs)
                else:
                    if isinstance(opponent, BaseAgent):
                        action = opponent.act(env)
                    else:
                        action = opponent.act(obs)
                next_obs, reward, hand_done, info = env.step(action)
                if acting_agent_is_main:
                    hand_transitions.append((obs, action, reward, next_obs, hand_done))
                obs = next_obs
                match_over = info.get('match_over', False)
            for s, a, r, s_next, d in hand_transitions:
                agent.rl_buffer.push(s, a, r, s_next, d)
                agent.sl_buffer.push(s, a)
            agent.learn()
        final_bankrolls = info['final_bankrolls']
        agent_bankroll = final_bankrolls[0] if is_agent_player_0 else final_bankrolls[1]
        if agent_bankroll > 0: match_wins += 1
        matches_played += 1

    match_win_rate = match_wins / matches_played if matches_played > 0 else 0.0

    final_weights = agent.get_state_dicts()
    result_data = {'id': worker_id, 'final_weights': final_weights, 'performance': match_win_rate}
    result_file_path = os.path.join(results_store_path, f"result_agent_{worker_id}.pth")
    torch.save(result_data, result_file_path)
    return


def main(args):
    """the main orchestrator for the pbt process, optimized for hpc."""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    os.makedirs(config['checkpointing']['output_dir'], exist_ok=True)

    policy_store_path = os.path.join(config['checkpointing']['output_dir'], "policy_store")
    results_store_path = os.path.join(config['checkpointing']['output_dir'], "results_store")
    os.makedirs(policy_store_path, exist_ok=True)
    os.makedirs(results_store_path, exist_ok=True)

    population = []
    start_generation = 0
    if args.resume_from:
        checkpoint = load_checkpoint(args.resume_from)
        if checkpoint:
            population, start_generation = checkpoint
            print(f"resuming training from generation {start_generation + 1}")
        else:
            return
    else:
        print("starting a new training run.")
        for i in range(config['pbt_params']['population_size']):
            population.append({
                'id': i, 'config': sample_hyperparameters(config['hyperparameter_space']),
                'weights': None, 'performance': 0.0,
            })

    num_generations = config['pbt_params']['generations']
    gen = start_generation

    try:
        for gen in range(start_generation, num_generations):
            print("\n" + "=" * 50 + f"\ngeneration {gen + 1}/{num_generations}\n" + "=" * 50)

            for f in os.listdir(policy_store_path): os.remove(os.path.join(policy_store_path, f))
            for p in population:
                if p['weights']:
                    policy_data = {'config': p['config'], 'weights': p['weights']['as_net']}
                    torch.save(policy_data, os.path.join(policy_store_path, f"policy_agent_{p['id']}.pth"))

            for f in os.listdir(results_store_path): os.remove(os.path.join(results_store_path, f))

            tasks = [(
                p['id'], p['config'], p['weights'],
                policy_store_path, results_store_path,
                config['pbt_params'], config['env_params']
            ) for p in population]

            with Pool() as pool:
                pool.starmap(worker, tasks)

            results = []
            for i in range(len(population)):
                result_file = os.path.join(results_store_path, f"result_agent_{i}.pth")
                results.append(torch.load(result_file))

            results_map = {res['id']: res for res in results}
            for p in population:
                p['weights'] = results_map[p['id']]['final_weights']
                p['performance'] = results_map[p['id']]['performance']

            population.sort(key=lambda x: x['performance'], reverse=True)
            best_perf = population[0]['performance'];
            worst_perf = population[-1]['performance']
            avg_perf = sum(p['performance'] for p in population) / len(population)
            print(
                f"generation summary | best perf: {best_perf:.2%} | avg perf: {avg_perf:.2%} | worst perf: {worst_perf:.2%}")

            num_to_replace = int(
                config['pbt_params']['population_size'] * (config['pbt_params']['exploit_bottom_k_percent'] / 100))
            if num_to_replace > 0:
                top_performers = population[:-num_to_replace];
                bottom_performers = population[-num_to_replace:]
                for bottom_agent in bottom_performers:
                    top_agent = random.choice(top_performers)
                    new_config_base = copy.deepcopy(top_agent['config'])
                    mutated_config = mutate_hyperparameters(new_config_base, config['hyperparameter_space'])
                    bottom_agent['config'] = mutated_config
                    top_hidden = top_agent['config'].get('hidden_size');
                    bottom_hidden = mutated_config.get('hidden_size')
                    if top_hidden != bottom_hidden:
                        bottom_agent['weights'] = None
                        print(
                            f"agent {bottom_agent['id']} replaced by {top_agent['id']}. architecture mutated ({top_hidden} -> {bottom_hidden}), re-initializing.")
                    else:
                        bottom_agent['weights'] = copy.deepcopy(top_agent['weights'])
                        print(f"agent {bottom_agent['id']} replaced by {top_agent['id']} and mutated.")

            should_save = (gen + 1) % config['checkpointing']['save_every_n_generations'] == 0
            if should_save:
                save_checkpoint(population, gen + 1, config['checkpointing']['output_dir'])

            should_evaluate = 'evaluation_params' in config and (gen + 1) % config['evaluation_params'][
                'evaluate_every'] == 0
            if should_evaluate:
                latest_checkpoint_path = os.path.join(config['checkpointing']['output_dir'],
                                                      f"checkpoint_gen_{gen + 1}.pth")
                if os.path.exists(latest_checkpoint_path):
                    try:
                        run_tournament_evaluation(latest_checkpoint_path, config['evaluation_params'])
                    except Exception as e:
                        print(f"error during evaluation: {e}, continuing.")
                else:
                    print(f"\nwarning: evaluation skipped for gen {gen + 1} as checkpoint file was not found.")
    except KeyboardInterrupt:
        print("\n\n" + "-" * 60 + "\nkeyboardinterrupt caught. attempting to save and exit gracefully.")
        if population and 'weights' in population[0] and population[0]['weights']:
            save_checkpoint(population, gen, config['checkpointing']['output_dir'])
            print(f"saved checkpoint for resuming at generation {gen}.")
        else:
            print("no population data with weights to save.")
        print("exiting.\n" + "-" * 60)
        sys.exit(0)

    print("\npbt training finished.")
    save_checkpoint(population, num_generations, config['checkpointing']['output_dir'])


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description="run hpc-optimized, match-aware pbt for nfsp.")
    parser.add_argument("--config", type=str, required=True, help="path to the pbt_config.yaml file.")
    parser.add_argument("--resume-from", type=str, help="path to a checkpoint file to resume training.")
    parsed_args = parser.parse_args()
    main(parsed_args)
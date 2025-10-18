# scripts/03_train_pbt.py

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


class InferenceAgent:
    """a lightweight opponent agent that only performs inference."""

    def __init__(self, avg_strategy_state_dict, state_dim, num_actions, device):
        self.avg_strategy_net = AverageStrategyNet(state_dim, num_actions).to(device)
        self.avg_strategy_net.load_state_dict(avg_strategy_state_dict)
        self.avg_strategy_net.eval()
        self.device = device

    def act(self, state: np.ndarray) -> int:
        state_tensor = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            logits = self.avg_strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)
            action = torch.multinomial(probs, num_samples=1).item()
        return action


def worker(worker_id, agent_config, agent_weights, opponent_policies, env_params, episodes_per_gen):
    """
    the function executed by each parallel process.
    trains and evaluates one agent for one generation.
    """
    # ensure different random seeds for each worker
    random.seed(os.getpid() * int(time.time()) % (2 ** 32 - 1))
    np.random.seed(os.getpid() * int(time.time()) % (2 ** 32 - 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup environment and agent
    env = KuhnPokerEnv(**env_params)

    # combine base config with specific agent config
    full_config = agent_config.copy()
    full_config['state_dim'] = env.state_dim
    full_config['num_actions'] = env.NUM_ACTIONS
    # add missing params from our test script config
    full_config.setdefault('epsilon_start', 1.0)
    full_config.setdefault('epsilon_end', 0.05)
    full_config.setdefault('rl_buffer_size', 100000)
    full_config.setdefault('sl_buffer_size', 100000)

    agent = NFSPAgent(full_config)
    if agent_weights:
        agent.load_state_dicts(agent_weights)

    # setup opponents
    opponents = []
    for policy_weights in opponent_policies:
        opponents.append(InferenceAgent(policy_weights, env.state_dim, env.NUM_ACTIONS, device))

    # training loop for one generation
    wins = 0
    games_played = 0

    for _ in range(episodes_per_gen):
        opponent = random.choice(opponents) if opponents else agent
        is_agent_player_0 = random.choice([True, False])

        obs = env.reset()
        done = False
        episode_transitions = []

        while not done:
            current_player_id = env.current_player

            if (is_agent_player_0 and current_player_id == 0) or \
                    (not is_agent_player_0 and current_player_id == 1):
                action = agent.act(obs)
                acting_agent_id = 0 if is_agent_player_0 else 1  # conceptual id for this game
            else:
                action = opponent.act(obs)
                acting_agent_id = 1 if is_agent_player_0 else 0

            next_obs, reward, done, _ = env.step(action)
            episode_transitions.append((obs, action, reward, next_obs, done, current_player_id))
            obs = next_obs

        # credit assignment
        final_reward_for_last_player = episode_transitions[-1][2]
        last_player_id = episode_transitions[-1][5]

        final_rewards = [0, 0]
        final_rewards[last_player_id] = final_reward_for_last_player
        final_rewards[1 - last_player_id] = -final_reward_for_last_player

        for trans in episode_transitions:
            s, a, _, s_next, d, player_id = trans

            if (is_agent_player_0 and player_id == 0) or \
                    (not is_agent_player_0 and player_id == 1):
                final_reward = final_rewards[player_id]
                agent.rl_buffer.push(s, a, final_reward, s_next, d)
                agent.sl_buffer.push(s, a)

        agent.learn()

        # track performance
        agent_final_reward = final_rewards[0] if is_agent_player_0 else final_rewards[1]
        if agent_final_reward > 0:
            wins += 1
        games_played += 1

    win_rate = wins / games_played if games_played > 0 else 0.0

    # move weights to cpu before returning
    final_weights = agent.get_state_dicts()
    for key in final_weights:
        final_weights[key] = {k: v.cpu() for k, v in final_weights[key].items()}

    return {
        'id': worker_id,
        'final_weights': final_weights,
        'performance': win_rate
    }


def main(args):
    """the main orchestrator for the pbt process."""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config['checkpointing']['output_dir'], exist_ok=True)

    population = []
    start_generation = 0

    if args.resume_from:
        checkpoint = load_checkpoint(args.resume_from)
        if checkpoint:
            population, start_generation = checkpoint
            print(f"resuming training from generation {start_generation + 1}")
        else:
            return  # exit if checkpoint not found
    else:
        print("starting a new training run.")
        for i in range(config['pbt_params']['population_size']):
            population.append({
                'id': i,
                'config': sample_hyperparameters(config['hyperparameter_space']),
                'weights': None,
                'performance': 0.0,
            })

    num_generations = config['pbt_params']['generations']
    gen = start_generation  # initialize gen for the except block

    try:
        for gen in range(start_generation, num_generations):
            print("\n" + "=" * 50)
            print(f"generation {gen + 1}/{num_generations}")
            print("=" * 50)

            # prepare tasks for workers
            opponent_policies = [p['weights']['as_net'] for p in population if p['weights']]
            tasks = [(
                p['id'],
                p['config'],
                p['weights'],
                opponent_policies,
                config['env_params'],
                config['pbt_params']['episodes_per_generation']
            ) for p in population]

            # run training in parallel
            with Pool() as pool:
                results = pool.starmap(worker, tasks)

            # update population with results
            results_map = {res['id']: res for res in results}
            for p in population:
                p['weights'] = results_map[p['id']]['final_weights']
                p['performance'] = results_map[p['id']]['performance']

            # evaluate and log
            population.sort(key=lambda x: x['performance'], reverse=True)
            best_perf = population[0]['performance']
            worst_perf = population[-1]['performance']
            avg_perf = sum(p['performance'] for p in population) / len(population)
            print(
                f"generation summary | best perf: {best_perf:.2%} | avg perf: {avg_perf:.2%} | worst perf: {worst_perf:.2%}")

            # exploit and explore
            num_to_replace = int(
                config['pbt_params']['population_size'] * (config['pbt_params']['exploit_bottom_k_percent'] / 100))
            if num_to_replace > 0:
                top_performers = population[:-num_to_replace]
                bottom_performers = population[-num_to_replace:]

                for bottom_agent in bottom_performers:
                    top_agent = random.choice(top_performers)
                    bottom_agent['weights'] = copy.deepcopy(top_agent['weights'])
                    bottom_agent['config'] = mutate_hyperparameters(top_agent['config'], config['hyperparameter_space'])
                    print(f"agent {bottom_agent['id']} replaced by {top_agent['id']} and mutated.")

            # checkpointing
            if (gen + 1) % config['checkpointing']['save_every_n_generations'] == 0:
                save_checkpoint(population, gen + 1, config['checkpointing']['output_dir'])

    except KeyboardInterrupt:
        print("\n\n" + "-" * 60)
        print("KeyboardInterrupt caught. attempting to save and exit gracefully.")
        if population:
            save_checkpoint(population, gen, config['checkpointing']['output_dir'])
            print(f"saved checkpoint for resuming at generation {gen + 1}.")
        else:
            print("no population data to save.")
        print("exiting.")
        print("-" * 60)
        sys.exit(0)

    print("pbt training finished.")
    save_checkpoint(population, num_generations, config['checkpointing']['output_dir'])  # final save


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="run population-based training for nfsp.")
    parser.add_argument("--config", type=str, required=True, help="path to the pbt_config.yaml file.")
    parser.add_argument("--resume-from", type=str, help="path to a checkpoint file to resume training.")

    parsed_args = parser.parse_args()
    main(parsed_args)
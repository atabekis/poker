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
from src.benchmark_agents import RandomAgent, CFRAgent, BaseAgent
from src.evaluation import run_tournament_evaluation


class InferenceAgent:
    """lightweight opponent agent for nfsp population members."""

    def __init__(self, opponent_config: dict, avg_strategy_state_dict, state_dim, num_actions, device):
        # deduce hidden_size from the weight tensor shape for robustness
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


def worker(worker_id, agent_config, agent_weights, opponent_infos, pbt_params, env_params):
    """
    the function executed by each parallel process.
    trains and evaluates one agent over full matches for one generation.
    """
    # ensure different random seeds for each worker process
    random.seed(os.getpid() * int(time.time()) % (2 ** 32 - 1))
    np.random.seed(os.getpid() * int(time.time()) % (2 ** 32 - 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = KuhnPokerEnv(**env_params)

    # setup main agent
    full_config = agent_config.copy()
    full_config['state_dim'] = env.state_dim
    full_config['num_actions'] = env.NUM_ACTIONS
    full_config.setdefault('rl_buffer_size', 200000)
    full_config.setdefault('sl_buffer_size', 200000)
    agent = NFSPAgent(full_config)
    if agent_weights:
        agent.load_state_dicts(agent_weights)

    # setup opponent pools
    population_opponents = [InferenceAgent(info['config'], info['weights'], env.state_dim, env.NUM_ACTIONS, device) for
                            info in opponent_infos]
    benchmark_opponents = [RandomAgent(), CFRAgent()]

    match_wins = 0
    matches_played = 0

    for _ in range(pbt_params['matches_per_generation']):
        # select opponent for the entire match
        if random.random() < pbt_params['benchmark_match_prob'] and benchmark_opponents:
            opponent = random.choice(benchmark_opponents)
        else:
            opponent = random.choice(population_opponents) if population_opponents else agent

        # start a new match
        env.reset_match()
        is_agent_player_0 = random.choice([True, False])

        match_over = False
        while not match_over:
            # play one hand
            obs = env.reset_hand()
            hand_done = False
            hand_transitions = []

            while not hand_done:
                current_player_id = env.current_player
                acting_agent_is_main = (is_agent_player_0 and current_player_id == 0) or \
                                       (not is_agent_player_0 and current_player_id == 1)

                if acting_agent_is_main:
                    action = agent.act(obs)
                else:
                    if isinstance(opponent, BaseAgent):  # handles benchmark agents
                        action = opponent.act(env)
                    else:  # handles inferenceagent and self-play
                        action = opponent.act(obs)

                next_obs, reward, hand_done, info = env.step(action)

                # store transition info relative to the main agent's actions
                if acting_agent_is_main:
                    hand_transitions.append((obs, action, reward, next_obs, hand_done))

                obs = next_obs
                match_over = info.get('match_over', False)

            # post-hand learning: attribute the final hand reward to all actions
            for s, a, r, s_next, d in hand_transitions:
                agent.rl_buffer.push(s, a, r, s_next, d)
                agent.sl_buffer.push(s, a)

            agent.learn()

        # after match, determine winner and update performance
        final_bankrolls = info['final_bankrolls']
        agent_bankroll = final_bankrolls[0] if is_agent_player_0 else final_bankrolls[1]
        if agent_bankroll > 0:
            match_wins += 1
        matches_played += 1

    match_win_rate = match_wins / matches_played if matches_played > 0 else 0.0

    # move weights to cpu before returning to the main process
    final_weights = agent.get_state_dicts()
    for key in final_weights:
        final_weights[key] = {k: v.cpu() for k, v in final_weights[key].items()}

    return {'id': worker_id, 'final_weights': final_weights, 'performance': match_win_rate}


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

            opponent_infos = [{'config': p['config'], 'weights': p['weights']['as_net']} for p in population if
                              p['weights']]
            tasks = [(p['id'], p['config'], p['weights'], opponent_infos, config['pbt_params'], config['env_params'])
                     for p in population]

            with Pool() as pool:
                results = pool.starmap(worker, tasks)

            results_map = {res['id']: res for res in results}
            for p in population:
                p['weights'] = results_map[p['id']]['final_weights']
                p['performance'] = results_map[p['id']]['performance']

            population.sort(key=lambda x: x['performance'], reverse=True)
            best_perf = population[0]['performance']
            worst_perf = population[-1]['performance']
            avg_perf = sum(p['performance'] for p in population) / len(population)
            print(
                f"generation summary | best perf: {best_perf:.2%} | avg perf: {avg_perf:.2%} | worst perf: {worst_perf:.2%}")

            num_to_replace = int(
                config['pbt_params']['population_size'] * (config['pbt_params']['exploit_bottom_k_percent'] / 100))
            if num_to_replace > 0:
                top_performers = population[:-num_to_replace]
                bottom_performers = population[-num_to_replace:]
                for bottom_agent in bottom_performers:
                    top_agent = random.choice(top_performers)
                    new_config_base = copy.deepcopy(top_agent['config'])
                    mutated_config = mutate_hyperparameters(new_config_base, config['hyperparameter_space'])
                    bottom_agent['config'] = mutated_config

                    top_hidden = top_agent['config'].get('hidden_size')
                    bottom_hidden = mutated_config.get('hidden_size')

                    if top_hidden != bottom_hidden:
                        bottom_agent['weights'] = None
                        print(
                            f"agent {bottom_agent['id']} replaced by {top_agent['id']}. architecture mutated ({top_hidden} -> {bottom_hidden}), re-initializing.")
                    else:
                        bottom_agent['weights'] = copy.deepcopy(top_agent['weights'])
                        print(f"agent {bottom_agent['id']} replaced by {top_agent['id']} and mutated.")

            # checkpointing
            should_save = (gen + 1) % config['checkpointing']['save_every_n_generations'] == 0
            if should_save:
                save_checkpoint(population, gen + 1, config['checkpointing']['output_dir'])

            # integrated evaluation
            should_evaluate = 'evaluation_params' in config and (gen + 1) % config['evaluation_params'][
                'evaluate_every'] == 0
            if should_evaluate:
                latest_checkpoint_path = os.path.join(config['checkpointing']['output_dir'],
                                                      f"checkpoint_gen_{gen + 1}.pth")
                if os.path.exists(latest_checkpoint_path):
                    try:
                        run_tournament_evaluation(latest_checkpoint_path, config['evaluation_params'])
                    except Exception as e:
                        print(f"error occurred during evaluation: {e}, continuing to next generation.")
                else:
                    print(
                        f"\nwarning: evaluation skipped for gen {gen + 1} as checkpoint file was not found (is save_every_n_generations a divisor of evaluate_every?).")
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
    parser = argparse.ArgumentParser(description="run match-aware pbt for nfsp.")
    parser.add_argument("--config", type=str, required=True, help="path to the pbt_config.yaml file.")
    parser.add_argument("--resume-from", type=str, help="path to a checkpoint file to resume training.")
    parsed_args = parser.parse_args()
    main(parsed_args)
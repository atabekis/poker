# rl_agent/training/train.py

"""
This module is the main orchestrator for our PBT (population-based training) approach.
It is designed for multi core machines or single HPC nodes to perform massively parallel training.
"""

# Python imports
import os
import sys
import time
import yaml
import copy
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
from multiprocessing import Pool, set_start_method

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..agent import NFSPAgent
from ..agent_utils import AverageStrategyNet
from .kuhn_env import KuhnPokerEnv
from .benchmark_agents import BaseAgent, RandomAgent, CFRAgent
from .evaluation import run_tournament_evaluation
from .utils import sample_hyperparameters, mutate_hyperparameters, save_checkpoint, load_checkpoint


class InferenceAgent:
    """Lightweight opponent agent for NFSP population members"""

    def __init__(self, avg_strategy_state_dict, state_dim, num_actions, hidden_size, device):
        self.avg_strategy_net = AverageStrategyNet(state_dim, num_actions, hidden_size).to(device)
        self.avg_strategy_net.load_state_dict(avg_strategy_state_dict)
        self.avg_strategy_net.eval()
        self.device = device

    def act(self, obs: np.ndarray) -> int:
        state_tensor = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            logits = self.avg_strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)
            return torch.multinomial(probs, num_samples=1).item()


def worker(worker_id, agent_config, agent_weights, opponent_infos, pbt_params, env_params):
    """
    The function executed by each parallel process for one generation
    trains and evaluates one agent by playing a series of full matches
    """
    # we should ensure different random seeds for each worker process
    random.seed(os.getpid() * int(time.time()) % (2 ** 32 - 1))
    np.random.seed(os.getpid() * int(time.time()) % (2 ** 32 - 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = KuhnPokerEnv(**env_params)

    # setup the main agent for this worker (each worker gets this)
    full_config = agent_config.copy()
    full_config['state_dim'] = env.state_dim
    full_config['num_actions'] = env.NUM_ACTIONS
    agent = NFSPAgent(full_config)
    if agent_weights:
        agent.load_state_dicts(agent_weights)

    # setup pools of opponents
    population_opponents = [
        InferenceAgent(info['weights'], env.state_dim, env.NUM_ACTIONS, info['config']['hidden_size'], device) for info
        in opponent_infos]
    benchmark_opponents = [RandomAgent(), CFRAgent()]

    match_wins, matches_played = 0, 0

    for _ in range(pbt_params['matches_per_generation']):
        # select an opponent for the entire match
        is_benchmark_match = random.random() < pbt_params['benchmark_match_prob'] and benchmark_opponents

        if is_benchmark_match:
            opponent = random.choice(benchmark_opponents)  # select from possible opponents
        else:
            # if population is empty (gen 1) or no benchmarks, self-play
            opponent = random.choice(population_opponents) if population_opponents else agent

        # --- match logic starts here
        env.reset_match()
        is_agent_player_0 = random.choice([True, False])
        agent_player_id = 0 if is_agent_player_0 else 1

        match_over = False
        while not match_over:
            obs = env.reset_hand()
            hand_done = False
            hand_transitions = []
            terminal_reward_for_last_actor = 0

            while not hand_done:
                current_player_id = env.current_player
                acting_agent_is_main = (agent_player_id == current_player_id)

                if acting_agent_is_main:
                    action_tuple = agent.act(obs)  # (int, str)
                    action_to_env = action_tuple[0]
                else:
                    # benchmark agents need the full env object, inference agents only need the obs
                    action_to_env = opponent.act(env if isinstance(opponent, BaseAgent) else obs)

                next_obs, reward, hand_done, info = env.step(action_to_env)  # next env state
                terminal_reward_for_last_actor = reward

                if acting_agent_is_main: # store transition with intermediate reward
                    hand_transitions.append((obs, action_tuple, reward, next_obs, hand_done)) # store the full action

                obs = next_obs
                match_over = info.get('match_over', False)

            # reward logic
            last_acting_player_id = env.current_player
            if last_acting_player_id == agent_player_id:
                # our agent acted last, so the terminal reward is its own.
                final_agent_reward = terminal_reward_for_last_actor
            else:
                # the opponent acted last, so our agent's reward is the inverse.
                final_agent_reward = -terminal_reward_for_last_actor

            # push transitions to the replay buffer
            for s, a_tuple, r_intermediate, s_next, d in hand_transitions:
                action_int, policy_str = a_tuple
                reward_to_store = final_agent_reward if d else 0

                # The rl buffer stores all transitions to learn the best response
                agent.rl_buffer.push(s, action_int, reward_to_store, s_next, d)

                # The sl buffer only stores transitions from the best response policy to learn the average strategy
                if policy_str == "br":
                    agent.sl_buffer.push(s, action_int)

            agent.learn()

        # score the match outcome
        agent_bankroll = info['final_bankrolls'][agent_player_id]
        if agent_bankroll > 0:
            match_wins += 1
        matches_played += 1

    win_rate = match_wins / matches_played if matches_played > 0 else 0.0

    # move weights to cpu before returning to the main process to avoid ipc issues
    final_weights = {k: {p: v.cpu() for p, v in w.items()} for k, w in agent.get_state_dicts().items()}

    return {'id': worker_id, 'final_weights': final_weights, 'performance': win_rate}


def main():
    """the main orchestrator for the pbt process"""
    parser = argparse.ArgumentParser(description="run pbt for nfsp.")
    parser.add_argument("--config", type=str, required=True, help="path to the training config.yaml file.")
    parser.add_argument("--resume-from", type=str, help="path to a checkpoint file to resume training.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    os.makedirs(config['checkpointing']['output_dir'], exist_ok=True)

    population, start_generation = [], 0
    if args.resume_from:
        checkpoint = load_checkpoint(args.resume_from)
        if checkpoint:
            population, start_generation = checkpoint
        else:
            return
    else:
        print(f"starting a new training run with {config['env_params']['num_cards']} cards")
        for i in range(config['pbt_params']['population_size']):
            population.append({
                'id': i, 'config': sample_hyperparameters(config['hyperparameter_space']),
                'weights': None, 'performance': 0.0,
            })

    num_generations = config['pbt_params']['generations']
    gen = start_generation

    try:
        for gen in range(start_generation, num_generations):
            # some more ugly printing to get table-ish results
            print(f"\n{'=' * 50}\ngeneration {gen + 1}/{num_generations}\n{'=' * 50}")

            # prepare tasks for the multiprocessing pool
            opponent_infos = [{'config': p['config'], 'weights': p['weights']['as_net']}
                              for p in population if p['weights']]

            tasks = [(p['id'], p['config'], p['weights'], opponent_infos, config['pbt_params'], config['env_params'])
                     for p in population]

            with Pool(config['pbt_params']['population_size']) as pool:  # map each of the tasks to a worker process
                results = pool.starmap(worker, tasks)  # this can be a major bottleneck if using ~40+ cores!

            # update population with results from workers
            for p in population:  # reduce
                res = next(r for r in results if r['id'] == p['id'])
                p['weights'], p['performance'] = res['final_weights'], res['performance']

            # log generation summary, sort all agents by performance, we get the best and worst agent performance
            population.sort(key=lambda x: x['performance'], reverse=True)
            best_perf = population[0]['performance']
            worst_perf = population[-1]['performance']
            avg_perf = sum(p['performance'] for p in population) / len(population)
            print(f"generation summary | best perf: {best_perf:.2%} | "
                  f"avg perf: {avg_perf:.2%} | worst perf: {worst_perf:.2%}")

            # exploit & explore step
            num_to_replace = int(config['pbt_params']['population_size']
                                 * (config['pbt_params']['exploit_bottom_k_percent'] / 100))

            if num_to_replace > 0:
                top_performers = population[:-num_to_replace]
                for p in population[-num_to_replace:]:
                    top_agent = random.choice(top_performers)
                    p['config'] = mutate_hyperparameters(top_agent['config'], config['hyperparameter_space'])

                    # if architecture changes (especially hidden_size),
                    # weights are incompatible and must be re-initialized
                    if p['config']['hidden_size'] != top_agent['config']['hidden_size']:
                        p['weights'] = None
                    else:
                        p['weights'] = copy.deepcopy(top_agent['weights'])

            # checkpointing
            if (gen + 1) % config['checkpointing']['save_every_n_generations'] == 0:
                save_checkpoint(population, gen + 1, config['checkpointing']['output_dir'])

            # integrated evaluation against benchmarks
            should_evaluate = (
                    'evaluation_params' in config and (gen + 1) % config['evaluation_params']['evaluate_every'] == 0
            )

            if should_evaluate:
                latest_checkpoint_path = os.path.join(
                    config['checkpointing']['output_dir'], f"checkpoint_gen_{gen + 1}.pth"
                )

                if os.path.exists(latest_checkpoint_path):
                    try:
                        run_tournament_evaluation(
                            latest_checkpoint_path, config['evaluation_params'], config['env_params']
                        )
                    except Exception as e:  # sometimes we have issues with saving,
                        # don't allow the error to crash training
                        print(f"error during evaluation: {e}, continuing.")
                else:
                    print(f"warning: evaluation skipped for gen {gen + 1} as checkpoint file was not found.")

    except KeyboardInterrupt:  # works half of the time
        print("\n\nkeyboardinterrupt caught. attempting to save final state and exit gracefully.")
        if population and 'weights' in population[0] and population[0]['weights']:
            save_checkpoint(population, gen, config['checkpointing']['output_dir'])

    print("\npbt training finished.")
    # we always save the final state of the population
    if population and 'weights' in population[0] and population[0]['weights']:
        save_checkpoint(population, num_generations, config['checkpointing']['output_dir'])


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()
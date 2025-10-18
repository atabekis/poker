# scripts/02_test_agent.py

import sys
import os
import argparse
from collections import deque
import matplotlib.pyplot as plt

# add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poker_env.kuhn_env import KuhnPokerEnv
from src.agent import NFSPAgent


def main(args):
    """main function to run the nfsp agent self-play test."""

    env = KuhnPokerEnv(num_cards=args.num_cards)

    config = {
        'state_dim': env.state_dim,
        'num_actions': env.NUM_ACTIONS,
        'rl_buffer_size': 1_000_000,
        'sl_buffer_size': 1_000_000,
        'batch_size': 128,
        'rl_learning_rate': 0.001,
        'sl_learning_rate': 0.001,
        'gamma': 0.99,
        'tau': 0.005,
        'gradient_clip': 1.0,
        'eta': 0.1,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 30000,
    }

    agents = [NFSPAgent(config), NFSPAgent(config)]

    # logging setup
    rl_losses = deque(maxlen=args.log_every)
    sl_losses = deque(maxlen=args.log_every)

    # history for plotting
    logged_episodes = []
    rl_loss_history = []
    sl_loss_history = []

    print("starting nfsp self-play test...")
    for episode in range(args.num_episodes):
        obs = env.reset()
        done = False

        # use a temporary list to store the episode's transitions
        episode_transitions = []

        while not done:
            current_player_id = env.current_player
            agent = agents[current_player_id]

            current_obs = obs
            action = agent.act(current_obs)
            next_obs, reward, done, _ = env.step(action)

            # store everything, including who acted and what the immediate reward was
            episode_transitions.append({
                "obs": current_obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "player_id": current_player_id
            })
            obs = next_obs

        # credit assignment logic
        # the final transition contains the terminal reward for the player who acted last
        final_reward_for_last_player = episode_transitions[-1]['reward']

        # the final reward for the other player is the inverse in a zero-sum game
        final_rewards = [0, 0]
        last_player_id = episode_transitions[-1]['player_id']
        first_player_id = 1 - last_player_id
        final_rewards[last_player_id] = final_reward_for_last_player
        final_rewards[first_player_id] = -final_reward_for_last_player

        # now, push all transitions to the buffers with the correct final reward
        for t in episode_transitions:
            player_id = t['player_id']
            agent = agents[player_id]

            # the reward for this player is their final outcome for the hand
            final_reward = final_rewards[player_id]

            agent.rl_buffer.push(t['obs'], t['action'], final_reward, t['next_obs'], t['done'])

            # sl buffer learns from the best-response actions, regardless of outcome
            agent.sl_buffer.push(t['obs'], t['action'])

        # perform a learning step for both agents
        for agent in agents:
            rl_loss, sl_loss = agent.learn()
            if rl_loss is not None:
                rl_losses.append(rl_loss)
            if sl_loss is not None:
                sl_losses.append(sl_loss)

        # periodic logging
        if (episode + 1) % args.log_every == 0:
            avg_rl_loss = sum(rl_losses) / len(rl_losses) if rl_losses else 0
            avg_sl_loss = sum(sl_losses) / len(sl_losses) if sl_losses else 0

            # store for plotting
            logged_episodes.append(episode + 1)
            rl_loss_history.append(avg_rl_loss)
            sl_loss_history.append(avg_sl_loss)

            print(f"episode {episode + 1}/{args.num_episodes} | "
                  f"avg rl loss: {avg_rl_loss:.4f} | "
                  f"avg sl loss: {avg_sl_loss:.4f}")

    print("\nself-play test finished.")

    # plotting the results
    if logged_episodes:
        plt.figure(figsize=(12, 6))
        plt.plot(logged_episodes, rl_loss_history, label='avg rl loss')
        plt.plot(logged_episodes, sl_loss_history, label='avg sl loss')
        plt.title('nfsp agent training losses')
        plt.xlabel('episodes')
        plt.ylabel('average loss')
        plt.legend()
        plt.grid(True)
        print("displaying loss plot. close the plot window to exit.")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test script for the nfsp agent.")
    parser.add_argument("--num-cards", type=int, default=3, choices=[3, 4], help="number of cards in the deck.")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes for self-play.")
    parser.add_argument("--log-every", type=int, default=1000, help="how often to log average losses.")

    parsed_args = parser.parse_args()
    main(parsed_args)
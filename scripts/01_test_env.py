# scripts/01_test_env.py

import sys
import os
import random
import argparse

# add the project root to the python path to allow imports from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poker_env.kuhn_env import KuhnPokerEnv


def main(args):
    """
    runs a test of the kuhn poker environment with random agents.
    """
    print(f"initializing environment for {args.num_cards}-card kuhn poker...")
    env = KuhnPokerEnv(num_cards=args.num_cards)

    # card mapping for logging
    card_map = {0: 'J', 1: 'Q', 2: 'K', 3: 'A'}

    for episode in range(args.num_episodes):
        print("\n" + "=" * 40)
        print(f"episode {episode + 1}")
        print("=" * 40)

        obs = env.reset()
        done = False

        # store hands for the final print
        player_hands = {0: card_map.get(env.cards[0], env.cards[0]),
                        1: card_map.get(env.cards[1], env.cards[1])}

        final_history = ""

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions()
            action = random.choice(legal_actions)
            action_str = "PASS" if action == KuhnPokerEnv.PASS else "BET"

            print(f"player {current_player} (hand: {player_hands[current_player]}) sees obs: {obs}")

            next_obs, reward, done, _ = env.step(action)

            print(f"  -> takes action '{action_str}'")
            if done:
                print(f"  -> hand ends. player {current_player} receives reward: {reward}")
                final_history = env.history

            obs = next_obs

        print("-" * 40)
        print(f"final result:")
        print(f"  player 0 had: {player_hands[0]}")
        print(f"  player 1 had: {player_hands[1]}")
        print(f"  betting history: '{final_history}'")
        print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test script for kuhn poker environment")
    parser.add_argument(
        "--num-cards",
        type=int,
        default=3,
        choices=[3, 4],
        help="number of cards in the deck."
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="number of hands to play."
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
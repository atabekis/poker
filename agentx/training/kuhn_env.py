# rl_agent/training/kuhn_env.py

"""
This module provides a lightweight, and fast implementation of the poker env, designed for training the RL agent.

Note on design:
this internal environment is intentionally decoupled from the main tournament's ClientGameState object.
The primary reasons for this are performance and flexibility.
By creating a minimal env we:
    - achieve (significantly) faster simulation speeds, which we use to train for millions of hands
    - parallelize the training across multiple CPU cores, this we use for embarrassingly parallel training in HPC.
"""

import random
import numpy as np
from typing import Optional


class KuhnPokerEnv:
    """
    A match-aware kuhn poker environment.

    Tracks player bankrolls throughout a match. The state representation includes a normalized bankroll feature,
    and the rewards are the direct change in a player's bankroll after a hand.
    """
    PASS = 0
    BET = 1
    NUM_ACTIONS = 2

    def __init__(self, num_cards: int = 3, starting_bankroll: int = 5):
        """
        Initializes the Kuhn poker environment

        :param num_cards: number of cards in the deck (3 or 4).
        :param starting_bankroll: the bankroll each player starts a match with
        """

        self.num_cards = num_cards
        self.deck = list(range(self.num_cards))
        self.starting_bankroll = starting_bankroll

        # state dim: one-hot card + 4 history features + 1 normalized bankroll feature
        self.state_dim = self.num_cards + 4 + 1

        # match-level state
        self.bankrolls = [self.starting_bankroll, self.starting_bankroll]

        # hand-level state
        self.cards = [0, 0]
        self.history = ""
        self.current_player = 0
        self.done = False

    def reset_match(self) -> np.ndarray:
        """
        Resets the environment to the start of a new match with full bankrolls

        :returns: the initial observation for the first hand of the match.
        """
        self.bankrolls = [self.starting_bankroll, self.starting_bankroll]
        return self.reset_hand()

    def reset_hand(self) -> np.ndarray:
        """
        Resets the env for the next hand within a match, deducts antes from the current bankrolls.

        :returns: the initial observation for the new hand.
        """
        # deduct antes
        self.bankrolls[0] -= 1
        self.bankrolls[1] -= 1

        # reset hand-specific state
        random.shuffle(self.deck)
        self.cards = self.deck[:2]
        self.history = ""
        self.current_player = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        creates the state vector for the current player, including bankroll info

        :returns: a numpy array representing the current state.
        """
        card_feature = np.zeros(self.num_cards)
        card_feature[self.cards[self.current_player]] = 1

        history_feature = np.zeros(4)
        if self.history == "":  # p0's turn to act first
            history_feature[0] = 1
        elif self.history == "p":  # p1's turn after p0 passed
            history_feature[1] = 1
        elif self.history == "b":  # p1's turn after p0 bet
            history_feature[2] = 1
        elif self.history == "pb":  # p0's turn after p1 raised
            history_feature[3] = 1

        # add normalized bankroll feature
        total_bankroll = sum(self.bankrolls)
        # handle zero division if both players are all-in and have 0 bankroll
        if total_bankroll <= 0: normalized_bankroll = 0.5
        else: normalized_bankroll = self.bankrolls[self.current_player] / total_bankroll

        bankroll_feature = np.array([normalized_bankroll])

        return np.concatenate((card_feature, history_feature, bankroll_feature)).astype(np.float32)

    def get_legal_actions(self) -> list[int]:
        return [self.PASS, self.BET]

    def _resolve_hand(self):
        """
        Resolves the hand and updates player bankrolls directly, does not return a reward, modifies the env state.
        """
        pot = 2  # from the antes

        # handle bets/calls to determine final pot and update bankrolls
        if self.history == 'bp':  # p0 bets, p1 folds
            self.bankrolls[0] -= 1
            pot += 1
            self.bankrolls[0] += pot
        elif self.history == 'pbp':  # p0 passes, p1 bets, p0 folds
            self.bankrolls[1] -= 1
            pot += 1
            self.bankrolls[1] += pot
        elif self.history in ['bb', 'pbb']:  # showdown after betting
            self.bankrolls[0] -= 1
            self.bankrolls[1] -= 1
            pot += 2
            winner = 0 if self.cards[0] > self.cards[1] else 1
            self.bankrolls[winner] += pot
        elif self.history == 'pp':  # showdown after passing
            winner = 0 if self.cards[0] > self.cards[1] else 1
            self.bankrolls[winner] += pot
        else:
            raise RuntimeError(f"invalid terminal history: {self.history}")

    def step(self, action: int) -> tuple[Optional[np.ndarray], int, bool, dict]:
        """
        Advances the env by one step. The reward is now the change in bankroll. The done flag indicates the end
        of a hand, while the info dict indicates the end of a match.

        :param action: action taken by the current player
        :returns: tuple of (next_observation, reward, hand_done, info_dict)
        """

        if self.done: raise ValueError("step() called after the hand was done.") # should not happen

        action_char = 'p' if action == self.PASS else 'b'
        self.history += action_char

        # check for terminal hand states
        if self.history in ["pp", "bb", "bp", "pbp", "pbb"]:
            self.done = True

            bankrolls_before = self.bankrolls[:]
            self._resolve_hand()

            # reward is the delta in bankroll for the player who just acted
            rewards = [b_after - b_before for b_after, b_before in zip(self.bankrolls, bankrolls_before)]
            reward_for_acting_player = rewards[self.current_player]

            match_over = any(b <= 0 for b in self.bankrolls)
            info = {
                'match_over': match_over,
                'final_bankrolls': self.bankrolls
            }

            return None, reward_for_acting_player, True, info

        # if the hand is not over, switch players and return zero reward
        self.current_player = 1 - self.current_player
        return self._get_observation(), 0, False, {}
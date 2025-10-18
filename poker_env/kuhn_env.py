# poker_env/kuhn_env.py

import random
import numpy as np
from typing import List, Tuple, Optional


class KuhnPokerEnv:
    """
    a class representing the kuhn poker environment.

    this environment is parameterized to handle a variable number of cards
    in the deck, making it suitable for both 3-card and 4-card kuhn poker.
    it provides a gym-like interface with reset() and step() methods.
    the state is represented as a one-hot encoded vector.
    """
    # define actions as class constants for clarity
    PASS = 0
    BET = 1
    NUM_ACTIONS = 2

    def __init__(self, num_cards: int = 3):
        """
        initializes the kuhn poker environment.

        :param num_cards: the number of cards in the deck (e.g., 3 or 4).
        """
        self.num_cards = num_cards
        self.deck = list(range(self.num_cards))

        # the state dimension is num_cards (for one-hot card) + 4 (for betting history)
        self.state_dim = self.num_cards + 4

        # state variables that will be reset each hand
        self.cards = [0, 0]
        self.history = ""
        self.current_player = 0
        self.done = False

    def _get_observation(self) -> np.ndarray:
        """
        creates the state vector for the current player.

        the state vector consists of two parts:
        1. a one-hot encoding of the player's card.
        2. a 4-element binary vector representing the betting history.
           [p0_passed, p0_bet, p1_passed, p1_bet]

        :returns: a numpy array representing the current state.
        """
        # one-hot encode the current player's card
        card_feature = np.zeros(self.num_cards)
        card_feature[self.cards[self.current_player]] = 1

        # encode the betting history
        history_feature = np.zeros(4)
        if 'p' in self.history[0::2]:  # player 0's moves are at even indices
            history_feature[0] = 1
        if 'b' in self.history[0::2]:
            history_feature[1] = 1
        if 'p' in self.history[1::2]:  # player 1's moves are at odd indices
            history_feature[2] = 1
        if 'b' in self.history[1::2]:
            history_feature[3] = 1

        return np.concatenate((card_feature, history_feature)).astype(np.float32)

    def reset(self) -> np.ndarray:
        """
        resets the environment to the start of a new hand.

        :returns: the initial observation for player 0.
        """
        random.shuffle(self.deck)
        self.cards = self.deck[:2]
        self.history = ""
        self.current_player = 0
        self.done = False
        return self._get_observation()

    def get_legal_actions(self) -> List[int]:
        """
        returns the list of legal actions for the current player.
        in kuhn poker, both actions are always legal.

        :returns: a list containing the integer representation of legal actions.
        """
        return [self.PASS, self.BET]

    def _resolve_hand(self) -> int:
        """
        calculates the reward for the player who just acted, at the end of a hand.

        :returns: the integer reward for the player who took the terminal action.
        """
        # when this is called, self.current_player is still the player who just made the terminal move.
        acting_player = self.current_player

        # case 1: one player bets and the other folds
        if self.history == "bp":  # p0 bets, p1 (acting_player) folds
            return -1
        if self.history == "pbp":  # p0 passes, p1 bets, p0 (acting_player) folds
            return -1

        # case 2: showdown
        winner = 0 if self.cards[0] > self.cards[1] else 1

        if self.history == "pp":  # both players pass
            pot_size = 1  # winner gets the ante
            return pot_size if acting_player == winner else -pot_size

        if self.history in ["bb", "pbb"]:  # both players bet
            pot_size = 2  # winner gets the ante + the bet
            return pot_size if acting_player == winner else -pot_size

        # this should not be reached
        raise RuntimeError(f"invalid terminal history: {self.history}")
    # def _resolve_hand(self) -> int:
    #     """
    #     calculates the reward for the player who just acted, at the end of a hand.
    #
    #     :returns: the integer reward for the player who took the terminal action.
    #     """
    #     # the player who just acted is the one *before* the turn switch
    #     acting_player = 1 - self.current_player
    #     opponent = self.current_player
    #
    #     # case 1: one player bets and the other folds
    #     if self.history == "bp":  # player 0 bets, player 1 passes (folds)
    #         # player 1 (acting_player) folded, loses their ante
    #         return -1
    #     if self.history == "pbp":  # player 0 passes, player 1 bets, player 0 passes (folds)
    #         # player 0 (acting_player) folded, loses their ante
    #         return -1
    #
    #     # case 2: showdown
    #     winner = 0 if self.cards[0] > self.cards[1] else 1
    #
    #     if self.history == "pp":  # both players pass
    #         # winner gets the ante from the loser
    #         return 1 if acting_player == winner else -1
    #
    #     if self.history in ["bb", "pbb"]:  # both players bet
    #         # winner gets the ante + the bet (2 units) from the loser
    #         return 2 if acting_player == winner else -2
    #
    #     # this should not be reached
    #     raise RuntimeError(f"invalid terminal history: {self.history}")

    def step(self, action: int) -> Tuple[Optional[np.ndarray], int, bool, dict]:
        """
        advances the environment by one step.

        :param action: the action taken by the current player.
        :returns: a tuple containing (next_observation, reward, done, info_dict).
        """
        if self.done:
            raise ValueError("step() called after the episode was done.")

        action_char = 'p' if action == self.PASS else 'b'
        self.history += action_char

        # check for terminal states
        if self.history in ["pp", "bb", "bp", "pbp", "pbb"]:
            self.done = True
            reward = self._resolve_hand()
            return None, reward, True, {}

        # if the game is not over, switch players
        self.current_player = 1 - self.current_player

        # no intermediate rewards
        reward = 0
        return self._get_observation(), reward, False, {}
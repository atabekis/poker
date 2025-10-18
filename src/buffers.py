# src/buffers.py

import random
import collections
import numpy as np
import torch

class ReplayBuffer:
    """a standard experience replay buffer for dqn."""

    def __init__(self, capacity: int):
        """
        initializes the replay buffer.

        :param capacity: the maximum number of transitions to store.
        """
        self.memory = collections.deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        adds a transition to the buffer.

        :param state: the state observed.
        :param action: the action taken.
        :param reward: the reward received.
        :param next_state: the next state observed.
        :param done: a flag indicating if the episode terminated.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """
        samples a batch of transitions from the buffer.

        :param batch_size: the number of transitions to sample.
        :returns: a tuple of tensors for states, actions, rewards, next_states, and dones.
        """
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1)  # actions need to be long for indexing
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1)

        # handle the case where some next_states are none (terminal states)
        non_final_next_states = torch.from_numpy(np.array([s for s in next_states if s is not None])).float()

        # create a mask for non-final states
        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)

        dones = torch.from_numpy(np.array(dones)).float().unsqueeze(1)

        return states, actions, rewards, non_final_next_states, non_final_mask, dones

    def __len__(self) -> int:
        return len(self.memory)


class ReservoirBuffer:
    """
    a reservoir sampling buffer for supervised learning.

    this buffer maintains a uniform random sample of all (state, action)
    pairs seen so far, which is required for training the average strategy network.
    """

    def __init__(self, capacity: int):
        """
        initializes the reservoir buffer.

        :param capacity: the maximum number of items to store.
        """
        self.capacity = capacity
        self.memory: list[tuple[np.ndarray, int]] = []
        self.num_items_seen = 0

    def push(self, state: np.ndarray, action: int):
        """
        adds a (state, action) pair to the buffer using reservoir sampling.

        :param state: the state observed.
        :param action: the action taken in that state by the best response policy.
        """
        self.num_items_seen += 1
        if len(self.memory) < self.capacity:
            self.memory.append((state, action))
        else:
            # with probability capacity/num_items_seen, replace a random item
            j = random.randint(0, self.num_items_seen - 1)
            if j < self.capacity:
                self.memory[j] = (state, action)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        samples a batch of (state, action) pairs from the buffer.

        :param batch_size: the number of pairs to sample.
        :returns: a tuple of tensors for states and actions.
        """
        samples = random.sample(self.memory, batch_size)
        states, actions = zip(*samples)

        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long()  # no unsqueeze needed for cross-entropy loss

        return states, actions

    def __len__(self) -> int:
        return len(self.memory)
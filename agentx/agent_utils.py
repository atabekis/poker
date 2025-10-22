# rl_agent/agent_utils.py

"""
This file includes the building blocks for the Neural Fictitious Self-PLay (NFSP) agent.

The main NFSP agent is defined in agent.py.

The NFSP implementation is adapted from:
 - https://arxiv.org/pdf/1603.01121,
 - https://towardsdatascience.com/neural-fictitious-self-play-800612b4a53f/
"""

# Python imports
import math
import random
import collections
import numpy as np

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    A noisy linear layer for exploration, replacing epsilon-greedy strategies.

    Implements factorised gaussian noise, which is more efficient than independent noise.
    The noise is generated on the fly during the forward pass to ensure we can have multiple forward calls within
    a single backward pass.

    The output of the layer is a linear transformation of the input, where the
    weights and biases are perturbed by a learned, parametric noise distribution.

    Adapted from: https://arxiv.org/abs/1706.10295
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        initializes the noisy linear layer.

        :param in_features: size of each input sample
        :param out_features: size of each output sample
        :param std_init: initial standard deviation of the noise parameters
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # learnable params for the mean of the weights and biases, this gives us deterministic vars.
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # learnable parameters for the standard deviation of the noise
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """initializes the learnable parameters using kaiming uniform for the mean."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        # initialize standard deviations to a constant value
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs a forward pass. if in training mode, it uses noisy weights.
        if in eval mode, it uses the deterministic mean weights for exploitation.
        """
        if self.training:
            # sample noise from a standard normal distribution
            weight_epsilon = torch.randn_like(self.weight_sigma)
            bias_epsilon = torch.randn_like(self.bias_sigma)

            # combine mean and noise to get stoch. w and b
            noisy_weight = self.weight_mu + self.weight_sigma * weight_epsilon
            noisy_bias = self.bias_mu + self.bias_sigma * bias_epsilon

            return F.linear(x, noisy_weight, noisy_bias)
        else: # in eval mode, use the deterministic mean weights which helps in stable evaluation
            return F.linear(x, self.weight_mu, self.bias_mu)



class BestResponseNet(nn.Module):
    """
    Neural network for the best response (RL) policy.

    This network uses the noisy linear layers to perform exploration, which replaces the need for an external
    epsilon-greedy strategy. It learns a q-function to approximate the value of taking an action in a given state (DQN).
    """
    def __init__(self, state_dim: int, num_actions: int, hidden_size: int = 128):
        super(BestResponseNet, self).__init__()
        self.network = nn.Sequential(
            NoisyLinear(state_dim, hidden_size),
            nn.ReLU(),
            NoisyLinear(hidden_size, hidden_size),
            nn.ReLU(),
            NoisyLinear(hidden_size, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1: x = x.unsqueeze(0) # if a single sample is passed, add batch dim.
        return self.network(x)


class AverageStrategyNet(nn.Module):
    """
    Neural network for the average strategy (SL) policy.

    This network is deterministic, it uses the linear layers. It learns and represents the historical average policy
    of the BestResponseNet.

    It is trained via supervised learning and is not used for exploration.
    """
    def __init__(self, state_dim: int, num_actions: int, hidden_size: int = 128):
        super(AverageStrategyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1: x = x.unsqueeze(0)
        return self.network(x)



class ReplayBuffer:
    """A fixed-size circular buffer for storing experience tuples."""
    def __init__(self, capacity: int):
        self.memory = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1)
        dones = torch.from_numpy(np.array(dones)).float().unsqueeze(1)

        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)

        # create a list of next states, using a zero array for terminal states
        next_states_list = []
        state_dim = states.shape[1]

        for s in next_states:
            # if exists, add to the list, else add a zero array for the next states
            if s is not None: next_states_list.append(s)
            else: next_states_list.append(np.zeros(state_dim, dtype=np.float32))

        next_states_tensor = torch.from_numpy(np.array(next_states_list)).float() # convert the list to tensor

        return states, actions, rewards, next_states_tensor, non_final_mask, dones


    def __len__(self) -> int:
        return len(self.memory)


class ReservoirBuffer:
    """
    Constructs a buffer that uses reservoir sampling to maintain a uniform random sample of all items seen so far.
    We have to use this bufer to ensure we are training an unbiased average strategy network.
    It ensures the training data is a uniform sample of the entire history of best-response actions.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: list[tuple[np.ndarray, int]] = []
        self.num_items_seen = 0

    def push(self, state: np.ndarray, action: int):
        """
        Adds a state-action pair to the buffer using reservoir sampling.

        If the buffer is not yet at capacity, the new item is added.
        If the buffer is full, each new item has a decreasing probability (capacity / num_items_seen) of replacing
        a random existing item in the buffer.

        :param state: The state observation.
        :param action: The action taken.
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
        """Randomly samples a batch of (state, action) pairs from the buffer."""
        samples = random.sample(self.memory, batch_size)
        states, actions = zip(*samples)

        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long()

        return states, actions

    def __len__(self) -> int:
        return len(self.memory)


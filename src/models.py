# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Module):
    """
    a noisy linear layer for exploration (corrected implementation).

    this version creates noise tensors on-the-fly during the forward pass
    to avoid in-place modification errors when the module is called multiple
    times within a single backward pass.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        initializes the noisy linear layer.

        :param in_features: size of each input sample.
        :param out_features: size of each output sample.
        :param std_init: initial standard deviation of the noise.
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # learnable parameters for the mean of the weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # learnable parameters for the standard deviation of the noise
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # note: we no longer register persistent buffers for noise

        self.reset_parameters()

    def reset_parameters(self):
        """initializes the learnable parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))  # corrected from in_features for bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs a forward pass. if in training mode, it uses noisy weights.
        if in eval mode, it uses the deterministic mean weights.
        """
        if self.training:
            # create noise as local variables on the correct device
            weight_epsilon = torch.randn_like(self.weight_sigma)
            bias_epsilon = torch.randn_like(self.bias_sigma)

            # combine mean and scaled noise to get the stochastic weights and biases
            noisy_weight = self.weight_mu + self.weight_sigma * weight_epsilon
            noisy_bias = self.bias_mu + self.bias_sigma * bias_epsilon

            return F.linear(x, noisy_weight, noisy_bias)
        else:
            # in eval mode, use the deterministic mean weights
            return F.linear(x, self.weight_mu, self.bias_mu)


class BestResponseNet(nn.Module):
    """
    a neural network for the best response (rl) policy.

    this version uses noisylinear layers to perform exploration, replacing the
    need for an epsilon-greedy strategy.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_size: int = 128):
        """
        initializes the best response network with noisy layers.

        :param state_dim: the dimension of the input state vector.
        :param num_actions: the number of possible actions.
        :param hidden_size: the number of neurons in the hidden layers.
        """
        super(BestResponseNet, self).__init__()
        self.network = nn.Sequential(
            NoisyLinear(state_dim, hidden_size),
            nn.ReLU(),
            NoisyLinear(hidden_size, hidden_size),
            nn.ReLU(),
            NoisyLinear(hidden_size, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs a forward pass through the network.

        :param x: the input state tensor.
        :returns: a tensor of q-values for each action.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        return self.network(x)


class AverageStrategyNet(nn.Module):
    """
    a neural network for the average strategy (sl) policy.

    this network remains deterministic, using standard linear layers. its purpose
    is to represent the learned average policy, not to explore.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_size: int = 128):
        """
        initializes the average strategy network.

        :param state_dim: the dimension of the input state vector.
        :param num_actions: the number of possible actions.
        :param hidden_size: the number of neurons in the hidden layers.
        """
        super(AverageStrategyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs a forward pass through the network.

        :param x: the input state tensor.
        :returns: a tensor of logits for each action.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        return self.network(x)
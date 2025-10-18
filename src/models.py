# src/models.py

import torch
import torch.nn as nn


class BestResponseNet(nn.Module):
    """
    a neural network for the best response (rl) policy.

    this is essentially a dqn model that learns the q-value for each action
    given a state. it is trained using experience replay.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_size: int = 128):
        """
        initializes the best response network.

        :param state_dim: the dimension of the input state vector.
        :param num_actions: the number of possible actions.
        :param hidden_size: the number of neurons in the hidden layers.
        """
        super(BestResponseNet, self).__init__()
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
        :returns: a tensor of q-values for each action.
        """
        # ensure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # if the tensor is 1d (a single sample), add a batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        return self.network(x)


class AverageStrategyNet(nn.Module):
    """
    a neural network for the average strategy (sl) policy.

    this model learns to imitate the historical average behavior of the
    best response policy. it is trained via supervised learning on
    (state, action) pairs.
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
        # ensure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # if the tensor is 1d (a single sample), add a batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        return self.network(x)
# rl_agent/inference.py

"""
This module provides our tournament agent class, R2Deal2.
It is designed for only inference, loading a pre-trained policy and containing no training or exploration logic.
"""

# Python imports
import torch
import numpy as np

# Local imports
from .agent_utils import AverageStrategyNet


class R2Deal2:
    """
    An agent for tournament play that loads a pre-trained policy

    It uses the AverageStrategyNet network to determine the best action from a trained NFSP agent
    by performing a single forward pass.
    """
    def __init__(self, policy_path: str, state_dim: int, num_actions: int, device: str = 'cpu'):
        """
        initializes the tournament agent by loading a policy from a file.

        :param policy_path: the file path to the saved .pth policy
        :param state_dim: the dimension of the environment's state vector
        :param num_actions: the number of legal actions in the environment
        :param device: the torch device to run inference on (cpu or cuda) defaults to cpu.
        """
        self.device = torch.device(device)

        # deduce hidden_size from the state_dict for robustness, so we don't have to pass the config here.
        temp_state_dict = torch.load(policy_path, map_location=self.device)
        hidden_size = temp_state_dict['network.0.weight'].shape[0]

        # initialize the net
        self.avg_strategy_net = AverageStrategyNet(state_dim, num_actions, hidden_size).to(self.device)

        # load the weights and set the network to eval
        self.avg_strategy_net.load_state_dict(temp_state_dict)
        self.avg_strategy_net.eval()

    def act(self, state: np.ndarray) -> int:
        """
        Gets the best action for a given state using the loaded policy. It performs a greedy action selection
        based on the highest logit output by the AverageStrategyNet.

        :param state: the current state of the environment (np.array)
        :returns: the integer representing the chosen action
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device)

            logits = self.avg_strategy_net(state_tensor) # perform a forward pass to get the action logits
            action = logits.argmax(dim=1).item() # select the action with the highest logit

        return action
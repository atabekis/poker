# src/agent.py

import random
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .models import BestResponseNet, AverageStrategyNet
from .buffers import ReplayBuffer, ReservoirBuffer


class NFSPAgent:
    """
    an agent that implements a robust version of the nfsp algorithm.
    this version uses noisy networks for exploration.
    """

    def __init__(self, config: dict):
        """
        initializes the nfsp agent.

        :param config: a dictionary containing hyperparameters.
        """
        self.state_dim = config['state_dim']
        self.num_actions = config['num_actions']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyperparameters
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.tau = config.get('tau', 0.005)
        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.eta = config['eta']

        # note: epsilon hyperparameters are no longer needed with noisy networks.

        hidden_size = config.get('hidden_size', 128)

        # networks
        self.best_response_net = BestResponseNet(self.state_dim, self.num_actions, hidden_size).to(self.device)
        self.target_net = BestResponseNet(self.state_dim, self.num_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.best_response_net.state_dict())
        self.target_net.eval()  # target network is only for inference (and uses deterministic weights)

        self.avg_strategy_net = AverageStrategyNet(self.state_dim, self.num_actions, hidden_size).to(self.device)

        # optimizers
        self.rl_optimizer = optim.Adam(self.best_response_net.parameters(), lr=config['rl_learning_rate'])
        self.sl_optimizer = optim.Adam(self.avg_strategy_net.parameters(), lr=config['sl_learning_rate'])

        # memory buffers
        self.rl_buffer = ReplayBuffer(config['rl_buffer_size'])
        self.sl_buffer = ReservoirBuffer(config['sl_buffer_size'])

    def act(self, state: np.ndarray) -> int:
        """
        chooses an action based on the nfsp policy.
        """
        if random.random() < self.eta:
            # use average strategy network (supervised learning policy)
            state_tensor = torch.from_numpy(state).float().to(self.device)
            # set to eval mode for deterministic action sampling
            self.avg_strategy_net.eval()
            with torch.no_grad():
                logits = self.avg_strategy_net(state_tensor)
                probs = F.softmax(logits, dim=1)
                action = torch.multinomial(probs, num_samples=1).item()
            self.avg_strategy_net.train()
            return action
        else:
            # use best response network (reinforcement learning policy)
            # exploration is handled by the noisy layers, so we can be greedy.
            state_tensor = torch.from_numpy(state).float().to(self.device)
            # ensure network is in train mode for noise to be active
            self.best_response_net.train()
            with torch.no_grad():
                q_values = self.best_response_net(state_tensor)
                action = q_values.max(1)[1].item()
            return action

    def _update_rl_net(self) -> float | None:
        """
        performs a single update step on the best response network (dqn).
        uses double dqn (ddqn) for more stable q-value estimation.
        """
        if len(self.rl_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, non_final_mask, dones = self.rl_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        non_final_mask = non_final_mask.to(self.device)

        # q-values for the current states from the main network. gradients will flow through this.
        current_q_values = self.best_response_net(states).gather(1, actions)

        # --- start of the fix ---
        # calculate the target q-values in a no_grad context to prevent in-place modification errors.
        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, device=self.device)
            if next_states.size(0) > 0:
                # 1. select the best action for the next state using the *main* network.
                #    the network is in train() mode, so noise is active, providing noisy action selection.
                next_state_actions = self.best_response_net(next_states).max(1)[1].unsqueeze(1)

                # 2. evaluate the q-value of that chosen action using the stable *target* network.
                #    the target network is in eval() mode, so its weights are deterministic.
                next_q_values[non_final_mask] = self.target_net(next_states).gather(1, next_state_actions).squeeze(1)

            # compute the final target value using the bellman equation.
            target_q_values = rewards + (self.gamma * next_q_values.unsqueeze(1) * (1 - dones))
        # --- end of the fix ---

        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.rl_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.best_response_net.parameters(), self.gradient_clip)
        self.rl_optimizer.step()
        return loss.item()

    def _update_sl_net(self) -> float | None:
        """performs a single update step on the average strategy network."""
        if len(self.sl_buffer) < self.batch_size:
            return None

        states, actions = self.sl_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)

        logits = self.avg_strategy_net(states)
        loss = F.cross_entropy(logits, actions)

        self.sl_optimizer.zero_grad()
        loss.backward()
        self.sl_optimizer.step()
        return loss.item()

    def _soft_update_target_net(self):
        """
        performs a soft update of the target network's weights.
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.best_response_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                        1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def learn(self) -> tuple[float | None, float | None]:
        """
        executes one learning step for both the rl and sl networks.
        """
        rl_loss = self._update_rl_net()
        sl_loss = self._update_sl_net()

        if rl_loss is not None:
            self._soft_update_target_net()

        return rl_loss, sl_loss

    def get_state_dicts(self) -> dict:
        """returns the state dictionaries of the agent's networks."""
        return {
            'br_net': self.best_response_net.state_dict(),
            'as_net': self.avg_strategy_net.state_dict()
        }

    def load_state_dicts(self, weights: dict):
        """loads weights into the agent's networks from state dictionaries."""
        self.best_response_net.load_state_dict(weights['br_net'])
        self.target_net.load_state_dict(weights['br_net'])
        self.avg_strategy_net.load_state_dict(weights['as_net'])
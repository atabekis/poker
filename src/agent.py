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
    includes a target network and gradient clipping for dqn stability.
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
        self.tau = config.get('tau', 0.005)  # for soft target network updates
        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.eta = config['eta']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']

        # networks
        self.best_response_net = BestResponseNet(self.state_dim, self.num_actions).to(self.device)
        # new: target network for dqn stability
        self.target_net = BestResponseNet(self.state_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.best_response_net.state_dict())
        self.target_net.eval()  # target network is only for inference

        self.avg_strategy_net = AverageStrategyNet(self.state_dim, self.num_actions).to(self.device)

        # optimizers
        self.rl_optimizer = optim.Adam(self.best_response_net.parameters(), lr=config['rl_learning_rate'])
        self.sl_optimizer = optim.Adam(self.avg_strategy_net.parameters(), lr=config['sl_learning_rate'])

        # memory buffers
        self.rl_buffer = ReplayBuffer(config['rl_buffer_size'])
        self.sl_buffer = ReservoirBuffer(config['sl_buffer_size'])

        self.steps_done = 0

    def act(self, state: np.ndarray) -> int:
        """
        chooses an action based on the nfsp policy.
        """
        if random.random() < self.eta:
            state_tensor = torch.from_numpy(state).float().to(self.device)
            self.avg_strategy_net.eval()
            with torch.no_grad():
                logits = self.avg_strategy_net(state_tensor)
                probs = F.softmax(logits, dim=1)
                action = torch.multinomial(probs, num_samples=1).item()
            self.avg_strategy_net.train()
            return action
        else:
            eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            math.exp(-1. * self.steps_done / self.epsilon_decay)
            self.steps_done += 1

            if random.random() > eps_threshold:
                state_tensor = torch.from_numpy(state).float().to(self.device)
                self.best_response_net.eval()
                with torch.no_grad():
                    q_values = self.best_response_net(state_tensor)
                    action = q_values.max(1)[1].item()
                self.best_response_net.train()
                return action
            else:
                return random.randrange(self.num_actions)

    def _update_rl_net(self) -> float | None:
        """performs a single update step on the best response network (dqn)."""
        if len(self.rl_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, non_final_mask, dones = self.rl_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        non_final_mask = non_final_mask.to(self.device)

        current_q_values = self.best_response_net(states).gather(1, actions)

        # the target is simply the final (monte carlo) reward we stored.
        target_q_values = rewards

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
        θ_target = τ*θ_local + (1 - τ)*θ_target
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

    # --- missing methods added below ---

    def get_state_dicts(self) -> dict:
        """
        returns the state dictionaries of the agent's networks.

        :returns: a dictionary containing the state dicts for both networks.
        """
        return {
            'br_net': self.best_response_net.state_dict(),
            'as_net': self.avg_strategy_net.state_dict()
        }

    def load_state_dicts(self, weights: dict):
        """
        loads weights into the agent's networks from state dictionaries.

        :param weights: a dictionary containing the state dicts.
        """
        self.best_response_net.load_state_dict(weights['br_net'])
        self.target_net.load_state_dict(weights['br_net'])  # target net syncs with br_net
        self.avg_strategy_net.load_state_dict(weights['as_net'])
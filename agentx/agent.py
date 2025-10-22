# rl_agent/agent.py

"""
This file provides the main wrapper class for the NFSP agent.

Note: the agent used for playing rounds is defined in inference.py, where we use the trained policies to make
inferences.
"""

# Python imports
import random
import numpy as np
from typing import Optional, Any

# Torch imports
import torch
import torch.optim as optim
import torch.nn.functional as F

# Local imports
from .agent_utils import BestResponseNet, AverageStrategyNet, ReplayBuffer, ReservoirBuffer

BEST_RESPONSE_POLICY = "br"  # we use these definitions in the training logic
AVERAGE_POLICY = "avg"


class NFSPAgent:
    """
    An agent that implements the Neural Fictitious Self-Play (NFSP) algorithm.

    Our agent keeps two separate learning processes:
    1. An RL process that learns the approximate best response to the opponent average strategy. This is implemented
     as a double-DQN with noisy networks for exploration.

    2. An SL process that learns the historical average of the RL agent's own behavior.

    During gameplay, the agent acts according to a mixture of these two policies, controlled by the eta parameter.

    The NFSP implementation is adapted from:
         - https://arxiv.org/pdf/1603.01121,
         - https://towardsdatascience.com/neural-fictitious-self-play-800612b4a53f/
    """

    def __init__(self, config: dict[str, Any]):
        self.state_dim = config['state_dim']
        self.num_actions = config['num_actions']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.tau = config.get('tau', 0.005)
        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.eta = config['eta']
        hidden_size = config.get('hidden_size', 128)

        self.best_response_net = BestResponseNet(self.state_dim, self.num_actions, hidden_size).to(self.device)
        self.target_net = BestResponseNet(self.state_dim, self.num_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.best_response_net.state_dict())
        self.target_net.eval()

        self.avg_strategy_net = AverageStrategyNet(self.state_dim, self.num_actions, hidden_size).to(self.device)

        self.rl_optimizer = optim.Adam(self.best_response_net.parameters(), lr=config['rl_learning_rate'])
        self.sl_optimizer = optim.Adam(self.avg_strategy_net.parameters(), lr=config['sl_learning_rate'])

        self.rl_buffer = ReplayBuffer(config['rl_buffer_size'])
        self.sl_buffer = ReservoirBuffer(config['sl_buffer_size'])

    def act(self, state: np.ndarray) -> tuple[int, str]:
        """Chooses an action based on the mixed NFSP policy."""
        if random.random() < self.eta:
            state_tensor = torch.from_numpy(state).float().to(self.device)
            self.avg_strategy_net.eval()
            with torch.no_grad():
                logits = self.avg_strategy_net(state_tensor)
                probs = F.softmax(logits, dim=1)
                action = torch.multinomial(probs, num_samples=1).item()
            self.avg_strategy_net.train()
            return action, AVERAGE_POLICY
        else:
            state_tensor = torch.from_numpy(state).float().to(self.device)
            self.best_response_net.train()
            with torch.no_grad():
                q_values = self.best_response_net(state_tensor)
                action = q_values.max(1)[1].item()
            return action, BEST_RESPONSE_POLICY

    def _update_rl_net(self) -> Optional[float]:
        """Performs a single DQN update using the double-DQN algorithm (DDQN)."""
        if len(self.rl_buffer) < self.batch_size: return None

        states, actions, rewards, next_states, non_final_mask, dones = self.rl_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = \
            states.to(self.device), actions.to(self.device), rewards.to(self.device), \
                next_states.to(self.device), dones.to(self.device)

        current_q_values = self.best_response_net(states).gather(1, actions)

        with torch.no_grad():
            # select best action for next state
            next_state_actions = self.best_response_net(next_states).max(1)[1].unsqueeze(1)

            # eval. the q-value of selected action via target network
            next_q_values = self.target_net(next_states).gather(1, next_state_actions)

            # compute final target val, bellman
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.rl_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.best_response_net.parameters(), self.gradient_clip)
        self.rl_optimizer.step()
        return loss.item()

    def _update_sl_net(self) -> Optional[float]:
        """Performs a single supervised learning update on the average strategy network"""
        if len(self.sl_buffer) < self.batch_size: return None

        states, actions = self.sl_buffer.sample(self.batch_size)
        states, actions = states.to(self.device), actions.to(self.device)

        logits = self.avg_strategy_net(states)
        loss = F.cross_entropy(logits, actions)

        self.sl_optimizer.zero_grad();
        loss.backward()
        self.sl_optimizer.step()
        return loss.item()

    def _soft_update_target_net(self):
        """Performs a soft update (Polyak averaging) of the target network's weights"""
        target_state_dict = self.target_net.state_dict()
        online_state_dict = self.best_response_net.state_dict()
        for key in online_state_dict:
            target_state_dict[key] = online_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_state_dict)

    def learn(self) -> tuple[Optional[float], Optional[float]]:
        """One learning step for both RL and SL networks"""
        rl_loss = self._update_rl_net()
        sl_loss = self._update_sl_net()

        if rl_loss is not None: self._soft_update_target_net()  # only update if we have valid losses, should not happen

        return rl_loss, sl_loss

    def get_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        """Returns the state dicts of the agent networks for checkpointing"""
        return {
            'br_net': self.best_response_net.state_dict(),
            'as_net': self.avg_strategy_net.state_dict()
        }

    def load_state_dicts(self, weights: dict[str, dict[str, torch.Tensor]]):
        """Loads weights into the agent networks from state dicts"""
        self.best_response_net.load_state_dict(weights['br_net'])
        self.target_net.load_state_dict(weights['br_net'])
        self.avg_strategy_net.load_state_dict(weights['as_net'])
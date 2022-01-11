from typing import Union, List

import torch.nn
# from algorithm.policy import Policy, RolloutRequest

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import Sequential


class PpoPolicyWorker:
    """
    Only implemented for discrete action space
    """

    def __init__(self, state_dim, action_dim, device='cpu'):
        super().__init__()
        self.device = None
        self.model = ActorCritic(state_dim, action_dim)
        self.to_device(device)  # initiate attribute self.device

    @property
    def net(self) -> Union[List[torch.nn.Module], torch.nn.Module]:
        return [self.model.get_actor(), self.model.get_critic()]

    def analyze(self, states):
        """

        Args:
            states: tensor

        Returns:
            logits: logit output from actor_model; tensor of shape (len_states, action_space_dim)
            values: output from critic_model; tensor of shape (len_states, 1)
        """
        states = states.to(self.device)
        logits, values = self.model(states)
        return logits, values

    def rollout(self, states):
        """

        Args:
            states: numpy

        Returns:
            action: tensor
            policy_prob: tensor
        """
        states = states.to(self.device)
        with torch.no_grad():
            logits = self.model.actor_forward(states)  # this right?
            dist = Categorical(logits=logits)
            action = dist.sample()
        return action, dist.log_prob(action)

    def to_device(self, device='cpu'):
        self.device = device
        self.model.to(device)


class ActorCritic(torch.nn.Module):
    """
    Two neural nets: actor & critic
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = torch.nn.Sequential(torch.nn.LayerNorm(state_dim),
                                         torch.nn.Linear(state_dim, 64, ),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(64, 32),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(32, action_dim),
                                         )
        self.critic = torch.nn.Sequential(torch.nn.LayerNorm(state_dim),
                                          torch.nn.Linear(state_dim, 64, ),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(64, 32),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(32, 1)
                                          )  # output dim: (1,)

        # self.train()    # Sets the module in training mode.

    def actor_forward(self, inputs):
        return self.actor(inputs)

    def critic_forward(self, inputs):
        return self.critic(inputs)

    def forward(self, inputs: object) -> object:
        return self.actor(inputs), self.critic(inputs)

    def get_actor(self):
        return self.actor

    def get_critic(self):
        return self.critic

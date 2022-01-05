from typing import Union, List

import torch.nn
from algorithm.policy import Policy, RolloutRequest

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import Sequential


class PpoPolicyWorker(Policy):
    """
    Only implemented for discrete action space
    """

    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)

    @property
    def net(self) -> Union[List[torch.nn.Module], torch.nn.Module]:
        return [self.model.get_actor(), self.model.get_critic()]

    def analyze(self, states, **kwargs):
        """ Generate outputs required for loss computation during training,
                    e.g. value target and action distribution entropies.
        Args:
            sample (namedarraytuple): list of states + list of actions? list of (s, a,  r, p(a)) tuples?
        Returns:
            1. policy softmax for each s (for entire action space)
            2. value_loss: v (r+gamma*v'-v for each (s, a) pair in the trajectory)
            :param states:
        """
        target_probs, values = self.model.forward(torch.tensor(states))
        return target_probs, values

    def rollout(self, states, **kwargs):
        """

        :param states:
        :param kwargs:
        :return: action and probability for action_space
        """
        with torch.no_grad():
            policy_probs = self.model.actor_forward(torch.tensor(states, requires_grad=False)) # this right?
            action = Categorical(policy_probs).sample()
        return action, policy_probs[action]


class ActorCritic(torch.nn.Module):
    """
    Two neural nets: actor & critic
    """

    def __init__(self, state_dim, action_dim):
        super.__init__()
        self.actor = torch.nn.Sequential(torch.nn.LayerNorm(state_dim),
                                         torch.nn.Linear(state_dim, 64, ),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(64, 32),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(32, action_dim),
                                         torch.nn.Functional.softmax()
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

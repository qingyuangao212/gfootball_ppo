import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class PPO_WORKER(torch.nn.Module):
    """
    Two neural nets: actor & critic
    """
    def __init__(self, state_dim, action_dim):
        super(PPO_WORKER, self).__init__()
        self.actor = nn.Sequential(nn.LayerNorm(state_dim),
                                       nn.Linear(state_dim, 64,),
                                       nn.ReLU(),
                                       nn.Linear(64, 32),
                                       nn.ReLU(),
                                       nn.Linear(32, action_dim)
                                       )    # outputs are logits (no softmax layer)
        self.critic = nn.Sequential(nn.LayerNorm(state_dim),
                                       nn.Linear(state_dim, 64,),
                                       nn.ReLU(),
                                       nn.Linear(64, 32),
                                       nn.ReLU(),
                                       nn.Linear(32, 1)
                                       )    # output dim: (1,)

        self.train()    # Sets the module in training mode.

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

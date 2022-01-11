import time
from datetime import datetime

from torch.distributions import Categorical
import torch
import numpy as np
from collections import deque

# device = torch.device('cuda')
# print(device.type)
# np.stack([np.array([1,1]), np.array([2,2])])
# torch.tensor(1) * 3

# a = torch.distributions.Categorical(torch.tensor([1,0,1]))
# print(a.log_prob(torch.tensor([0,2, 1])))
# # print(np.log(0.5))
# x = []
# x += [0]*10 + [1]
# print(x)

# x = torch.tensor([1, 2])
# x.to('cuda')
# print(x.device)
# from actor_worker import FootballActorWorker
# from config import ENV_CONFIG
# from policy_worker import PpoPolicyWorker
#
# def print_variable(var):
#     print(type(var))
#     print(np.array(var).shape)
#     print(np.array(var).flatten())
#
# actor_worker = FootballActorWorker(ENV_CONFIG)
# policy_worker = PpoPolicyWorker(actor_worker.state_dim, actor_worker.action_dim, 'cuda')
#
# state = actor_worker.env.reset()
# print_variable(state)
# state = torch.unsqueeze(torch.from_numpy(state), 0).to('cuda')     # state: to tensor, unsqueeze at 0, to gpu
#
# action, action_prob = policy_worker.rollout(state)  # first dim batch_size: (1, action_dim)
# action = action.squeeze().cpu().numpy()
# action_prob = action_prob.squeeze().cpu().numpy()     # move back CPU
# print_variable(action)
# print_variable(action_prob)
#
# state, reward, done, info = actor_worker.step(action)
#
# print_variable(state)
# print_variable(reward)
# print_variable(done)
# print_variable(info)

# print(float(torch.Tensor((2,))))
# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

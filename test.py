from torch.distributions import Categorical
import torch
import numpy as np
x= torch.array([[1,2,3], [3,2,1]])
# x_dist = Categorical(x)
# sample_x = x_dist.sample([2, 3])
# print(sample_x)
# print(x_dist.entropy())
y = torch.tensor([[1,2,3], [3,2,1]])

z = torch.stack([x, y])
# print(z)
# print(z.shape)
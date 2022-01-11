from datetime import datetime

import torch.optim
from torch.distributions import Categorical

from actor_worker import FootballActorWorker
from policy_worker import PpoPolicyWorker
from utils import *
from config import ENV_CONFIG

import wandb
import logging
# ================================================ PARAMETERS ================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# TRAIN PARAMETERS
MAX_BATCH_SIZE = 1024  # batch size for train_step, drawn from buffer
MAX_TRAJECTORY_STEP = 1024  # max trajectory size for rollout
GAMMA = 0.99
LAMBDA = 0.95
EPSILON_CLIP = 0.2
C1 = 1  # loss weight from MSE(r + v’ - v)
C2 = 0.01  # loss weight for entropy
# OPTIM PARAMETERS
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4

# ================================================ SET UP ================================================

# INITIATE ACTOR and POLICY WORKER
actor_worker = FootballActorWorker(ENV_CONFIG)
policy_worker = PpoPolicyWorker(actor_worker.state_dim, actor_worker.action_dim, DEVICE)

# INITIALIZE MSE loss
MSE = torch.nn.MSELoss()


# ================================================ FUNCTIONS ================================================


def simulate_trajectory(max_steps: int = MAX_TRAJECTORY_STEP):
    """
    Interact with Actor Worker and Policy Worker to Simulate a Trajectory

    Each of the trajectories is a numpy array (with numpy_wrapper decorator) and is of equal length

    Note: Roll out is done on policy_worker.device
    policy_worker.rollout() moves inputs (states) to device (cuda if available) by default;
    Output need to be moved back to CPU


    Args:
        roll_out_device
        max_steps: max length of trajectory

    Returns:
        trajectory ( Tuple[states, actions, action_logprobs, rewards] ):
            Each element is list of length T the following dtype and dimension:
                STATES: numpy array of (state_dim,)
                ACTIONS: numpy array of (,)
                LOG_PROBS: numpy array of (,)
                REWARDS: Float
        done: boolean
    """
    # initiate empty trajectories
    state_trajectory = []
    action_trajectory = []
    logprob_trajectory = []
    reward_trajectory = []

    # initiate state from environment
    state = actor_worker.env.reset()
    state[state == -1] = 0
    done = False

    for _ in range(max_steps):

        state_trajectory.append(state)  # state: numpy

        state = torch.unsqueeze(torch.from_numpy(state), 0)  # first dim batch_size: (1, action_dim)
        action, action_logprob = policy_worker.rollout(state)  # first dim batch_size: (1, action_dim)
        action = action.squeeze().cpu().numpy()
        action_logprob = action_logprob.squeeze().cpu().numpy()  # move back CPU

        state, reward, done, info = actor_worker.step(action)
        state[state == -1] = 0

        # current period
        action_trajectory.append(action)
        logprob_trajectory.append(action_logprob)
        reward_trajectory.append(reward)  # this might need to customize depending on done

        if done:
            logging.info("trajectory done info: {}".format(info))
            break

    return (state_trajectory, action_trajectory, logprob_trajectory, reward_trajectory), done


def batch_loss(episode, device, **params):
    """

    Args:
        episode: tuple of tensors: (states, behavior_actions, behavior_logprob, rewards, masks)
        params: gamma, gae_lambda, eps_clip

    Returns:
        loss (torch.Tensor [T,])
        entropy (torch.Tensor [T,])
        values (torch.Tensor [T,])

    """
    states, behavior_actions, behavior_logprob, rewards, masks = (item.to(device) for item in
                                                                  episode)  # unpack episode: tensors, on cpu

    action_logits, values = policy_worker.analyze(states)  # note states length [T+1]: tensors, with grads, on cuda
    action_logits = action_logits[:-1]  # action_logits: [T]
    action_dist = Categorical(logits=action_logits)  # categorical can't be moved to device

    values = values.squeeze()  # squeeze the batch_size dimension to (T,)

    # with torch.no_grad():
    #     advantages = gae_trace(
    #         rewards, values, masks, gamma=params.get('gamma', GAMMA), gae_lambda=params.get('gae_lambda', LAMBDA)
    #         )  # on cpu
    advantages = gae_trace(
        rewards, values, masks, gamma=params.get('gamma', GAMMA), gae_lambda=params.get('gae_lambda', LAMBDA)
    ).detach()

    action_logprob = action_dist.log_prob(behavior_actions)  # cuda

    importance_sampling = torch.exp(action_logprob - behavior_logprob)  # torch

    surrogate_1 = importance_sampling * advantages
    surrogate_2 = torch.clamp(importance_sampling,
                              1 - params.get('eps_clip', EPSILON_CLIP), 1 + params.get('eps_clip', EPSILON_CLIP))\
                  * advantages

    entropy = action_dist.entropy()
    loss = - torch.min(surrogate_1, surrogate_2) + \
           params.get('c1', C1) * MSE(rewards + params.get('gamma', GAMMA) * values.detach()[1:], values[:-1]) \
           - params.get('c2', C2) * entropy
    wandb.log({'advantage': advantages.mean()})
    return loss, entropy, values


def train_step(episode_, optimizer: torch.optim.Optimizer, device, **loss_param_dict):
    """
    Update gradients by forward and backward pass on the loss function, using episode_ data

    Args:
        episode_ (same as in batch_loss): tuple of tensors: (states, behavior_actions, behavior_logprob, rewards, masks)
        optimizer: torch.optim
        device (str): 'cuda' or 'cpu'
        **loss_param_dict: keyword params to be passed into batch_loss as **params

    Returns:
        batch mean of loss, entropy, values
    """
    optimizer.zero_grad()
    loss, entropy, values = batch_loss(episode_, device, **loss_param_dict)
    loss.mean().backward()
    optimizer.step()
    return torch.mean(loss), torch.mean(entropy), torch.mean(values)


if __name__ == '__main__':

    wandb.init(project='gfootball-ppo', entity='peter_gao', name=f'run_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    buffer = Buffer(max_buffer_size=2048, batch_size=MAX_BATCH_SIZE)  # initiate buffer

    # INITIALIZE OPTIMIZER
    adam_optimizer = torch.optim.Adam([
        {'params': policy_worker.net[0].parameters(), 'lr': LR_ACTOR},
        {'params': policy_worker.net[1].parameters(), 'lr': LR_CRITIC}])

    # for step in range(TRAIN_EPISODES):
    for episode in range(10_000):

        # generate new trajectory until buffer has enough data to pop
        while buffer.size <= buffer.batch_size:
            trajectory, done = simulate_trajectory(max_steps=MAX_TRAJECTORY_STEP)  # note returns trajectory, done
            buffer.put(trajectory=trajectory)

            # ========log reward==============
            if done:
                reward_trajectory = trajectory[-1]
                v = 0
                while len(reward_trajectory) > 0:
                    v += reward_trajectory.pop(-1) * GAMMA
            wandb.log({"discounted sum of rewards": v})

        batch_data = buffer.pop()  # fetch batch data
        [data.to(DEVICE) for data in batch_data]  # move to gpu if available

        loss_mean, entropy_mean, value_mean = [float(_) for _ in
                                               train_step(batch_data, optimizer=adam_optimizer, device=DEVICE)]
        metrics_names = ['episode', 'loss_mean', 'entropy_mean', 'value_mean']

        # if episode % 10 == 0:

        episode_result_dict = dict(zip(metrics_names, [episode, loss_mean, entropy_mean, value_mean]))
        # for name in episode_result_dict:
        #     logging.info(name, ":   ", episode_result_dict[name])

        # wandb
        wandb.log(episode_result_dict)

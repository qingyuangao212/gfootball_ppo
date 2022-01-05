from actor_worker import FootballActorWorker
from policy_worker import PpoPolicyWorker
from typing import Optional
from gfootball.env import create_environment
import torch
import torch.functional as F
from utils import *

actor_worker = FootballActorWorker()
policy_worker = PpoPolicyWorker(actor_worker.get_action_dim, actor_worker.get_state_dim)

SIMPLE_CONFIG = dict(
    env_name="academy_3_vs_1_with_keeper",
    number_of_left_players_agent_controls=3,
    number_of_right_players_agent_controls=0,
    representation="simple115v2"
)

MAX_BATCH_SIZE = 10_000
MAX_EPISODE_STEP = 1_000
GAMMA = 0.99
LAMBDA = 0.9

LR_ACTOR = 5e-4
LR_CRITIC = 5e-4


ADAM_OPTIM = torch.optim.Adam([
    {'params': policy_worker.actor.parameters(), 'lr': LR_ACTOR},
    {'params': policy_worker.critic.parameters(), 'lr': LR_CRITIC}
])



@numpy_wrapper
def generate_trajectory(max_episode_step: int = MAX_EPISODE_STEP):
    """

    :param max_episode_step:
    :return: state_trajectory, action_trajectory, action_prob_trajectory, reward_trajectory: all lists
             done: boolean
    """
    state_trajectory = []
    action_trajectory = []
    action_prob_trajectory = []
    reward_trajectory = []

    state = actor_worker.env.reset()
    state_trajectory.append(state)

    for step in range(max_episode_step):

        action, action_prob = policy_worker.rollout(state)  # action_prob is scalar
        state, reward, done, info = actor_worker.step(action)  # action may need some transformation

        # current period
        action_trajectory.append(action)
        action_prob_trajectory.append(action_prob)
        reward_trajectory.append(reward)  # this might need to customize depending on done

        # next period
        state_trajectory.append(state)  # state_trajectory length is 1 more than the others for s'
        if done:
            break

    return (state_trajectory, action_trajectory, action_prob_trajectory, reward_trajectory), done


def trajectory_loss(trajectory, params):
    """
    assume trajectory is done
    """
    states, behavior_actions, behavior_probs, rewards = trajectory      # all detached
    action_prob_dist, values = policy_worker.analyze(states)          # tensors with grads

    T = len(rewards)

    assert len(values) == T + 1
    advantages = torch.zeros(T + 1)
    for i in range(T)[::-1]:
        delta = rewards[i] + params['gamma'] * values.clone().detach()[i + 1] - values[i]       # detach V_prime, delta grad from V
        advantages[i] = delta + params['gae_lambda'] * params['gamma'] * advantages[i + 1]     # detach advantages

    advantages = torch.stack(advantages[:-1]).detach()

    pi_theta = torch.stack([action_prob_dist[i][behavior_actions[i]] for i in range(T)])
    pi_theta_old = torch.stack(behavior_probs)
    ratios = pi_theta / pi_theta_old        # torch

    entropy = -(F.log(action_prob_dist) * action_prob_dist).sum(1, keepdim=True)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - params['eps_clip'], 1 + params['eps_clip']) * advantages

    loss = -torch.min(surr1, surr2) + params['c1'] * torch.nn.MSELoss()(rewards + params['gamma'] * values[:-1], values.detach()[1:]) - params['c2'] * entropy
    loss.mean.backward()
    return loss


def train_step(optimizer=ADAM_OPTIM, params=SIMPLE_CONFIG, max_batch_size=MAX_BATCH_SIZE):
    optimizer.zero_grad()
    sample_length = 0
    while sample_length < max_batch_size:
        trajectory = generate_trajectory()
        loss = trajectory_loss(trajectory, params)
        loss.mean().backward()

    optimizer.step()
    return loss

def train()

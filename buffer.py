from actor_worker import FootballActorWorker
from policy_worker import PpoPolicyWorker
from typing import Optional
from gfootball.env import create_environment
import torch
from utils import *

MAX_EPISODE_LENGTH = 10_000

actor_worker = FootballActorWorker()
policy_worker = PpoPolicyWorker(actor_worker.get_action_dim, actor_worker.get_state_dim)

class Buffer:

    def __init__(self):
        self.



@numpy_wrapper
def generate_sample(max_episode_LENGTH: int = MAX_EPISODE_LENGTH):
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

        action, action_prob = policy_worker.rollout(state)
        state, reward, done, info = actor_worker.step(action)  # action may need some transformation

        # current period
        action_trajectory.append(action)
        action_prob_trajectory.append(action_prob)
        reward_trajectory.append(reward)  # this might need to customize depending on done

        # next period
        state_trajectory.append(state)  # state_trajectory length is 1 more than the others for s'

        if done:
            break

    return state_trajectory, action_trajectory, action_prob_trajectory, reward_trajectory, done


def generate_batch_samples(sample_size=64):
    """This can use Parallelism?"""
    return [generate_sample() for i in range(sample_size)]


GAMMA = 0.99
LAMBDA = 0.9


def train_batch(states, behavior_actions, behavior_probs, rewards, state_primes):
    """
    Fixed batch size
    No need for done since we recorded state_primes
    Each input should be np.ndarrays with first dimension denoting batch location (BATCH_SIZE, ...)
    """
    target_probs, values = policy_worker.analyze(torch.tensor(states))


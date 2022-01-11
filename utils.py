import time
import logging

import torch
import numpy as np



def test_wrapper(func):
    def wrap(*args, **kwargs):
        print(
            f"\nExecuting function {func.__name__} with args {args if len(args) > 0 else None} and kwargs {kwargs if len(kwargs) > 0 else None} ")
        print("=" * 40)
        start = time.time()
        result = func(*args, **kwargs)
        print("Time taken: {:2f}\n".format(time.time() - start))
        return result

    return wrap


def numpy_wrapper(func):
    def wrap(*args, **kwargs):
        result = func(*args, **kwargs)
        return tuple(np.array(r) for r in result)

    return wrap


def tensor_wrapper(func):
    def wrap(*args, **kwargs):
        result = func(*args, **kwargs)
        return tuple(torch.tensor(r) for r in result)

    return wrap


def gae(values, rewards, gae_lambda, gamma):
    """
    :param values: value trajectory, length T+1
    :param rewards: reward trajectory, length T
    :param gae_lambda: param as in TD(LAMBDA); controls bias-variance tradeoff
    :param gamma: discount rate
    :return: GAE values of length T (for each time point in the trajectory)
    """
    T = len(rewards)
    assert len(values) == T + 1
    advantage = torch.zeros(T + 1)
    for i in range(T)[::-1]:  # or use "reversed()"
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        advantage[i] = delta + gae_lambda * gamma * advantage[i + 1]

    return advantage[:-1]


def print_env_details(env):
    print(env.action_space.n)
    print(env.observation_space.high, env.observation_space.low)
    env.reset()

    observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
    print(reward)
    print(env.reward_range)


class Buffer:
    """
    Stores buffer data for state, action, action_prob, reward, done
    """

    def __init__(self, max_buffer_size: int = 2048, batch_size: int = 1024):

        self._MaxBufferSize = max_buffer_size
        self.state_buffer = []
        self.action_buffer = []
        self.action_prob_buffer = []
        self.reward_buffer = []
        self.batch_size = batch_size
        self.mask_buffer = []  # 0,1 denoting trajectory done

    def pop(self):
        """
        Pop batch_size sample  (state, action, action_prob, reward, mask_buffer) from buffer.
        Sample size is batch_size + 1 for states (for the purpose of evaluating advantage at T )
            and batch_size for all others

        return:
            Sample ( Tuple[states, actions, action_logprobs, rewards] ):
            Each element is list of length T the following dtype and dimension:
                STATES (T+1): numpy array of (state_dim,)
                ACTIONS: numpy array of (,)
                LOG_PROBS: numpy array of (,)
                REWARDS: Float
        """

        sample = (
            torch.from_numpy(np.stack(self.state_buffer[:self.batch_size + 1])),
            torch.from_numpy(np.stack(self.action_buffer[:self.batch_size])),
            torch.from_numpy(np.stack(self.action_prob_buffer[:self.batch_size])),
            torch.from_numpy(np.stack(self.reward_buffer[:self.batch_size])),
            torch.from_numpy(np.stack(self.mask_buffer[:self.batch_size]))
        )

        self.state_buffer = self.state_buffer[self.batch_size:]  # note here not batch_size+1
        self.action_buffer = self.action_buffer[self.batch_size:]
        self.action_prob_buffer = self.action_prob_buffer[self.batch_size:]
        self.reward_buffer = self.reward_buffer[self.batch_size:]
        self.mask_buffer = self.mask_buffer[self.batch_size:]

        # for debug
        try:
            assert len(sample[0]) == self.batch_size + 1
            assert len(sample[1]) == self.batch_size
            assert len(sample[2]) == self.batch_size
            assert len(sample[3]) == self.batch_size
            assert len(sample[4]) == self.batch_size
        except AssertionError:
            logging.error("Sample Length Mismatch: {}".format((len(s) for s in sample)))

        return sample

    def put(self, trajectory):

        """
        Put a trajectory to buffer by attaching each element (list) in the trajectory to its relevant buffer (list)
            For instance: state_trajectory -> state_buffer

        For mask_buffer, data is generated automatically by attaching [0, 0, ..., 0, 0, 1] to self.mask_buffer

        Args:
            trajectory ( Tuple[states, actions, action_logprobs, rewards] ):
            Each element is list of length T the following dtype and dimension:
                STATES: numpy array of (state_dim,)
                ACTIONS: numpy array of (,)
                LOG_PROBS: numpy array of (,)
                REWARDS: Float

        Returns:
            None
        """

        trajectory_length = len(trajectory[3])
        if trajectory_length + len(self.mask_buffer) > self._MaxBufferSize:
            raise Warning("Max Buffer Size reached, buffer not updated!")
        else:
            self.state_buffer += trajectory[0]
            self.action_buffer += trajectory[1]
            self.action_prob_buffer += trajectory[2]
            self.reward_buffer += trajectory[3]
            self.mask_buffer += [0] * (trajectory_length - 1) + [1]

    @property
    def size(self):
        return len(self.mask_buffer)


def gae_trace(rewards, values, done, gamma=torch.tensor(0.99), gae_lambda=torch.tensor(0.97)):
    """
    Calculate GAE (torch.tensor) for an episode (sequence of rewards and values)

    An episode is okay to contain multiple sequences (which can be identified with done)

    Note: Value Sequence needs to have 1 more length than rewards (need V' for last period)
          Does not allow calculating GAE for batches: first dimension is T+1 for values and T for other inputs
          For batch operation, flatten into one sequence first.
    Args:
        rewards (torch.Tensor):
            Reward of shape [T,]
        values (torch.Tensor):
            Values of shape [T+1,]
        done (torch.Tensor[bool]):
            masks of shape [T,]
        gamma (torch.Tensor or Float, Optional):
            Discount Rate; usually 0< gamma <1

        gae_lambda (torch.Tensor or Float, Optional):
            GAE weights; usually 0< gamma <1
    Returns:
        advantage (torch.Tensor):
            GAE of length T
    """
    episode_length = len(rewards)
    assert episode_length == len(values) - 1
    delta = rewards + gamma * values[1:] * (1 - done) - values[:-1]  # if done, then delta = reward
    # gae = torch.zeros_like(rewards[0])
    gae = 0
    advantage = torch.zeros_like(rewards)  # initiate with fixed length for efficiency
    m = gamma * gae_lambda * (1 - done)  # if done, m = 0
    step = episode_length - 1
    while step >= 0:
        gae = delta[step] + m[step] * gae  # if done, gae = delta = reward
        advantage[step] = gae
        step -= 1
    return advantage


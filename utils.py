import time
import torch
import numpy as np

def test_wrapper(func):
    def wrap(*args, **kwargs):
        print(f"\nExecuting function {func.__name__} with args {args if len(args)>0 else None} and kwargs {kwargs if len(kwargs)>0 else None} ")
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
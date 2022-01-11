from gfootball.env import create_environment
from utils import tensor_wrapper
from config import ENV_CONFIG
"""
Generate samples
"""

class FootballActorWorker:
    def __init__(self, config):
        """

        :param config:
        """
        self.env = create_environment(**config)

    def step(self, action):
        """discard info"""
        return self.env.step(action)

    @property
    def action_dim(self):
        return self.env.action_space.n
    @property
    def state_dim(self):
        return self.env.observation_space.shape[0]


if __name__ == '__main__':
    actor_worker = FootballActorWorker(config=ENV_CONFIG)
    print(actor_worker.state_dim)
    state = actor_worker.env.reset()
    print(state.shape)
    print(actor_worker.action_dim)

    obs, reward, done, info = actor_worker.step(actor_worker.env.action_space.sample())
    print(reward.shape)
    print(reward)

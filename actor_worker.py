from gfootball.env import football_env

"""
Generate samples
"""
simple_config = dict(env_name="1_vs_1_easy",
                     rewards='scoring,checkpoints',
                     number_of_left_players_agent_controls=1,
                     number_of_right_players_agent_controls=0,
                     representation="simple115v2"
                     )


class FootballActorWorker:
    def __init__(self, config=simple_config):
        """

        :param config:
        """
        self.env = football_env.create_environment(**config)

    def step(self, action):
        return self.env.step()

    @property
    def get_action_dim(self):
        return self.env.action_space_n

    @property
    def get_state_dim(self):
        return self.env.observation_space.shape[0]

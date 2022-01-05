from gfootball.env import config
from gfootball.env import football_env


config = {"save_video": False,
               "scenario_name": "11_vs_11_kaggle",
               "running_in_notebook": True}
# env = football_env.FootballEnv(config)
# env.reset()
# observation, reward, done, info = env.step()
# print(observation)
# print(reward)

env = football_env.create_environment()
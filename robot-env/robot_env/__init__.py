from gym.envs.registration import register

register(
    id='robot-env-v0',
    entry_point='robot_env.envs:RobotEnv',
)
# register(
#     id='robot-env-extrahard-v0',
#     entry_point='robot_extrahard_env.envs:RobotExtraHardEnv',
# )
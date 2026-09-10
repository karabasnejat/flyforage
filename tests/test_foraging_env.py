from flyforage.envs.foraging import FlyForageEnv


def test_environment_reset_and_step():
    env = FlyForageEnv()
    observation, info = env.reset(seed=42)
    assert observation.shape == (6,)
    assert info["food_collected"] == 0

    next_observation, reward, terminated, truncated, info = env.step(3)
    assert next_observation.shape == (6,)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert info["steps"] == 1

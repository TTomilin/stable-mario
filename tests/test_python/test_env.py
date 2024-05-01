import os

import pytest

import stable_retro


@pytest.fixture(
    params=[
        os.path.splitext(rom)[0]
        for rom in os.listdir(os.path.join(os.path.dirname(__file__), "../roms"))
    ],
)
def generate_test_env(request):
    import stable_retro.data

    path = os.path.join(os.path.dirname(__file__), "../roms")

    get_file_path_fn = stable_retro.data.get_file_path
    get_romfile_path_fn = stable_retro.data.get_romfile_path

    stable_retro.data.get_file_path = lambda game, file, *args, **kwargs: os.path.join(
        path,
        file,
    )
    stable_retro.data.get_romfile_path = lambda game, *args, **kwargs: [
        os.path.join(path, rom) for rom in os.listdir(path) if rom.startswith(game)
    ][0]

    created_env = []

    def create(state=stable_retro.State.NONE, *args, **kwargs):
        env = stable_retro.make(game=request.param, state=state, *args, **kwargs)
        created_env.append(env)  # noqa: F821
        return env

    yield create

    created_env[0].close()
    del created_env

    stable_retro.data.get_file_path = get_file_path_fn
    stable_retro.data.get_romfile_path = get_romfile_path_fn


def test_env_create(generate_test_env):
    json_path = os.path.join(os.path.dirname(__file__), "../dummy.json")
    assert generate_test_env(info=json_path, scenario=json_path)


@pytest.mark.parametrize("obs_type", [stable_retro.Observations.IMAGE, stable_retro.Observations.RAM])
def test_env_basic(obs_type, generate_test_env):
    json_path = os.path.join(os.path.dirname(__file__), "../dummy.json")

    env = generate_test_env(info=json_path, scenario=json_path, obs_type=obs_type)

    obs, info = env.reset()
    assert obs in env.observation_space
    assert isinstance(info, dict)

    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs in env.observation_space

    assert isinstance(rew, float)
    assert rew == 0

    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    assert terminated is False
    assert truncated is False

    assert isinstance(info, dict)


def test_env_data(generate_test_env):
    json_path = os.path.join(os.path.dirname(__file__), "../dummy.json")

    env = generate_test_env(info=json_path, scenario=json_path)
    assert isinstance(env.data[env.system], int)

    env.data["foo"] = 1
    assert env.data["foo"] == 1

    env.reset()

    with pytest.raises(KeyError):
        val = env.data["foo"]
        assert val

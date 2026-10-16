import pytest

from dotabotai.env import ACTIONS, SyntheticLaneEnv


def test_same_seed_replays_same_transition_sequence() -> None:
    env_a = SyntheticLaneEnv(seed=1)
    env_b = SyntheticLaneEnv(seed=2)
    obs_a = env_a.reset(seed=987)
    obs_b = env_b.reset(seed=987)
    assert obs_a == obs_b
    for action in [0, 1, 0, 3, 2, 1, 3, 0]:
        if env_a.done or env_b.done:
            break
        transition_a = env_a.step(action)
        transition_b = env_b.step(action)
        assert transition_a == transition_b


def test_invalid_action_is_rejected() -> None:
    env = SyntheticLaneEnv(seed=3)
    env.reset()
    with pytest.raises(ValueError):
        env.step(len(ACTIONS))
    with pytest.raises(ValueError):
        env.step(-1)


def test_step_after_terminal_requires_reset() -> None:
    env = SyntheticLaneEnv(seed=4, max_steps=1)
    env.reset()
    _, _, done, _ = env.step(0)
    assert done is True
    with pytest.raises(RuntimeError):
        env.step(0)

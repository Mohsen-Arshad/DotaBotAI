import json

import pytest

from dotabotai.agent import QLearningAgent, encode_state
from dotabotai.env import SyntheticLaneEnv


def test_checkpoint_roundtrip_preserves_policy(tmp_path) -> None:
    env = SyntheticLaneEnv(seed=12)
    agent = QLearningAgent(seed=4)
    obs = env.reset()
    next_obs, reward, done, _ = env.step(1)
    agent.learn(obs, 1, reward, next_obs, done)
    agent.finish_episode()
    path = tmp_path / "model.json"
    agent.save(path)

    restored = QLearningAgent(seed=100)
    restored.load(path)
    assert restored.q == agent.q
    assert restored.episodes_trained == agent.episodes_trained
    assert restored.total_updates == agent.total_updates
    assert restored.epsilon == agent.epsilon


def test_evaluation_action_does_not_explore_or_mutate_q() -> None:
    agent = QLearningAgent(seed=3, epsilon=1.0)
    obs = {"hp": 90, "enemy_distance": 600, "creep_hp": 20, "enemy_hp": 100}
    agent.q[encode_state(obs)] = [0.0, 5.0, 1.0, 2.0]
    before = json.dumps(agent.q, sort_keys=True)
    actions = [agent.choose_action(obs, explore=False) for _ in range(10)]
    assert actions == [1] * 10
    assert json.dumps(agent.q, sort_keys=True) == before


def test_low_hp_safety_mask_is_applied() -> None:
    agent = QLearningAgent(seed=5, epsilon=1.0)
    obs = {"hp": 10, "enemy_distance": 260, "creep_hp": 20, "enemy_hp": 100}
    assert all(agent.choose_action(obs, explore=True) in (0, 2) for _ in range(100))


def test_incompatible_checkpoint_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format_version": 999}), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        QLearningAgent().load(path)


def test_greedy_action_on_unseen_state_does_not_mutate_checkpoint() -> None:
    agent = QLearningAgent(seed=8)
    obs = {"hp": 90, "enemy_distance": 600, "creep_hp": 95, "enemy_hp": 90}
    before = dict(agent.q)
    first = agent.choose_action(obs, explore=False)
    second = agent.choose_action(obs, explore=False)
    assert first == second
    assert agent.q == before


def test_checkpoint_resume_preserves_exploration_rng(tmp_path) -> None:
    agent = QLearningAgent(seed=41, epsilon=1.0)
    path = tmp_path / "resume.json"
    agent.save(path)
    expected = [agent.choose_action({"hp": 90, "enemy_distance": 500, "creep_hp": 20, "enemy_hp": 50}, explore=True) for _ in range(20)]
    restored = QLearningAgent(seed=999)
    restored.load(path)
    actual = [restored.choose_action({"hp": 90, "enemy_distance": 500, "creep_hp": 20, "enemy_hp": 50}, explore=True) for _ in range(20)]
    assert actual == expected

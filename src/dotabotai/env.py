"""Deterministic-seed, synthetic lane-control environment.

This is a small RL validation task, not a Dota 2 emulator and not a live-game adapter.
The environment is intentionally dependency-free so its learning loop can be tested locally.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

ACTIONS: tuple[str, ...] = ("wait", "last_hit", "retreat", "harass")
ENVIRONMENT_ID = "synthetic-lane-v1"


@dataclass
class LaneState:
    hp: int = 100
    enemy_distance: int = 550
    creep_hp: int = 60
    enemy_hp: int = 100
    gold: int = 0
    last_hits: int = 0
    deaths: int = 0
    tick: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class SyntheticLaneEnv:
    """Small stochastic task with last-hit, survival, and lane-pressure trade-offs."""

    def __init__(self, seed: int = 0, max_steps: int = 70) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.rng = random.Random(self.seed)
        self.state = LaneState()
        self.done = False

    def reset(self, seed: int | None = None) -> dict[str, int]:
        if seed is not None:
            self.seed = int(seed)
            self.rng.seed(self.seed)
        self.state = LaneState(
            hp=100,
            enemy_distance=self.rng.randint(300, 650),
            creep_hp=self.rng.randint(20, 100),
            enemy_hp=100,
            gold=0,
            last_hits=0,
            deaths=0,
            tick=0,
        )
        self.done = False
        return self.observe()

    def observe(self) -> dict[str, int]:
        return self.state.as_dict()

    def step(self, action: int) -> tuple[dict[str, int], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is finished; call reset() before step().")
        if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < len(ACTIONS):
            raise ValueError(f"action must be an integer in [0, {len(ACTIONS) - 1}]")

        s = self.state
        name = ACTIONS[action]
        reward = -0.015
        info: dict[str, Any] = {
            "action_name": name,
            "last_hit": False,
            "death": False,
            "creep_missed": False,
        }

        if name == "wait":
            s.creep_hp -= self.rng.randint(8, 18)
            if s.creep_hp <= 0:
                reward -= 0.35
                info["creep_missed"] = True
                self._spawn_creep()
            if self.rng.random() < 0.24:
                s.enemy_distance -= self.rng.randint(10, 35)

        elif name == "last_hit":
            if s.creep_hp <= 27:
                if self.rng.random() < 0.92:
                    reward += 1.25
                    s.gold += 35
                    s.last_hits += 1
                    info["last_hit"] = True
                    self._spawn_creep()
                else:
                    reward -= 0.15
                    s.creep_hp -= self.rng.randint(8, 16)
            else:
                reward -= 0.30
                s.creep_hp -= self.rng.randint(8, 16)
            if s.enemy_distance < 370:
                s.hp -= self.rng.randint(3, 9)

        elif name == "retreat":
            if s.hp < 65 or s.enemy_distance < 390:
                reward += 0.18
            else:
                reward -= 0.09
            s.enemy_distance = min(700, s.enemy_distance + self.rng.randint(70, 115))
            if s.enemy_distance >= 460:
                s.hp = min(100, s.hp + self.rng.randint(2, 5))
            s.creep_hp -= self.rng.randint(3, 10)
            if s.creep_hp <= 0:
                reward -= 0.18
                info["creep_missed"] = True
                self._spawn_creep()

        elif name == "harass":
            if s.enemy_distance <= 500 and s.hp >= 38:
                reward += 0.20
                s.enemy_hp -= self.rng.randint(8, 18)
                if s.enemy_hp <= 0:
                    reward += 1.0
                    s.enemy_hp = 100
                    s.enemy_distance = min(650, s.enemy_distance + 70)
                if s.enemy_distance < 390:
                    s.hp -= self.rng.randint(5, 11)
            else:
                reward -= 0.22
            s.creep_hp -= self.rng.randint(4, 12)
            if s.creep_hp <= 0:
                reward -= 0.20
                info["creep_missed"] = True
                self._spawn_creep()

        # Exogenous lane movement. Reset(seed=...) makes runs reproducible.
        if self.rng.random() < 0.16:
            s.enemy_distance = max(240, s.enemy_distance - self.rng.randint(5, 25))
        elif self.rng.random() < 0.18:
            s.enemy_distance = min(700, s.enemy_distance + self.rng.randint(5, 20))

        s.tick += 1
        if s.hp <= 0:
            s.hp = 0
            s.deaths += 1
            reward -= 2.0
            info["death"] = True
            self.done = True
        elif s.tick >= self.max_steps:
            self.done = True

        return self.observe(), float(reward), self.done, info

    def _spawn_creep(self) -> None:
        self.state.creep_hp = self.rng.randint(45, 100)

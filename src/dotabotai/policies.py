"""Reference policies used to keep evaluation honest."""
from __future__ import annotations

import random

from .env import ACTIONS


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, obs: dict[str, int]) -> int:
        actions = list(range(len(ACTIONS)))
        if obs["hp"] <= 20:
            actions = [0, 2]
        return self.rng.choice(actions)


class HeuristicPolicy:
    """Fixed rule baseline; not learned from reward feedback."""

    name = "heuristic"

    def choose_action(self, obs: dict[str, int]) -> int:
        if obs["hp"] < 50 or obs["enemy_distance"] < 330:
            return 2  # retreat
        if obs["creep_hp"] <= 27:
            return 1  # last hit
        if obs["enemy_distance"] <= 500 and obs["hp"] >= 65:
            return 3  # harass
        return 0  # wait

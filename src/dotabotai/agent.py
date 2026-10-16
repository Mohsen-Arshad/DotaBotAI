"""Tabular Q-learning agent with explicit state schema and atomic checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable

from .env import ACTIONS

MODEL_FORMAT_VERSION = 3
STATE_ENCODING = "hp4|distance3|creep3|enemy_hp3-v1"


def encode_state(obs: dict[str, int]) -> str:
    """Discretize the state features that directly affect this toy environment."""
    hp = 0 if obs["hp"] < 25 else 1 if obs["hp"] < 50 else 2 if obs["hp"] < 75 else 3
    distance = 0 if obs["enemy_distance"] < 330 else 1 if obs["enemy_distance"] < 500 else 2
    creep = 0 if obs["creep_hp"] <= 27 else 1 if obs["creep_hp"] <= 60 else 2
    enemy_hp = 0 if obs["enemy_hp"] <= 30 else 1 if obs["enemy_hp"] <= 65 else 2
    return f"h{hp}|d{distance}|c{creep}|e{enemy_hp}"


class QLearningAgent:
    def __init__(
        self,
        seed: int = 7,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 0.50,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 <= epsilon_min <= epsilon <= 1:
            raise ValueError("require 0 <= epsilon_min <= epsilon <= 1")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = random.Random(self.seed)
        self.q: dict[str, list[float]] = {}
        self.episodes_trained = 0
        self.total_updates = 0

    def _row(self, key: str) -> list[float]:
        if key not in self.q:
            self.q[key] = [0.0] * len(ACTIONS)
        return self.q[key]

    @staticmethod
    def valid_actions(obs: dict[str, int], candidate_actions: Iterable[int] | None = None) -> list[int]:
        candidates = list(range(len(ACTIONS))) if candidate_actions is None else list(candidate_actions)
        if not candidates:
            candidates = [0]
        if any(not isinstance(a, int) or a < 0 or a >= len(ACTIONS) for a in candidates):
            raise ValueError("candidate_actions contains an invalid action")
        if obs["hp"] <= 20:
            safe = [a for a in candidates if a in (0, 2)]
            if safe:
                return safe
        return candidates

    def choose_action(
        self,
        obs: dict[str, int],
        allowed_actions: Iterable[int] | None = None,
        explore: bool = True,
        decision_seed: int | None = None,
        tie_rng: random.Random | None = None,
    ) -> int:
        choices = self.valid_actions(obs, allowed_actions)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(choices)
        key = encode_state(obs)
        # Evaluation must not add unseen states to the checkpoint. Deterministic
        # tie-breaking also makes a held-out episode independent of evaluation order.
        row = self.q.get(key, [0.0] * len(ACTIONS)) if not explore else self._row(key)
        best = max(row[a] for a in choices)
        tied = [a for a in choices if math.isclose(row[a], best, abs_tol=1e-12)]
        if explore:
            return self.rng.choice(tied)
        if tie_rng is not None:
            return tie_rng.choice(tied)
        # Seed-derived deterministic fallback avoids fixed action-index bias.
        tie_context = (
            f"{self.seed}|{decision_seed if decision_seed is not None else 0}|{key}|"
            f"{obs.get('tick', 0)}|{obs.get('last_hits', 0)}|{obs.get('gold', 0)}"
        ).encode("utf-8")
        digest = hashlib.blake2s(tie_context, digest_size=8).digest()
        return tied[int.from_bytes(digest, byteorder="big") % len(tied)]

    def learn(
        self,
        obs: dict[str, int],
        action: int,
        reward: float,
        next_obs: dict[str, int],
        done: bool,
    ) -> None:
        if action < 0 or action >= len(ACTIONS):
            raise ValueError("action index is invalid")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        current = self._row(encode_state(obs))
        future = self._row(encode_state(next_obs))
        next_actions = self.valid_actions(next_obs)
        next_best = 0.0 if done else max(future[a] for a in next_actions)
        target = reward + self.gamma * next_best
        current[action] += self.alpha * (target - current[action])
        self.total_updates += 1

    def finish_episode(self) -> None:
        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": MODEL_FORMAT_VERSION,
            "algorithm": "tabular_q_learning",
            "environment_id": "synthetic-lane-v1",
            "state_encoding": STATE_ENCODING,
            "actions": list(ACTIONS),
            "seed": self.seed,
            "hyperparameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "episodes_trained": self.episodes_trained,
            "total_updates": self.total_updates,
            "rng_state": self.rng.getstate(),
            "q_table": self.q,
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {
            "format_version": MODEL_FORMAT_VERSION,
            "environment_id": "synthetic-lane-v1",
            "state_encoding": STATE_ENCODING,
            "actions": list(ACTIONS),
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise ValueError(f"Model {field!r} is incompatible: expected {value!r}")
        raw_q = payload.get("q_table")
        if not isinstance(raw_q, dict):
            raise ValueError("Invalid checkpoint: q_table must be an object")
        q: dict[str, list[float]] = {}
        for key, values in raw_q.items():
            if not isinstance(values, list) or len(values) != len(ACTIONS):
                raise ValueError(f"Invalid Q row for state {key!r}")
            row = [float(value) for value in values]
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"Non-finite Q value in state {key!r}")
            q[str(key)] = row
        hp = payload.get("hyperparameters", {})
        alpha = float(hp.get("alpha", self.alpha))
        gamma = float(hp.get("gamma", self.gamma))
        epsilon = float(hp.get("epsilon", self.epsilon))
        epsilon_min = float(hp.get("epsilon_min", self.epsilon_min))
        epsilon_decay = float(hp.get("epsilon_decay", self.epsilon_decay))
        if not (math.isfinite(alpha) and 0 < alpha <= 1):
            raise ValueError("Invalid checkpoint hyperparameter alpha")
        if not (math.isfinite(gamma) and 0 <= gamma <= 1):
            raise ValueError("Invalid checkpoint hyperparameter gamma")
        if not (math.isfinite(epsilon) and 0 <= epsilon <= 1):
            raise ValueError("Invalid checkpoint hyperparameter epsilon")
        if not (math.isfinite(epsilon_min) and 0 <= epsilon_min <= epsilon):
            raise ValueError("Invalid checkpoint hyperparameter epsilon_min")
        if not (math.isfinite(epsilon_decay) and 0 < epsilon_decay <= 1):
            raise ValueError("Invalid checkpoint hyperparameter epsilon_decay")
        self.q = q
        self.seed = int(payload.get("seed", self.seed))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes_trained = int(payload.get("episodes_trained", 0))
        self.total_updates = int(payload.get("total_updates", 0))
        if self.episodes_trained < 0 or self.total_updates < 0:
            raise ValueError("Invalid checkpoint counters")
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            def to_tuple(value: object) -> object:
                if isinstance(value, list):
                    return tuple(to_tuple(item) for item in value)
                return value
            try:
                self.rng.setstate(to_tuple(rng_state))  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid checkpoint RNG state") from exc

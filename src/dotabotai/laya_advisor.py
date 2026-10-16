"""Optional Laya macro advisor. It does not execute game actions."""
from __future__ import annotations

from typing import Any


class LayaAdvisor:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.router: Any = None

    def _load(self) -> Any:
        if self.router is None:
            try:
                from laya import Router  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError("Laya is not installed. Run: pip install -e .[laya]") from exc
            self.router = Router(device=self.device)
        return self.router

    def advise(self, obs: dict[str, int]) -> dict[str, Any]:
        router = self._load()
        state = {
            "environment": "synthetic-lane-v1; not live Dota 2",
            "hero_hp_pct": obs["hp"],
            "enemy_distance": obs["enemy_distance"],
            "creep_hp": obs["creep_hp"],
            "enemy_hp": obs["enemy_hp"],
            "gold": obs["gold"],
            "last_hits": obs["last_hits"],
        }
        questions = {
            "plan": {
                "type": "choice",
                "instructions": "Choose the most appropriate high-level lane plan.",
                "criteria": {
                    "farm": "Prioritize a safe last hit when a creep is nearly dead.",
                    "retreat": "Prioritize preserving health when threatened or low on health.",
                    "harass": "Pressure the opponent only when health and distance are favorable.",
                    "balanced": "No clear high-level preference; let the learned policy decide.",
                },
            }
        }
        return router.predict(state, questions)

"""A bot that plays the CFR blueprint straight out of the table."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

from ..engine import Observation
from ..ml.abstraction import NUM_ACTIONS, legal_mask
from ..ml.cfr import Blueprint, observation_infoset
from .rule import EquityBot, ScriptedBot

DEFAULT_BLUEPRINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "blueprint.npz",
)


class BlueprintBot(ScriptedBot):
    """Look the situation up in the solved table; fall back when it is missing.

    The blueprint is solved heads-up over a bucketed abstraction, so a live
    multiway spot occasionally maps to an info-set that was never visited.  In
    that case the bot defers to the equity baseline rather than guessing.
    """

    def __init__(self, name: str, blueprint: Optional[Blueprint] = None,
                 path: Optional[str] = None, rng: Optional[random.Random] = None,
                 temperature: float = 1.0):
        super().__init__(name, rng)
        if blueprint is None and (path or os.path.exists(DEFAULT_BLUEPRINT)):
            blueprint = Blueprint.load(path or DEFAULT_BLUEPRINT)
        self.blueprint = blueprint
        self.temperature = temperature
        self.fallback = EquityBot(name + "-fallback", rng)
        self.lookups = 0
        self.misses = 0

    def strategy_for(self, obs: Observation, mask: np.ndarray) -> Optional[np.ndarray]:
        if self.blueprint is None:
            return None
        key = observation_infoset(obs)
        if key not in self.blueprint.strategy_sum:
            return None
        legal = [i for i in range(NUM_ACTIONS) if mask[i]]
        probs = self.blueprint.average_strategy(key, legal)
        if self.temperature != 1.0:
            probs = np.where(mask, probs, 0.0) ** (1.0 / max(1e-3, self.temperature))
        total = probs.sum()
        return probs / total if total > 0 else None

    def choose(self, obs, mask):
        self.lookups += 1
        probs = self.strategy_for(obs, mask)
        if probs is None:
            self.misses += 1
            return self.fallback.choose(obs, mask)
        return int(self.rng.choices(range(NUM_ACTIONS), weights=probs, k=1)[0])

    @property
    def hit_rate(self) -> float:
        return 1.0 - self.misses / max(1, self.lookups)

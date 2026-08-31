"""Hand-written opponents.

They exist for three reasons: they are the low difficulty settings, they are
the opponents the neural bot trains against (a self-play pool of only clones
learns a narrow strategy), and they are the yardstick every trained model is
measured against in the test-suite.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np

from ..engine import Action, Observation
from ..equity import fast_equity
from ..game import BaseAgent
from ..ml.abstraction import A_CALL, A_FOLD, NUM_ACTIONS, legal_mask, to_engine_action


class ScriptedBot(BaseAgent):
    """Base for bots that pick an abstract action index."""

    def __init__(self, name: str, rng: Optional[random.Random] = None):
        super().__init__(name)
        self.rng = rng or random.Random()

    def choose(self, obs: Observation, mask: np.ndarray) -> int:  # pragma: no cover
        raise NotImplementedError

    def act(self, obs: Observation) -> Action:
        mask = legal_mask(obs)
        index = self.choose(obs, mask)
        if not mask[index]:
            index = A_CALL
        return to_engine_action(obs, index)

    def _pick(self, mask: np.ndarray, weights: Sequence[float]) -> int:
        w = np.array(weights, dtype=np.float64) * mask
        if w.sum() <= 0:
            return A_CALL
        w = w / w.sum()
        return int(self.rng.choices(range(NUM_ACTIONS), weights=w, k=1)[0])


class RandomBot(ScriptedBot):
    """Uniform over legal actions — the floor of the difficulty ladder."""

    def choose(self, obs, mask):
        return self._pick(mask, np.ones(NUM_ACTIONS))


class CallingStation(ScriptedBot):
    """Calls too much, raises almost never.  The classic losing player."""

    def choose(self, obs, mask):
        return self._pick(mask, [0.05, 0.88, 0.04, 0.02, 0.01, 0.0, 0.0])


class TightRock(ScriptedBot):
    """Only plays big hands, but plays them straightforwardly."""

    def choose(self, obs, mask):
        eq = fast_equity(obs.hole, obs.board, max(1, obs.live_opponents), iters=200)
        if eq > 0.78:
            return self._pick(mask, [0.0, 0.35, 0.1, 0.25, 0.25, 0.05, 0.0])
        if eq > 0.6:
            return self._pick(mask, [0.0, 0.75, 0.15, 0.10, 0.0, 0.0, 0.0])
        if obs.to_call == 0:
            return A_CALL
        return A_FOLD if mask[A_FOLD] and eq < 0.5 else A_CALL


class LooseAggressive(ScriptedBot):
    """Raises far too often; punishes anyone who folds too much."""

    def choose(self, obs, mask):
        eq = fast_equity(obs.hole, obs.board, max(1, obs.live_opponents), iters=200)
        if eq > 0.55 or self.rng.random() < 0.35:
            return self._pick(mask, [0.0, 0.30, 0.20, 0.25, 0.20, 0.05, 0.0])
        return self._pick(mask, [0.45, 0.45, 0.05, 0.05, 0.0, 0.0, 0.0])


class EquityBot(ScriptedBot):
    """A solid, honest baseline: pot odds, position and a little bluffing.

    Beats every other rule bot here and is the reference the learned bots have
    to clear before they can claim to be any good.
    """

    def __init__(self, name: str, rng: Optional[random.Random] = None,
                 bluff: float = 0.12, iters: int = 400):
        super().__init__(name, rng)
        self.bluff = bluff
        self.iters = iters
        self._cache: dict = {}

    def equity(self, obs: Observation) -> float:
        key = (tuple(sorted(obs.hole)), tuple(sorted(obs.board)), max(1, obs.live_opponents))
        cached = self._cache.get(key)
        if cached is None:
            if len(self._cache) > 4000:
                self._cache.clear()
            cached = fast_equity(obs.hole, obs.board,
                                 max(1, obs.live_opponents), iters=self.iters)
            self._cache[key] = cached
        return cached

    def choose(self, obs, mask):
        eq = self.equity(obs)
        odds = obs.pot_odds
        position = obs.seat != ((obs.button + 1) % obs.num_players)

        if eq > 0.82:
            return self._pick(mask, [0.0, 0.30, 0.05, 0.25, 0.30, 0.10, 0.0])
        if eq > 0.66:
            return self._pick(mask, [0.0, 0.45, 0.20, 0.25, 0.10, 0.0, 0.0])
        if eq > 0.52:
            if obs.to_call == 0:
                return self._pick(mask, [0.0, 0.55, 0.30, 0.15, 0.0, 0.0, 0.0])
            return self._pick(mask, [0.10, 0.80, 0.10, 0.0, 0.0, 0.0, 0.0])
        if obs.to_call == 0:
            # Free card, or a cheap bluff in position.
            if position and self.rng.random() < self.bluff:
                return self._pick(mask, [0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.0])
            return A_CALL
        if eq > odds + 0.04:
            return self._pick(mask, [0.15, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0])
        return A_FOLD if mask[A_FOLD] else A_CALL


class HonestBot(EquityBot):
    """EquityBot with the bluffing switched off — a pure value player."""

    def __init__(self, name: str, rng: Optional[random.Random] = None):
        super().__init__(name, rng, bluff=0.0)

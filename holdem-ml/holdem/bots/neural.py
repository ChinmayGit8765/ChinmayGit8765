"""The learned bot: neural policy + CFR blueprint + live opponent exploitation.

Decision pipeline for a single spot:

1. Roll out equity (budget set by the difficulty level) and encode the state.
2. Ask the policy network for a distribution over the 7 abstract actions.
3. Blend in the CFR blueprint where it has a solved entry for this info-set.
4. **Exploit**: nudge that distribution using the opponent model — bluff more
   into someone who folds too much, value-bet thinner against a station, call
   down a habitual bluffer.  How hard it leans is the difficulty's
   ``exploit_weight``.
5. Apply the difficulty's aggression multiplier, temperature and blunder rate,
   then sample.

Between hands the bot updates its opponent models and, in adaptive mode, its
own difficulty, from a :class:`~holdem.ml.difficulty.SkillTracker` reading of
the human.
"""

from __future__ import annotations

import os
import random
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

from ..engine import Action, ActionRecord, HandResult, Observation
from ..equity import fast_equity
from ..game import BaseAgent
from ..ml.abstraction import (
    A_ALLIN,
    A_CALL,
    A_FOLD,
    NUM_ACTIONS,
    RAISE_FRACTIONS,
    legal_mask,
    raise_target,
    to_abstract,
    to_engine_action,
)
from ..ml.cfr import Blueprint, observation_infoset
from ..ml.difficulty import (
    AdaptiveDifficulty,
    Difficulty,
    SkillTracker,
    resolve_difficulty,
)
from ..ml.features import encode, last_aggressor
from ..ml.opponent import OpponentTracker
from ..ml.policy import PolicyValueNet, load_policy, mask_softmax

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"
)
DEFAULT_POLICY = os.path.join(MODEL_DIR, "policy.npz")
DEFAULT_BLUEPRINT = os.path.join(MODEL_DIR, "blueprint.npz")

RAISE_ACTIONS = tuple(RAISE_FRACTIONS) + (A_ALLIN,)

# Heuristic prior used before a policy is trained, keyed by equity band.
_PRIOR_BANDS = [
    (0.82, [0.00, 0.30, 0.05, 0.25, 0.30, 0.10, 0.00]),
    (0.66, [0.00, 0.45, 0.20, 0.25, 0.10, 0.00, 0.00]),
    (0.52, [0.05, 0.65, 0.20, 0.10, 0.00, 0.00, 0.00]),
    (0.40, [0.30, 0.55, 0.10, 0.05, 0.00, 0.00, 0.00]),
    (0.00, [0.55, 0.32, 0.09, 0.04, 0.00, 0.00, 0.00]),
]


class NeuralBot(BaseAgent):
    def __init__(
        self,
        name: str,
        policy: Optional[PolicyValueNet] = None,
        blueprint: Optional[Blueprint] = None,
        difficulty="regular",
        rng: Optional[random.Random] = None,
        tracker: Optional[OpponentTracker] = None,
        study: Optional[str] = None,
        load_defaults: bool = True,
    ):
        """``study`` names the player this bot should track for adaptive play."""
        super().__init__(name)
        self.rng = rng or random.Random()
        self.np_rng = np.random.default_rng(self.rng.randrange(1 << 31))

        if policy is None and load_defaults and os.path.exists(DEFAULT_POLICY):
            policy = load_policy(DEFAULT_POLICY)
        if blueprint is None and load_defaults and os.path.exists(DEFAULT_BLUEPRINT):
            blueprint = Blueprint.load(DEFAULT_BLUEPRINT)
        self.policy = policy
        self.blueprint = blueprint

        self.adaptive_ctl: Optional[AdaptiveDifficulty] = None
        if isinstance(difficulty, str) and difficulty.startswith("adapt"):
            self.adaptive_ctl = AdaptiveDifficulty()
            self.difficulty = self.adaptive_ctl.current()
        else:
            self.difficulty = resolve_difficulty(difficulty)

        self.tracker = tracker or OpponentTracker(rng=self.np_rng)
        self.study = study
        self.skill = SkillTracker(study) if study else None
        self._equity_cache: Dict[tuple, float] = {}
        self.blueprint_hits = 0
        self.decisions = 0

    # -- helpers -------------------------------------------------------------

    def _equity(self, obs: Observation, opponents: int, iters: int) -> float:
        key = (tuple(sorted(obs.hole)), tuple(sorted(obs.board)), opponents)
        value = self._equity_cache.get(key)
        if value is None:
            if len(self._equity_cache) > 6000:
                self._equity_cache.clear()
            value = fast_equity(obs.hole, obs.board, opponents, iters=iters)
            self._equity_cache[key] = value
        return value

    def _opponent_names(self, obs: Observation) -> List[str]:
        return [obs.names[i] for i in range(obs.num_players)
                if i != obs.seat and obs.in_hand[i] and not obs.folded[i]]

    def _prior_probs(self, equity: float) -> np.ndarray:
        for threshold, weights in _PRIOR_BANDS:
            if equity >= threshold:
                return np.array(weights, dtype=np.float64)
        return np.array(_PRIOR_BANDS[-1][1], dtype=np.float64)

    def reference_probs(self, obs: Observation) -> np.ndarray:
        """The policy's own view of a spot, with no difficulty handicapping.

        Used to grade decisions (skill tracking and the analyser), so it must
        not be softened by temperature or blunders.
        """
        mask = legal_mask(obs)
        opponents = max(1, obs.live_opponents)
        equity = self._equity(obs, opponents, 400)
        if self.policy is not None:
            features = encode(obs, equity_field=equity,
                              opponent_stats=self.tracker.aggregate_stats(
                                  self._opponent_names(obs)))
            probs = self.policy.action_probs(features, mask, 1.0)
        else:
            probs = self._prior_probs(equity) * mask
        if self.blueprint is not None:
            bp = self._blueprint_probs(obs, mask)
            if bp is not None:
                probs = 0.5 * probs + 0.5 * bp
        total = probs.sum()
        return probs / total if total > 0 else mask / mask.sum()

    def _blueprint_probs(self, obs: Observation, mask: np.ndarray) -> Optional[np.ndarray]:
        if self.blueprint is None:
            return None
        key = observation_infoset(obs)
        if key not in self.blueprint.strategy_sum:
            return None
        legal = [i for i in range(NUM_ACTIONS) if mask[i]]
        probs = self.blueprint.average_strategy(key, legal) * mask
        total = probs.sum()
        return probs / total if total > 0 else None

    # -- exploitation --------------------------------------------------------

    def _target_opponent(self, obs: Observation) -> Optional[int]:
        aggressor = last_aggressor(obs)
        if aggressor is not None and aggressor != obs.seat:
            return aggressor
        live = [i for i in range(obs.num_players)
                if i != obs.seat and obs.in_hand[i] and not obs.folded[i]]
        if not live:
            return None
        return max(live, key=lambda s: obs.total_committed[s])

    def _exploit(self, probs: np.ndarray, obs: Observation, mask: np.ndarray,
                 equity: float) -> np.ndarray:
        weight = self.difficulty.exploit_weight
        if weight <= 0:
            return probs
        seat = self._target_opponent(obs)
        if seat is None:
            return probs
        model = self.tracker.models.get(obs.names[seat])
        if model is None or model.confidence < 0.05:
            return probs

        log_adj = np.zeros(NUM_ACTIONS, dtype=np.float64)
        confidence = model.confidence

        # How often does this player fold to a two-thirds-pot bet here?
        if mask[RAISE_ACTIONS[1]] or mask[A_ALLIN]:
            target = raise_target(obs, 0.66)
            risk = max(1, target - obs.street_committed[obs.seat])
            their_call = max(0, target - obs.street_committed[seat])
            their_obs = replace(
                obs, seat=seat, name=obs.names[seat], hole=[],
                pot=obs.pot + risk, to_call=their_call, current_bet=target, legal=[],
            )
            fold_p = model.fold_probability(their_obs)
            breakeven = risk / max(1.0, obs.pot + risk)
            edge = fold_p - breakeven
            if equity < 0.5:
                # Bluff: profitable exactly when they fold more than breakeven.
                log_adj[list(RAISE_ACTIONS)] += 2.2 * weight * confidence * edge
            else:
                # Value: bet bigger against someone who does not fold.
                log_adj[list(RAISE_ACTIONS)] += 1.4 * weight * confidence * (0.5 - fold_p)

        # Facing a bet from a habitual bluffer: fold less.
        if obs.to_call > 0 and mask[A_FOLD]:
            stats = model.stats()
            bluffy = (stats["aggression"] - 0.5) + (0.6 - stats["showdown_strength"])
            log_adj[A_FOLD] -= 1.6 * weight * confidence * bluffy
            log_adj[A_CALL] += 0.8 * weight * confidence * bluffy

        adjusted = probs * np.exp(np.clip(log_adj, -3.0, 3.0)) * mask
        total = adjusted.sum()
        return adjusted / total if total > 0 else probs

    # -- the decision --------------------------------------------------------

    def policy_probs(self, obs: Observation) -> np.ndarray:
        mask = legal_mask(obs)
        opponents = max(1, obs.live_opponents)
        equity = self._equity(obs, opponents, self.difficulty.equity_iters)
        stats = self.tracker.aggregate_stats(self._opponent_names(obs))

        if self.policy is not None:
            features = encode(obs, equity_field=equity, opponent_stats=stats)
            probs = self.policy.action_probs(features, mask, self.difficulty.temperature)
        else:
            probs = self._prior_probs(equity) * mask
            probs = probs / probs.sum() if probs.sum() > 0 else mask / mask.sum()
            if self.difficulty.temperature != 1.0:
                probs = mask_softmax(np.log(probs + 1e-9), mask,
                                     self.difficulty.temperature)

        bp = self._blueprint_probs(obs, mask)
        if bp is not None and self.difficulty.use_blueprint > 0:
            self.blueprint_hits += 1
            w = self.difficulty.use_blueprint
            probs = (1 - w) * probs + w * bp

        probs = self._exploit(probs, obs, mask, equity)

        if self.difficulty.aggression != 1.0:
            probs = probs.copy()
            probs[list(RAISE_ACTIONS)] *= self.difficulty.aggression
            total = probs.sum()
            probs = probs / total if total > 0 else probs

        return probs

    def act(self, obs: Observation) -> Action:
        self.decisions += 1
        mask = legal_mask(obs)
        if self.rng.random() < self.difficulty.blunder_rate:
            legal = [i for i in range(NUM_ACTIONS) if mask[i]]
            return to_engine_action(obs, self.rng.choice(legal))
        probs = self.policy_probs(obs)
        probs = np.where(mask, probs, 0.0)
        total = probs.sum()
        if total <= 0:
            return to_engine_action(obs, A_CALL)
        index = int(self.rng.choices(range(NUM_ACTIONS), weights=probs / total, k=1)[0])
        return to_engine_action(obs, index)

    # -- learning between decisions -----------------------------------------

    def on_action(self, record: ActionRecord, obs_public: Observation) -> None:
        if record.name != self.name:
            self.tracker.on_action(record, obs_public)

    def on_hand_end(self, result: HandResult, seat: int) -> None:
        self.tracker.on_hand_end(result)
        self._grade_student(result)
        if self.adaptive_ctl is not None and self.skill is not None:
            model = self.tracker.models.get(self.study)
            rating = self.skill.update(model.stats() if model else None)
            self.difficulty = self.adaptive_ctl.update(rating, self.skill.confidence)

    def _grade_student(self, result: HandResult) -> None:
        """Score the tracked player's decisions — only on hands they showed down."""
        if self.skill is None or self.study is None:
            return
        seats = [s for s, n in result.names.items() if n == self.study]
        if not seats or seats[0] not in result.revealed:
            return
        from ..analysis.replay import replay_hand

        for dp in replay_hand(result):
            if dp.name != self.study:
                continue
            probs = self.reference_probs(dp.obs)
            self.skill.observe_decision(probs, to_abstract(dp.obs, dp.action))
        self.skill.observe_hand(result.net.get(seats[0], 0), 2)

    # -- introspection -------------------------------------------------------

    def read_on(self, name: str) -> str:
        model = self.tracker.models.get(name)
        if model is None:
            return f"no read on {name} yet"
        s = model.stats()
        return (f"{name}: {model.label()} — VPIP {s['vpip']:.0%}, PFR {s['pfr']:.0%}, "
                f"AF {s['aggression_factor']:.1f}, folds to bets {s['fold_to_bet']:.0%} "
                f"({int(s['hands'])} hands)")

    def status(self) -> str:
        line = f"{self.name}: difficulty {self.difficulty.name} (level {self.difficulty.level:.2f})"
        if self.skill:
            model = self.tracker.models.get(self.study)
            line += " | " + self.skill.report(model.stats() if model else None)
        return line

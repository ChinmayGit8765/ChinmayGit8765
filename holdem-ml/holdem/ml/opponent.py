"""Opponent modelling: learn how *this particular person* plays, while playing.

Two layers, both online:

1. **Statistics** — the classic HUD numbers (VPIP, PFR, 3-bet, aggression
   factor, fold-to-bet, went-to-showdown), tracked as exponentially decayed
   counters so the model follows a player who changes gear.
2. **A learned action predictor** — a small neural net, trained by SGD on every
   decision it observes, that predicts P(fold / call / raise) for that opponent
   in a given public spot.  This is what lets a bot exploit a *specific* human
   rather than an average one: if you over-fold to river bets, the predictor
   learns it within a few dozen hands and the bot starts bluffing more.

Nothing here ever sees hole cards it is not entitled to.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np

from ..engine import ActionRecord, ActionType, HandResult, Observation
from .abstraction import strength_percentile
from .features import NUM_PUBLIC_FEATURES, encode_public, street_context
from .nn import Adam, SoftmaxCrossEntropy, mlp, softmax

PRED_FOLD, PRED_CALL, PRED_RAISE = 0, 1, 2
NUM_PRED_CLASSES = 3
STAT_FEATURES = 6
PREDICTOR_INPUT = NUM_PUBLIC_FEATURES + STAT_FEATURES


@dataclass
class DecayCounter:
    """Exponentially decayed hit-rate: recent hands count more than old ones."""

    decay: float = 0.995
    hits: float = 0.0
    opportunities: float = 0.0
    prior: float = 0.5
    prior_weight: float = 4.0

    def observe(self, hit: bool, weight: float = 1.0) -> None:
        self.hits = self.hits * self.decay + (weight if hit else 0.0)
        self.opportunities = self.opportunities * self.decay + weight

    @property
    def value(self) -> float:
        return ((self.hits + self.prior * self.prior_weight)
                / (self.opportunities + self.prior_weight))

    @property
    def samples(self) -> float:
        return self.opportunities


@dataclass
class RatioCounter:
    """Aggression factor style ratio (bets+raises) / calls."""

    decay: float = 0.995
    top: float = 0.0
    bottom: float = 0.0

    def observe_top(self, weight: float = 1.0) -> None:
        self.top = self.top * self.decay + weight
        self.bottom *= self.decay

    def observe_bottom(self, weight: float = 1.0) -> None:
        self.bottom = self.bottom * self.decay + weight
        self.top *= self.decay

    @property
    def value(self) -> float:
        return self.top / max(0.5, self.bottom)

    @property
    def normalised(self) -> float:
        """Squashed to [0, 1] so it can be a network input."""
        return float(np.tanh(self.value / 3.0))


class ActionPredictor:
    """Small online-trained classifier of one opponent's next action."""

    def __init__(self, hidden: Sequence[int] = (48, 32), lr: float = 3e-3,
                 buffer_size: int = 4000, batch_size: int = 32,
                 rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(0)
        sizes = [PREDICTOR_INPUT, *hidden, NUM_PRED_CLASSES]
        self.net = mlp(sizes, rng=self.rng)
        self.opt = Adam(self.net.parameters(), lr=lr)
        self.loss = SoftmaxCrossEntropy()
        self.buffer: Deque[Tuple[np.ndarray, int]] = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.updates = 0
        self.recent_loss = math.log(NUM_PRED_CLASSES)

    def observe(self, x: np.ndarray, label: int, train_every: int = 2) -> None:
        self.buffer.append((x.astype(np.float32), int(label)))
        if len(self.buffer) < 16 or len(self.buffer) % train_every:
            return
        self.train_step()

    def train_step(self) -> float:
        n = min(self.batch_size, len(self.buffer))
        idx = self.rng.integers(len(self.buffer), size=n)
        xs = np.stack([self.buffer[i][0] for i in idx])
        ys = np.array([self.buffer[i][1] for i in idx])
        self.opt.zero_grad()
        logits = self.net.forward(xs, training=True)
        loss, grad = self.loss(logits, ys)
        self.net.backward(grad)
        self.opt.clip_grad_norm(5.0)
        self.opt.step()
        self.updates += 1
        self.recent_loss = 0.9 * self.recent_loss + 0.1 * loss
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.net.forward(x[None, :].astype(np.float32), training=False)
        return softmax(logits)[0]

    @property
    def trained(self) -> bool:
        return self.updates >= 20


class OpponentModel:
    """Everything one bot knows about one other player."""

    def __init__(self, name: str, decay: float = 0.995,
                 rng: Optional[np.random.Generator] = None, learn: bool = True):
        self.name = name
        self.hands = 0
        self.decisions = 0
        self.vpip = DecayCounter(decay, prior=0.30)
        self.pfr = DecayCounter(decay, prior=0.20)
        self.three_bet = DecayCounter(decay, prior=0.08)
        self.fold_to_bet = DecayCounter(decay, prior=0.45)
        self.cbet = DecayCounter(decay, prior=0.55)
        self.wtsd = DecayCounter(decay, prior=0.28)
        self.aggression = RatioCounter(decay)
        self.showdown_strength = DecayCounter(decay, prior=0.6)
        self.net_bb = 0.0
        self.predictor = ActionPredictor(rng=rng) if learn else None
        self._saw_flop = False
        self._acted_preflop = False

    # -- observation ---------------------------------------------------------

    def stat_vector(self) -> np.ndarray:
        return np.array([
            self.vpip.value, self.pfr.value, self.aggression.normalised,
            self.fold_to_bet.value, self.wtsd.value, self.confidence,
        ], dtype=np.float32)

    def stats(self) -> Dict[str, float]:
        return {
            "vpip": self.vpip.value,
            "pfr": self.pfr.value,
            "three_bet": self.three_bet.value,
            "aggression": self.aggression.normalised,
            "aggression_factor": self.aggression.value,
            "fold_to_bet": self.fold_to_bet.value,
            "cbet": self.cbet.value,
            "wtsd": self.wtsd.value,
            "showdown_strength": self.showdown_strength.value,
            "confidence": self.confidence,
            "hands": float(self.hands),
        }

    @property
    def confidence(self) -> float:
        return min(1.0, self.hands / 40.0)

    def observe_action(self, record: ActionRecord, obs: Observation) -> None:
        self.decisions += 1
        ctx = street_context(obs)
        aggressive = record.action in (ActionType.BET, ActionType.RAISE)

        if record.street == 0:
            voluntary_spot = record.to_call > 0 or aggressive
            if voluntary_spot and not self._acted_preflop:
                self.vpip.observe(record.action in (ActionType.CALL, ActionType.BET,
                                                    ActionType.RAISE))
                self.pfr.observe(aggressive)
                self._acted_preflop = True
            if ctx["raises"] >= 1 and record.to_call > 0:
                self.three_bet.observe(aggressive)
        else:
            self._saw_flop = True
            if aggressive:
                self.aggression.observe_top()
            elif record.action == ActionType.CALL:
                self.aggression.observe_bottom()
            if record.street == 1 and ctx["raises"] == 0 and record.to_call == 0:
                pf_agg = None
                for r in obs.history:
                    if r.street == 0 and r.action in (ActionType.BET, ActionType.RAISE):
                        pf_agg = r.seat
                if pf_agg == record.seat:
                    self.cbet.observe(aggressive)

        if record.to_call > 0:
            self.fold_to_bet.observe(record.action == ActionType.FOLD)

        if self.predictor is not None:
            x = np.concatenate([encode_public(obs), self.stat_vector()])
            label = (PRED_FOLD if record.action == ActionType.FOLD else
                     PRED_RAISE if aggressive else PRED_CALL)
            self.predictor.observe(x, label)

    def observe_hand_end(self, result: HandResult, seat: int) -> None:
        self.hands += 1
        if self._saw_flop:
            self.wtsd.observe(seat in result.revealed)
        hole = result.revealed.get(seat)
        if hole and len(result.board) >= 3:
            self.showdown_strength.observe(
                strength_percentile(hole, result.board) > 0.5
            )
        self.net_bb += result.net.get(seat, 0)
        self._saw_flop = False
        self._acted_preflop = False

    # -- prediction ----------------------------------------------------------

    def predict_action(self, obs: Observation) -> np.ndarray:
        """P(fold, call, raise) for this opponent in the spot described by ``obs``.

        Blends the learned predictor with the statistical prior, weighted by
        how much evidence we have.
        """
        prior = self._statistical_prior(obs)
        if self.predictor is None or not self.predictor.trained:
            return prior
        x = np.concatenate([encode_public(obs), self.stat_vector()])
        learned = self.predictor.predict(x)
        w = self.confidence
        return (1 - w) * prior + w * learned

    def _statistical_prior(self, obs: Observation) -> np.ndarray:
        if obs.to_call > 0:
            fold = self.fold_to_bet.value
            raise_p = 0.12 + 0.2 * self.aggression.normalised
            raise_p = min(raise_p, max(0.0, 1.0 - fold - 0.05))
            return np.array([fold, max(0.05, 1 - fold - raise_p), raise_p])
        raise_p = 0.25 + 0.35 * self.aggression.normalised
        return np.array([0.0, 1 - raise_p, raise_p])

    def fold_probability(self, obs: Observation) -> float:
        return float(self.predict_action(obs)[PRED_FOLD])

    def label(self) -> str:
        """Short human-readable read, e.g. ``"loose-aggressive"``."""
        if self.hands < 12:
            return "unknown"
        loose = self.vpip.value > 0.32
        aggro = self.aggression.value > 1.6 or self.pfr.value > 0.22
        style = ("loose" if loose else "tight") + "-" + ("aggressive" if aggro else "passive")
        return style


class OpponentTracker:
    """Keeps an :class:`OpponentModel` per player name."""

    def __init__(self, decay: float = 0.995, learn: bool = True,
                 rng: Optional[np.random.Generator] = None):
        self.models: Dict[str, OpponentModel] = {}
        self.decay = decay
        self.learn = learn
        self.rng = rng or np.random.default_rng(0)

    def get(self, name: str) -> OpponentModel:
        model = self.models.get(name)
        if model is None:
            model = OpponentModel(name, self.decay, rng=self.rng, learn=self.learn)
            self.models[name] = model
        return model

    def on_action(self, record: ActionRecord, obs: Observation) -> None:
        self.get(record.name).observe_action(record, obs)

    def on_hand_end(self, result: HandResult) -> None:
        for seat, name in result.names.items():
            if seat in result.hole or seat in result.net:
                self.get(name).observe_hand_end(result, seat)

    def aggregate_stats(self, names: Sequence[str]) -> Dict[str, float]:
        """Average the models for a set of opponents (for the feature vector)."""
        models = [self.models[n] for n in names if n in self.models]
        if not models:
            return {}
        keys = ["vpip", "pfr", "aggression", "fold_to_bet", "wtsd", "confidence"]
        out = {}
        for key in keys:
            out[key] = float(np.mean([m.stats()[key] for m in models]))
        return out

    def report(self) -> str:
        rows = ["player            hands  vpip   pfr   3bet   af  fold/bet  wtsd  style"]
        for name, m in sorted(self.models.items()):
            s = m.stats()
            rows.append(
                f"{name:<16}{s['hands']:>6.0f}{s['vpip']:>6.0%}{s['pfr']:>6.0%}"
                f"{s['three_bet']:>7.0%}{s['aggression_factor']:>5.1f}"
                f"{s['fold_to_bet']:>9.0%}{s['wtsd']:>6.0%}  {m.label()}"
            )
        return "\n".join(rows)

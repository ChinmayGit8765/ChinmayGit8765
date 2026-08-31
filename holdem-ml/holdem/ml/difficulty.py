"""Difficulty levels, and difficulty that *tracks the player*.

Two ways to set how hard the bots are:

* **Fixed** — pick ``novice`` .. ``pro``.  Levels differ in policy temperature
  (how sharply the net's preferred action is followed), blunder rate, how much
  the opponent model is used to exploit you, and how much roll-out budget the
  bot spends on equity.
* **Adaptive** — :class:`AdaptiveDifficulty` watches a :class:`SkillTracker`
  estimate of the human and slides the level to sit slightly above them, so the
  game stays a fair fight as the player improves.

The skill estimate is deliberately a blend of three weak signals (agreement
with a reference policy, statistical leaks, realised win-rate) because any one
of them alone is either noisy or gameable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Difficulty:
    name: str
    level: float           # 0 = beginner, 1 = strongest
    temperature: float     # >1 softens the policy toward random
    blunder_rate: float    # chance of ignoring the policy entirely
    exploit_weight: float  # how hard to lean on the opponent model
    aggression: float      # multiplier on raise probabilities
    equity_iters: int      # Monte-Carlo budget per decision
    use_blueprint: float   # weight on the CFR blueprint vs the neural policy
                           # (measured: blending it in *weakens* a trained
                           # policy, so it is used to make the lower levels
                           # play a textbook-but-exploitable style)

    def blend(self, other: "Difficulty", t: float) -> "Difficulty":
        t = float(np.clip(t, 0.0, 1.0))

        def mix(a, b):
            return a + (b - a) * t

        return Difficulty(
            name=f"{self.name}->{other.name}",
            level=mix(self.level, other.level),
            temperature=mix(self.temperature, other.temperature),
            blunder_rate=mix(self.blunder_rate, other.blunder_rate),
            exploit_weight=mix(self.exploit_weight, other.exploit_weight),
            aggression=mix(self.aggression, other.aggression),
            equity_iters=int(mix(self.equity_iters, other.equity_iters)),
            use_blueprint=mix(self.use_blueprint, other.use_blueprint),
        )


LEVELS: List[Difficulty] = [
    Difficulty("novice", 0.00, temperature=2.6, blunder_rate=0.22, exploit_weight=0.0,
               aggression=0.80, equity_iters=80, use_blueprint=0.50),
    Difficulty("casual", 0.25, temperature=1.8, blunder_rate=0.10, exploit_weight=0.15,
               aggression=0.90, equity_iters=160, use_blueprint=0.35),
    Difficulty("regular", 0.50, temperature=1.30, blunder_rate=0.04, exploit_weight=0.40,
               aggression=1.00, equity_iters=320, use_blueprint=0.18),
    Difficulty("strong", 0.75, temperature=1.05, blunder_rate=0.01, exploit_weight=0.70,
               aggression=1.05, equity_iters=650, use_blueprint=0.05),
    Difficulty("pro", 1.00, temperature=0.90, blunder_rate=0.00, exploit_weight=1.00,
               aggression=1.10, equity_iters=1200, use_blueprint=0.00),
]
LEVELS_BY_NAME: Dict[str, Difficulty] = {d.name: d for d in LEVELS}


def difficulty_at(level: float) -> Difficulty:
    """Interpolate the preset ladder at an arbitrary level in ``[0, 1]``."""
    level = float(np.clip(level, 0.0, 1.0))
    for i in range(len(LEVELS) - 1):
        lo, hi = LEVELS[i], LEVELS[i + 1]
        if lo.level <= level <= hi.level:
            span = hi.level - lo.level
            out = lo.blend(hi, 0.0 if span <= 0 else (level - lo.level) / span)
            out.name = f"level {level:.2f}"
            out.level = level
            return out
    return replace(LEVELS[-1])


def resolve_difficulty(spec) -> Difficulty:
    """Accept a name, a number, or a :class:`Difficulty`."""
    if isinstance(spec, Difficulty):
        return spec
    if isinstance(spec, (int, float)):
        return difficulty_at(float(spec))
    if isinstance(spec, str):
        if spec in LEVELS_BY_NAME:
            return replace(LEVELS_BY_NAME[spec])
        try:
            return difficulty_at(float(spec))
        except ValueError:
            raise ValueError(
                f"unknown difficulty {spec!r}; choose from "
                f"{sorted(LEVELS_BY_NAME)} or a number in [0, 1]"
            )
    raise TypeError(f"cannot interpret difficulty {spec!r}")


# --- skill estimation -------------------------------------------------------

HEALTHY_RANGES: Dict[str, Tuple[float, float]] = {
    # stat: (low, high) band that competent play tends to sit in
    "vpip": (0.17, 0.33),
    "pfr": (0.12, 0.27),
    "fold_to_bet": (0.32, 0.58),
    "aggression_factor": (1.1, 3.2),
    "wtsd": (0.21, 0.34),
}


class SkillTracker:
    """Estimate how well one player plays, on a 0..1 scale.

    Signals, in order of weight:

    ``agreement``  the probability a strong reference policy assigns to the
                   action the player actually chose.  Strong play agrees with a
                   near-equilibrium policy far more often than weak play.
    ``leaks``      how far the player's HUD stats sit outside healthy bands.
    ``results``    realised big blinds per 100 hands (noisy, so weighted least).
    """

    def __init__(self, name: str, decay: float = 0.99):
        self.name = name
        self.decay = decay
        self.agreement = 0.0
        self.agreement_weight = 0.0
        self.decisions = 0
        self.hands = 0
        self.net_bb = 0.0
        self.rating_ewma = 0.5
        self.history: List[float] = []

    def observe_decision(self, reference_probs: np.ndarray, chosen: int) -> None:
        p = float(reference_probs[chosen]) if 0 <= chosen < len(reference_probs) else 0.0
        legal = max(1, int((reference_probs > 1e-6).sum()))
        # Normalise against the "no information" baseline so wide spots (where
        # even a strong policy is mixed) do not look like mistakes.
        baseline = 1.0 / legal
        score = float(np.clip((p - baseline * 0.5) / max(1e-6, 1.0 - baseline * 0.5), 0.0, 1.0))
        self.agreement = self.agreement * self.decay + score
        self.agreement_weight = self.agreement_weight * self.decay + 1.0
        self.decisions += 1

    def observe_hand(self, net_chips: float, big_blind: float) -> None:
        self.hands += 1
        self.net_bb += net_chips / max(1.0, big_blind)

    @property
    def agreement_score(self) -> float:
        if self.agreement_weight <= 0:
            return 0.5
        mean = self.agreement / self.agreement_weight
        return float(np.clip((mean - 0.18) / 0.5, 0.0, 1.0))

    def leak_score(self, stats: Optional[Dict[str, float]]) -> float:
        if not stats:
            return 0.5
        penalties = []
        for key, (lo, hi) in HEALTHY_RANGES.items():
            value = stats.get(key)
            if value is None:
                continue
            width = hi - lo
            if value < lo:
                penalties.append(min(1.0, (lo - value) / width))
            elif value > hi:
                penalties.append(min(1.0, (value - hi) / width))
            else:
                penalties.append(0.0)
        if not penalties:
            return 0.5
        return float(np.clip(1.0 - np.mean(penalties), 0.0, 1.0))

    def result_score(self) -> float:
        if self.hands < 30:
            return 0.5
        bb100 = 100.0 * self.net_bb / self.hands
        return float(1.0 / (1.0 + np.exp(-bb100 / 12.0)))

    @property
    def confidence(self) -> float:
        return float(min(1.0, self.decisions / 120.0))

    def raw_rating(self, stats: Optional[Dict[str, float]] = None) -> float:
        """The instantaneous estimate, with no smoothing."""
        raw = (0.50 * self.agreement_score
               + 0.35 * self.leak_score(stats)
               + 0.15 * self.result_score())
        # Shrink toward 0.5 until we have seen enough decisions to trust it.
        return 0.5 + (raw - 0.5) * (0.35 + 0.65 * self.confidence)

    def update(self, stats: Optional[Dict[str, float]] = None) -> float:
        """Fold the current estimate into the smoothed rating and return it."""
        self.rating_ewma = 0.9 * self.rating_ewma + 0.1 * self.raw_rating(stats)
        self.history.append(self.rating_ewma)
        return self.rating_ewma

    @property
    def rating(self) -> float:
        """The smoothed rating.  Reading it never changes it — call
        :meth:`update` once per hand to advance it."""
        return self.rating_ewma

    def report(self, stats: Optional[Dict[str, float]] = None) -> str:
        return (f"{self.name}: skill {self.raw_rating(stats):.2f} "
                f"(agreement {self.agreement_score:.2f}, leaks {self.leak_score(stats):.2f}, "
                f"results {self.result_score():.2f}, confidence {self.confidence:.0%}, "
                f"{self.decisions} decisions)")


class AdaptiveDifficulty:
    """Slides the bot difficulty to sit just above the player's current skill."""

    def __init__(self, start: float = 0.35, offset: float = 0.08,
                 rate: float = 0.06, floor: float = 0.05, ceiling: float = 1.0):
        self.level = float(start)
        self.offset = offset
        self.rate = rate
        self.floor = floor
        self.ceiling = ceiling
        self.trail: List[float] = [self.level]

    def update(self, skill: float, confidence: float = 1.0) -> Difficulty:
        target = float(np.clip(skill + self.offset, self.floor, self.ceiling))
        step = self.rate * (0.3 + 0.7 * confidence)
        self.level += float(np.clip(target - self.level, -step, step))
        self.level = float(np.clip(self.level, self.floor, self.ceiling))
        self.trail.append(self.level)
        return self.current()

    def current(self) -> Difficulty:
        d = difficulty_at(self.level)
        d.name = f"adaptive({self.level:.2f})"
        return d

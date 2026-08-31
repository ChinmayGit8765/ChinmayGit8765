"""Game abstraction: how a poker situation is compressed for learning.

Two separate abstractions:

*Cards* — a hand becomes ``(strength decile, draw class)``.  Strength is the
percentile of the made hand's score against the distribution of all hands on
that street (precomputed once into ``data/strength_percentiles.json``); the draw
class captures flush/straight equity that a made-hand score misses.

*Betting* — continuous no-limit bet sizing collapses to a fixed 7-action menu
(fold, check/call, four pot-fractions, all-in).  Every learner in the project
shares this menu, so the CFR blueprint, the neural policy and the analyser all
speak the same language.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_left
from typing import List, Optional, Sequence

import numpy as np

from ..cards import hole_label
from ..engine import Action, ActionType, LegalAction, Observation
from ..evaluator import evaluate

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")
PERCENTILE_PATH = os.path.join(DATA_DIR, "strength_percentiles.json")

# --- action abstraction -----------------------------------------------------

A_FOLD = 0
A_CALL = 1
A_RAISE_33 = 2
A_RAISE_66 = 3
A_RAISE_100 = 4
A_RAISE_200 = 5
A_ALLIN = 6
NUM_ACTIONS = 7

ACTION_NAMES = ["fold", "check/call", "raise 33%", "raise 66%", "raise pot",
                "raise 2x pot", "all-in"]
ACTION_SHORT = ["f", "c", "r33", "r66", "r100", "r200", "ai"]
RAISE_FRACTIONS = {A_RAISE_33: 0.33, A_RAISE_66: 0.66, A_RAISE_100: 1.0, A_RAISE_200: 2.0}


def raise_target(obs: Observation, fraction: float) -> int:
    """Chip total to raise *to* for a pot-fraction bet."""
    pot_after_call = obs.pot + obs.to_call
    return obs.current_bet + max(1, int(round(fraction * pot_after_call)))


def legal_mask(obs: Observation) -> np.ndarray:
    """Boolean mask over the 7 abstract actions for this spot."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[A_CALL] = True
    if obs.to_call > 0:
        mask[A_FOLD] = True
    raise_la: Optional[LegalAction] = None
    for la in obs.legal:
        if la.type in (ActionType.BET, ActionType.RAISE):
            raise_la = la
    if raise_la is None:
        return mask
    mask[A_ALLIN] = True
    seen = {raise_la.max_amount}
    for idx, frac in RAISE_FRACTIONS.items():
        target = raise_target(obs, frac)
        target = max(raise_la.min_amount, min(target, raise_la.max_amount))
        # Drop sizes that collapse onto all-in so the policy isn't given
        # duplicate choices with different labels.
        if target >= raise_la.max_amount or target in seen:
            continue
        seen.add(target)
        mask[idx] = True
    return mask


def to_engine_action(obs: Observation, abstract: int) -> Action:
    """Turn an abstract action index into a concrete, legal engine action."""
    raise_la: Optional[LegalAction] = None
    for la in obs.legal:
        if la.type in (ActionType.BET, ActionType.RAISE):
            raise_la = la
    if abstract == A_FOLD:
        if obs.to_call > 0:
            return Action(ActionType.FOLD)
        abstract = A_CALL
    if abstract == A_CALL:
        if obs.to_call > 0:
            la = obs.legal_of(ActionType.CALL)
            return Action(ActionType.CALL, la.min_amount if la else obs.to_call)
        return Action(ActionType.CHECK)
    if raise_la is None:  # cannot raise: fall back to check/call
        return to_engine_action(obs, A_CALL)
    if abstract == A_ALLIN:
        return Action(raise_la.type, raise_la.max_amount)
    frac = RAISE_FRACTIONS.get(abstract)
    if frac is None:  # pragma: no cover - defensive
        return to_engine_action(obs, A_CALL)
    target = raise_target(obs, frac)
    target = max(raise_la.min_amount, min(target, raise_la.max_amount))
    return Action(raise_la.type, target)


def to_abstract(obs: Observation, action: Action) -> int:
    """Map a concrete engine action back to the nearest abstract action index.

    The result is always an index :func:`legal_mask` marks as available for this
    spot.  That matters: sizes that collapse onto all-in (or onto each other)
    are dropped from the mask, and returning one of them would produce a
    training target the policy is forbidden to predict.
    """
    if action.type == ActionType.FOLD:
        return A_FOLD
    if action.type in (ActionType.CHECK, ActionType.CALL):
        return A_CALL
    mask = legal_mask(obs)
    raise_la = None
    for la in obs.legal:
        if la.type in (ActionType.BET, ActionType.RAISE):
            raise_la = la
    if raise_la is None:
        return A_CALL
    if action.amount >= raise_la.max_amount or not mask[list(RAISE_FRACTIONS)].any():
        return A_ALLIN if mask[A_ALLIN] else A_CALL

    pot_after_call = max(1, obs.pot + obs.to_call)
    fraction = (action.amount - obs.current_bet) / pot_after_call
    if fraction > 3.0:
        return A_ALLIN if mask[A_ALLIN] else A_CALL
    best, best_gap = None, float("inf")
    for idx, frac in RAISE_FRACTIONS.items():
        if not mask[idx]:
            continue
        gap = abs(frac - fraction)
        if gap < best_gap:
            best, best_gap = idx, gap
    if best is None:
        return A_ALLIN if mask[A_ALLIN] else A_CALL
    return best


def describe_abstract(index: int) -> str:
    return ACTION_NAMES[index]


# --- card abstraction -------------------------------------------------------

_percentiles: Optional[List[List[int]]] = None
NUM_STRENGTH_BUCKETS = 10
NUM_DRAW_CLASSES = 4
NUM_POSTFLOP_BUCKETS = NUM_STRENGTH_BUCKETS * NUM_DRAW_CLASSES


def load_percentiles() -> List[List[int]]:
    """Sorted score quantiles per street (index 1..3 = flop/turn/river)."""
    global _percentiles
    if _percentiles is None:
        if os.path.exists(PERCENTILE_PATH):
            with open(PERCENTILE_PATH) as fh:
                _percentiles = json.load(fh)
        else:  # pragma: no cover - before generation
            _percentiles = [[], [], [], []]
    return _percentiles


def strength_percentile(hole: Sequence[int], board: Sequence[int]) -> float:
    """Where this made hand sits in the distribution of hands on this street."""
    if not board:
        return 0.0
    street = {3: 1, 4: 2, 5: 3}[len(board)]
    table = load_percentiles()[street]
    if not table:  # pragma: no cover
        return 0.5
    score = evaluate(list(hole) + list(board))
    return bisect_left(table, score) / len(table)


def draw_class(hole: Sequence[int], board: Sequence[int]) -> int:
    """0 none, 1 gutshot, 2 open-ended straight draw, 3 flush draw (or better)."""
    if not board or len(board) >= 5:
        return 0
    cards = list(hole) + list(board)
    suits = [0, 0, 0, 0]
    rank_mask = 0
    suit_rank_mask = [0, 0, 0, 0]
    for c in cards:
        s = c & 3
        r = c >> 2
        suits[s] += 1
        rank_mask |= 1 << r
        suit_rank_mask[s] |= 1 << r
    for s in range(4):
        if suits[s] == 4:
            return 3
        if suits[s] >= 5:
            return 3
    mask = rank_mask | ((rank_mask >> 12) & 1)  # ace plays low for the wheel
    best = 0
    for high in range(12, 2, -1):
        window = (mask >> max(0, high - 4)) & 0b11111
        bits = bin(window).count("1")
        if bits == 4:
            best = max(best, 2 if _is_open_ended(mask, high) else 1)
    return best


def _is_open_ended(mask: int, high: int) -> bool:
    """Four to a straight with two live ends (not an ace-terminated run)."""
    for low in range(0, 10):
        run = (mask >> low) & 0b1111
        if run == 0b1111:
            if low > 0 and low + 4 <= 12:
                return True
    return False


def card_bucket(hole: Sequence[int], board: Sequence[int]) -> int:
    """Postflop bucket id in ``[0, NUM_POSTFLOP_BUCKETS)``."""
    pct = strength_percentile(hole, board)
    decile = min(NUM_STRENGTH_BUCKETS - 1, int(pct * NUM_STRENGTH_BUCKETS))
    return decile * NUM_DRAW_CLASSES + draw_class(hole, board)


def preflop_bucket(hole: Sequence[int]) -> str:
    return hole_label(hole)


def info_bucket(hole: Sequence[int], board: Sequence[int]) -> str:
    if not board:
        return preflop_bucket(hole)
    return f"{len(board)}:{card_bucket(hole, board)}"

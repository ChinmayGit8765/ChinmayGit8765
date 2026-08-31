"""Monte-Carlo equity: how often a hand wins against random or ranged opponents.

Equity is the single most informative feature a poker bot can look at, so this
is written to be fast (batched NumPy roll-outs) and cached (a precomputed
169 x 8 preflop table loaded from ``data/preflop_equity.json``).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cards import NUM_CARDS, hole_label
from .evaluator_np import evaluate_batch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PREFLOP_TABLE_PATH = os.path.join(DATA_DIR, "preflop_equity.json")

_preflop_table: Optional[Dict[str, List[float]]] = None


def _rollout_cards(
    deck: np.ndarray, rows: int, per_row: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``per_row`` distinct cards from ``deck`` for each of ``rows`` rows."""
    keys = rng.random((rows, deck.size))
    idx = np.argpartition(keys, per_row - 1, axis=1)[:, :per_row]
    return deck[idx]


def equity(
    hole: Sequence[int],
    board: Sequence[int] = (),
    opponents: int = 1,
    iters: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """P(win) + P(tie)/n_tied for ``hole`` against ``opponents`` random hands."""
    if opponents < 1:
        return 1.0
    rng = rng or np.random.default_rng()
    hole = list(hole)
    board = list(board)
    known = set(hole) | set(board)
    deck = np.array([c for c in range(NUM_CARDS) if c not in known], dtype=np.int64)

    board_needed = 5 - len(board)
    draw = board_needed + 2 * opponents
    if draw > deck.size:
        raise ValueError("not enough cards left to roll out")

    sampled = _rollout_cards(deck, iters, draw, rng)
    full_board = np.concatenate(
        [np.tile(np.array(board, dtype=np.int64), (iters, 1)), sampled[:, :board_needed]],
        axis=1,
    ) if board or board_needed else sampled[:, :0]

    hero = np.concatenate([np.tile(np.array(hole, dtype=np.int64), (iters, 1)), full_board], axis=1)
    hero_score = evaluate_batch(hero)

    opp_scores = np.empty((opponents, iters), dtype=np.int64)
    for k in range(opponents):
        cols = sampled[:, board_needed + 2 * k: board_needed + 2 * k + 2]
        opp_scores[k] = evaluate_batch(np.concatenate([cols, full_board], axis=1))
    best_opp = opp_scores.max(axis=0)
    ties = ((opp_scores == hero_score) & (opp_scores == best_opp)).sum(axis=0)

    wins = hero_score > best_opp
    chops = hero_score == best_opp
    share = np.where(chops, 1.0 / (1.0 + ties), 0.0)
    return float((wins.astype(np.float64) + share).mean())


def equity_vs_range(
    hole: Sequence[int],
    board: Sequence[int],
    opponent_ranges: Sequence[Sequence[Sequence[int]]],
    iters: int = 1500,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Equity when each opponent's hand is drawn from an explicit list of combos.

    Ranges are lists of two-card combos; combos that clash with known cards are
    skipped for that sample.  This is what the exploitative bot uses once the
    opponent model has narrowed somebody's range.
    """
    rng = rng or np.random.default_rng()
    hole, board = list(hole), list(board)
    board_needed = 5 - len(board)
    wins = 0.0
    total = 0
    for _ in range(iters):
        used = set(hole) | set(board)
        opp_hands = []
        ok = True
        for combos in opponent_ranges:
            choices = [c for c in combos if not (used & set(c))]
            if not choices:
                ok = False
                break
            pick = choices[rng.integers(len(choices))]
            opp_hands.append(list(pick))
            used |= set(pick)
        if not ok:
            continue
        deck = [c for c in range(NUM_CARDS) if c not in used]
        extra = list(rng.choice(deck, size=board_needed, replace=False)) if board_needed else []
        full_board = board + [int(c) for c in extra]
        hero = evaluate_batch(np.array([hole + full_board]))[0]
        opp = max(evaluate_batch(np.array([h + full_board]))[0] for h in opp_hands)
        total += 1
        if hero > opp:
            wins += 1.0
        elif hero == opp:
            wins += 0.5
    return wins / total if total else 0.5


# --- preflop lookup ---------------------------------------------------------

def load_preflop_table() -> Dict[str, List[float]]:
    global _preflop_table
    if _preflop_table is None:
        if os.path.exists(PREFLOP_TABLE_PATH):
            with open(PREFLOP_TABLE_PATH) as fh:
                _preflop_table = json.load(fh)
        else:  # pragma: no cover - only before the table is generated
            _preflop_table = {}
    return _preflop_table


def preflop_equity(hole: Sequence[int], opponents: int = 1) -> float:
    """Cached preflop equity for the hand's 169-bucket label."""
    table = load_preflop_table()
    row = table.get(hole_label(hole))
    if row:
        return row[min(max(opponents, 1), len(row)) - 1]
    key = tuple(sorted(hole))
    return equity(hole, (), opponents, iters=400,
                  rng=np.random.default_rng(_seed_for(key, opponents, 400)))


def _seed_for(key: tuple, opponents: int, iters: int) -> int:
    """A stable seed derived from the situation itself.

    Roll-outs used to draw from OS entropy, which made a bot's decisions differ
    between runs even with a seeded game — reproducing a hand was impossible.
    Seeding from the (cards, opponents, budget) key instead makes ``fast_equity``
    a deterministic function of its arguments, so a seeded session replays
    exactly, while different situations still get independent samples.
    """
    h = 1469598103934665603
    for value in (*key, opponents, iters):
        h = ((h ^ (int(value) + 0x9E3779B9)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h & 0x7FFFFFFF


@lru_cache(maxsize=200_000)
def _cached_equity_key(key: tuple, opponents: int, iters: int) -> float:
    hole = list(key[:2])
    board = list(key[2:])
    return equity(hole, board, opponents, iters=iters,
                  rng=np.random.default_rng(_seed_for(key, opponents, iters)))


def fast_equity(
    hole: Sequence[int], board: Sequence[int] = (), opponents: int = 1, iters: int = 600
) -> float:
    """Equity with preflop lookup and a memo on exact (hole, board) states."""
    if not board:
        return preflop_equity(hole, opponents)
    key = tuple(sorted(hole)) + tuple(sorted(board))
    return _cached_equity_key(key, opponents, iters)

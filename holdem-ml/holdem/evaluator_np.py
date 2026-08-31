"""Vectorised hand evaluator — scores whole batches of 7-card hands at once.

Same scoring scheme and exact same results as :mod:`holdem.evaluator`, but
table-driven over NumPy arrays so Monte-Carlo equity and self-play training can
evaluate hundreds of thousands of hands a second.
"""

from __future__ import annotations

import numpy as np

from .evaluator import _straight_high, _top_n  # reuse the reference logic

_SIZE = 8192

TOPBIT = np.zeros(_SIZE, dtype=np.int64)
TOP2P = np.zeros(_SIZE, dtype=np.int64)
TOP3P = np.zeros(_SIZE, dtype=np.int64)
TOP5P = np.zeros(_SIZE, dtype=np.int64)
STRAIGHT_HI = np.full(_SIZE, -1, dtype=np.int64)
FLUSH_SCORE = np.zeros(_SIZE, dtype=np.int64)

for _m in range(_SIZE):
    _top = _top_n(_m, 5)
    TOPBIT[_m] = _top[0] if _top else 0
    if len(_top) >= 2:
        TOP2P[_m] = (_top[0] << 12) | (_top[1] << 8)
    elif _top:
        TOP2P[_m] = _top[0] << 12
    if len(_top) >= 3:
        TOP3P[_m] = (_top[0] << 12) | (_top[1] << 8) | (_top[2] << 4)
    elif len(_top) == 2:
        TOP3P[_m] = (_top[0] << 12) | (_top[1] << 8)
    elif _top:
        TOP3P[_m] = _top[0] << 12
    packed = 0
    for _i in range(5):
        packed = (packed << 4) | (_top[_i] if _i < len(_top) else 0)
    TOP5P[_m] = packed
    _sh = _straight_high(_m)
    STRAIGHT_HI[_m] = _sh
    if bin(_m).count("1") >= 5:
        FLUSH_SCORE[_m] = ((8 << 20) | (_sh << 16)) if _sh >= 0 else ((5 << 20) | packed)

_BIT = (1 << np.arange(13, dtype=np.int64))


def evaluate_batch(hands: np.ndarray) -> np.ndarray:
    """Score an ``(n, k)`` int array of cards (``k`` in 5..7); returns ``(n,)`` int64."""
    hands = np.asarray(hands, dtype=np.int64)
    if hands.ndim == 1:
        hands = hands[None, :]
    n, k = hands.shape
    if not 5 <= k <= 7:
        raise ValueError(f"evaluate_batch() needs 5-7 cards, got {k}")

    ranks = hands >> 2
    suits = hands & 3
    bits = (np.int64(1) << ranks)

    rank_mask = np.bitwise_or.reduce(bits, axis=1)

    # Flushes: OR the rank bits per suit, then a single table lookup.
    flush = np.zeros(n, dtype=np.int64)
    for s in range(4):
        sm = np.bitwise_or.reduce(np.where(suits == s, bits, 0), axis=1)
        flush = np.maximum(flush, FLUSH_SCORE[sm])

    # Rank multiplicities via a flattened bincount.
    flat = (np.arange(n, dtype=np.int64)[:, None] * 13 + ranks).ravel()
    counts = np.bincount(flat, minlength=n * 13).reshape(n, 13)

    m4 = ((counts >= 4) * _BIT).sum(axis=1)
    m3 = ((counts == 3) * _BIT).sum(axis=1)
    m2 = ((counts == 2) * _BIT).sum(axis=1)

    sh = STRAIGHT_HI[rank_mask]

    # Quads
    qr = TOPBIT[m4]
    q_kick = TOPBIT[rank_mask & ~(np.int64(1) << qr)]
    quads_score = (7 << 20) | (qr << 16) | (q_kick << 12)

    # Full house (a second set plays as the pair)
    tr = TOPBIT[m3]
    pair_pool = m2 | (m3 & ~(np.int64(1) << tr))
    pr = TOPBIT[pair_pool]
    fh_score = (6 << 20) | (tr << 16) | (pr << 12)

    straight_score = (4 << 20) | (np.maximum(sh, 0) << 16)

    trips_score = (3 << 20) | (tr << 16) | TOP2P[rank_mask & ~(np.int64(1) << tr)]

    hi2 = TOPBIT[m2]
    lo2 = TOPBIT[m2 & ~(np.int64(1) << hi2)]
    tp_kick = TOPBIT[rank_mask & ~(np.int64(1) << hi2) & ~(np.int64(1) << lo2)]
    two_pair_score = (2 << 20) | (hi2 << 16) | (lo2 << 12) | (tp_kick << 8)

    pair_score = (1 << 20) | (hi2 << 16) | TOP3P[rank_mask & ~(np.int64(1) << hi2)]

    high_score = TOP5P[rank_mask]

    n_pairs = (counts == 2).sum(axis=1)
    has_quads = m4 != 0
    has_trips = m3 != 0
    has_fh = has_trips & (pair_pool != 0)

    score = np.select(
        [flush > 0, has_quads, has_fh, sh >= 0, has_trips, n_pairs >= 2, n_pairs == 1],
        [flush, quads_score, fh_score, straight_score, trips_score, two_pair_score, pair_score],
        default=high_score,
    )
    return score.astype(np.int64)

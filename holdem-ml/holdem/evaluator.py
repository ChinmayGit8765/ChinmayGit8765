"""Exact 5/6/7-card Texas Hold'em hand evaluator.

Hands are scored as a single ``int``; bigger is better and equal ints are exact
ties (chop pots).  The layout is::

    score = category << 20 | k1 << 16 | k2 << 12 | k3 << 8 | k4 << 4 | k5

with ``category`` in ``0..8`` (high card .. straight flush) and ``k1..k5`` the
tie-break ranks, most significant first.  Straights/flush-straights only fill
``k1`` (their high card), which is all that distinguishes them.

Everything is table-driven: 13-bit rank-set lookups are precomputed once at
import (8192 entries), so evaluating a hand is a handful of integer ops.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

from .cards import RANK_CHARS

HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8

CATEGORY_NAMES = [
    "high card", "pair", "two pair", "three of a kind", "straight",
    "flush", "full house", "four of a kind", "straight flush",
]

WHEEL = 0b1000000001111  # A5432


def _straight_high(mask: int) -> int:
    """Highest rank completing a straight in a 13-bit rank mask, else -1."""
    for high in range(12, 3, -1):
        if (mask >> (high - 4)) & 0b11111 == 0b11111:
            return high
    if mask & WHEEL == WHEEL:
        return 3  # five-high wheel
    return -1


def _top_n(mask: int, n: int) -> List[int]:
    out = []
    for r in range(12, -1, -1):
        if mask >> r & 1:
            out.append(r)
            if len(out) == n:
                break
    return out


def _pack(category: int, kickers: Sequence[int]) -> int:
    score = category
    for i in range(5):
        score = (score << 4) | (kickers[i] if i < len(kickers) else 0)
    return score


# --- precomputed tables -----------------------------------------------------
# _STRAIGHT[mask] -> straight high rank or -1
# _FLUSH_SCORE[mask] -> score of the best 5-card flush from that suit's ranks
_STRAIGHT: List[int] = [0] * 8192
_FLUSH_SCORE: List[int] = [0] * 8192
_TOP5: List[int] = [0] * 8192

for _mask in range(8192):
    _sh = _straight_high(_mask)
    _STRAIGHT[_mask] = _sh
    _bits = bin(_mask).count("1")
    if _bits >= 5:
        _TOP5[_mask] = _pack(HIGH_CARD, _top_n(_mask, 5))
        if _sh >= 0:
            _FLUSH_SCORE[_mask] = _pack(STRAIGHT_FLUSH, [_sh])
        else:
            _FLUSH_SCORE[_mask] = _pack(FLUSH, _top_n(_mask, 5))


def evaluate(cards: Sequence[int]) -> int:
    """Score the best five-card hand out of 5, 6 or 7 cards."""
    n = len(cards)
    if n < 5 or n > 7:
        raise ValueError(f"evaluate() needs 5-7 cards, got {n}")

    suit_masks = [0, 0, 0, 0]
    rank_mask = 0
    counts: Dict[int, int] = {}
    for c in cards:
        r = c >> 2
        suit_masks[c & 3] |= 1 << r
        rank_mask |= 1 << r
        counts[r] = counts.get(r, 0) + 1

    for sm in suit_masks:
        if sm and _FLUSH_SCORE[sm]:
            return _FLUSH_SCORE[sm]

    sh = _STRAIGHT[rank_mask]

    # Group ranks by multiplicity, highest count then highest rank first.
    quads = trips = -1
    pairs: List[int] = []
    for r in range(12, -1, -1):
        c = counts.get(r, 0)
        if c == 4:
            if quads < 0:
                quads = r
        elif c == 3:
            if trips < 0:
                trips = r
            else:
                pairs.append(r)  # second set plays as a pair
        elif c == 2:
            pairs.append(r)

    if quads >= 0:
        kicker = max(r for r in counts if r != quads)
        return _pack(QUADS, [quads, kicker])
    if trips >= 0 and pairs:
        return _pack(FULL_HOUSE, [trips, pairs[0]])
    if sh >= 0:
        return _pack(STRAIGHT, [sh])
    if trips >= 0:
        kickers = [r for r in _top_n(rank_mask, 3) if r != trips][:2]
        return _pack(TRIPS, [trips] + kickers)
    if len(pairs) >= 2:
        hi, lo = pairs[0], pairs[1]
        kicker = max(r for r in counts if r != hi and r != lo)
        return _pack(TWO_PAIR, [hi, lo, kicker])
    if pairs:
        p = pairs[0]
        kickers = [r for r in _top_n(rank_mask, 4) if r != p][:3]
        return _pack(PAIR, [p] + kickers)
    return _pack(HIGH_CARD, _top_n(rank_mask, 5))


def category_of(score: int) -> int:
    return score >> 20


def kickers_of(score: int) -> List[int]:
    return [(score >> shift) & 0xF for shift in (16, 12, 8, 4, 0)]


def describe(score: int) -> str:
    """Human-readable hand name, e.g. ``"full house, kings full of nines"``."""
    cat = category_of(score)
    k = kickers_of(score)
    R = RANK_CHARS
    plural = {"6": "6s", "T": "tens", "J": "jacks", "Q": "queens", "K": "kings", "A": "aces"}

    def many(r: int) -> str:
        ch = R[r]
        return plural.get(ch, ch + "s")

    if cat == STRAIGHT_FLUSH:
        return "royal flush" if k[0] == 12 else f"straight flush, {R[k[0]]}-high"
    if cat == QUADS:
        return f"four of a kind, {many(k[0])}"
    if cat == FULL_HOUSE:
        return f"full house, {many(k[0])} full of {many(k[1])}"
    if cat == FLUSH:
        return f"flush, {R[k[0]]}-high"
    if cat == STRAIGHT:
        return f"straight, {R[k[0]]}-high"
    if cat == TRIPS:
        return f"three of a kind, {many(k[0])}"
    if cat == TWO_PAIR:
        return f"two pair, {many(k[0])} and {many(k[1])}"
    if cat == PAIR:
        return f"pair of {many(k[0])}"
    return f"{R[k[0]]}-high"


def best_five(cards: Sequence[int]) -> Tuple[int, List[int]]:
    """Return ``(score, five cards)`` — the actual cards that make the hand."""
    if len(cards) == 5:
        return evaluate(cards), list(cards)
    best, combo = -1, None
    for c in combinations(cards, 5):
        s = evaluate(c)
        if s > best:
            best, combo = s, list(c)
    return best, combo or []

"""Card primitives.

A card is a plain ``int`` in ``[0, 52)`` encoded as ``rank * 4 + suit`` so that
integer ordering is rank-major.  Ranks are ``0..12`` for deuce..ace, suits are
``0..3`` for clubs, diamonds, hearts, spades.  Ints keep the evaluator and the
feature encoders allocation-free in the hot loops.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}

NUM_CARDS = 52
DECK: List[int] = list(range(NUM_CARDS))

RANK_NAMES = [
    "deuce", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "jack", "queen", "king", "ace",
]


def rank_of(card: int) -> int:
    return card >> 2


def suit_of(card: int) -> int:
    return card & 3


def make_card(rank: int, suit: int) -> int:
    return (rank << 2) | suit


def card_str(card: int) -> str:
    return RANK_CHARS[rank_of(card)] + SUIT_CHARS[suit_of(card)]


def card_pretty(card: int) -> str:
    return RANK_CHARS[rank_of(card)] + SUIT_SYMBOLS[SUIT_CHARS[suit_of(card)]]


def cards_str(cards: Iterable[int]) -> str:
    return " ".join(card_str(c) for c in cards)


def parse_card(text: str) -> int:
    text = text.strip()
    if len(text) != 2:
        raise ValueError(f"bad card {text!r}")
    rank = RANK_CHARS.find(text[0].upper())
    suit = SUIT_CHARS.find(text[1].lower())
    if rank < 0 or suit < 0:
        raise ValueError(f"bad card {text!r}")
    return make_card(rank, suit)


def parse_cards(text: str | Sequence[str]) -> List[int]:
    if not isinstance(text, str):
        return [parse_card(t) for t in text]
    text = text.replace(",", " ").strip()
    if " " in text:
        parts = text.split()
    else:
        parts = [text[i:i + 2] for i in range(0, len(text), 2)]
    return [parse_card(p) for p in parts if p]


def hole_label(cards: Sequence[int]) -> str:
    """Canonical 169-bucket label for two hole cards, e.g. ``AKs``/``AKo``/``TT``."""
    if len(cards) != 2:
        raise ValueError("hole_label needs exactly two cards")
    a, b = sorted(cards, key=rank_of, reverse=True)
    ra, rb = rank_of(a), rank_of(b)
    if ra == rb:
        return RANK_CHARS[ra] * 2
    suited = "s" if suit_of(a) == suit_of(b) else "o"
    return RANK_CHARS[ra] + RANK_CHARS[rb] + suited


HOLE_LABELS: List[str] = []
for _i in range(12, -1, -1):
    for _j in range(12, -1, -1):
        if _i == _j:
            HOLE_LABELS.append(RANK_CHARS[_i] * 2)
        elif _i > _j:
            HOLE_LABELS.append(RANK_CHARS[_i] + RANK_CHARS[_j] + "s")
        else:
            HOLE_LABELS.append(RANK_CHARS[_j] + RANK_CHARS[_i] + "o")
HOLE_LABEL_INDEX = {label: i for i, label in enumerate(sorted(set(HOLE_LABELS)))}
assert len(HOLE_LABEL_INDEX) == 169


class Deck:
    """Shuffled deck with a caller-supplied RNG so games are reproducible."""

    __slots__ = ("_cards", "_pos", "rng")

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self._cards = list(DECK)
        self._pos = 0
        self.shuffle()

    def shuffle(self) -> None:
        self.rng.shuffle(self._cards)
        self._pos = 0

    def remove(self, cards: Iterable[int]) -> None:
        """Pull specific cards out of the undealt remainder (for replays)."""
        wanted = set(cards)
        rest = [c for c in self._cards[self._pos:] if c not in wanted]
        self._cards = self._cards[:self._pos] + rest
        if len(self._cards) != self._pos + len(rest):  # pragma: no cover - defensive
            raise ValueError("deck corruption")

    def stack_next(self, cards: Iterable[int]) -> None:
        """Force ``cards`` to be the next ones dealt, in order (used by replays)."""
        cards = list(cards)
        self.remove(cards)
        self._cards = self._cards[:self._pos] + cards + self._cards[self._pos:]

    def deal(self, n: int = 1) -> List[int]:
        if self._pos + n > len(self._cards):
            raise ValueError("deck exhausted")
        out = self._cards[self._pos:self._pos + n]
        self._pos += n
        return out

    def deal_one(self) -> int:
        return self.deal(1)[0]

    @property
    def remaining(self) -> int:
        return len(self._cards) - self._pos

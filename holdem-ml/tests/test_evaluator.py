import random
from itertools import combinations

import numpy as np
import pytest

from holdem.cards import parse_cards as P
from holdem.evaluator import (
    FLUSH, FULL_HOUSE, HIGH_CARD, PAIR, QUADS, STRAIGHT, STRAIGHT_FLUSH, TRIPS,
    TWO_PAIR, best_five, category_of, describe, evaluate,
)
from holdem.evaluator_np import evaluate_batch


def test_category_ordering():
    hands = [
        ("7c2d4h9sJc", HIGH_CARD), ("7c7d4h9sJc", PAIR), ("7c7d4h4sJc", TWO_PAIR),
        ("7c7d7h4sJc", TRIPS), ("5c6d7h8s9c", STRAIGHT), ("2s7s9sJsKs", FLUSH),
        ("7c7d7h4s4c", FULL_HOUSE), ("7c7d7h7s4c", QUADS), ("5s6s7s8s9s", STRAIGHT_FLUSH),
    ]
    scores = [evaluate(P(h)) for h, _ in hands]
    assert [category_of(s) for s in scores] == [c for _, c in hands]
    assert scores == sorted(scores), "categories must rank in order"


def test_wheel_is_the_lowest_straight():
    assert evaluate(P("Ac2d3h4s5c")) < evaluate(P("2d3h4s5c6d"))
    assert category_of(evaluate(P("Ac2d3h4s5c"))) == STRAIGHT
    assert describe(evaluate(P("As2s3s4s5s"))) == "straight flush, 5-high"
    assert describe(evaluate(P("AsKsQsJsTs"))) == "royal flush"


def test_kickers_matter():
    assert evaluate(P("AcAdKh7s2c")) > evaluate(P("AcAdQh7s2c"))
    assert evaluate(P("AcAdKh7s2c")) == evaluate(P("AsAhKd7c2d"))


@pytest.mark.parametrize("n_cards", [5, 6, 7])
def test_vectorised_matches_scalar(n_cards):
    rng = random.Random(7)
    hands = np.array([rng.sample(range(52), n_cards) for _ in range(4000)])
    assert (evaluate_batch(hands) == np.array([evaluate(h) for h in hands])).all()


def test_seven_card_picks_the_best_five():
    rng = random.Random(11)
    for _ in range(300):
        cards = rng.sample(range(52), 7)
        best, five = best_five(cards)
        assert best == evaluate(cards)
        assert evaluate(five) == best


def test_exhaustive_five_card_frequencies():
    """Every published five-card hand frequency, checked over all C(52,5)."""
    counts = {}
    for hand in combinations(range(52), 5):
        c = category_of(evaluate(hand))
        counts[c] = counts.get(c, 0) + 1
    assert counts == {
        HIGH_CARD: 1302540, PAIR: 1098240, TWO_PAIR: 123552, TRIPS: 54912,
        STRAIGHT: 10200, FLUSH: 5108, FULL_HOUSE: 3744, QUADS: 624,
        STRAIGHT_FLUSH: 40,
    }

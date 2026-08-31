import numpy as np
import pytest

from holdem.cards import parse_cards as P
from holdem.equity import equity, equity_vs_range, fast_equity, preflop_equity


@pytest.mark.parametrize("hand,opponents,expected", [
    ("AsAd", 1, 0.852),   # published all-in equities
    ("AsAd", 5, 0.490),
    ("AsKs", 1, 0.670),
    ("7c2d", 1, 0.351),
    ("JhTh", 1, 0.577),
])
def test_preflop_equities_match_published_values(hand, opponents, expected):
    got = equity(P(hand), (), opponents, iters=30000,
                 rng=np.random.default_rng(0))
    assert got == pytest.approx(expected, abs=0.015)


def test_equity_between_two_specific_hands_is_zero_sum():
    board = P("Kh7c2d")
    aces, kings = P("AsAd"), P("KsKd")
    a = equity_vs_range(aces, board, [[kings]], iters=600,
                        rng=np.random.default_rng(1))
    b = equity_vs_range(kings, board, [[aces]], iters=600,
                        rng=np.random.default_rng(2))
    assert a + b == pytest.approx(1.0, abs=0.02)
    assert b > 0.9, "a set of kings crushes aces on this board"


def test_the_nuts_wins_everything():
    assert equity(P("AsKs"), P("QsJsTs"), 1, iters=3000) > 0.75
    assert equity(P("2h2d"), P("AsKsQsJs"), 1, iters=3000) < 0.25


def test_more_opponents_means_less_equity():
    values = [equity(P("AsAd"), (), n, iters=8000, rng=np.random.default_rng(2))
              for n in (1, 3, 6)]
    assert values[0] > values[1] > values[2]


def test_preflop_table_agrees_with_simulation():
    for hand in ("AsAd", "7h2c", "KsQs"):
        table = preflop_equity(P(hand), 2)
        mc = equity(P(hand), (), 2, iters=20000, rng=np.random.default_rng(3))
        assert table == pytest.approx(mc, abs=0.02)


def test_fast_equity_caches_postflop():
    a = fast_equity(P("AsAd"), P("Kh7c2d"), 1, iters=800)
    b = fast_equity(P("AsAd"), P("Kh7c2d"), 1, iters=800)
    assert a == b, "repeat lookups should hit the memo"


def test_equity_vs_an_explicit_range():
    tight = [P("AsAd"), P("KsKd"), P("QsQd")]
    wide = [P("7c2d"), P("8c3d"), P("9c4d")]
    hero = P("JhJc")
    assert equity_vs_range(hero, P("2h5s9d"), [wide], iters=400) > \
           equity_vs_range(hero, P("2h5s9d"), [tight], iters=400)

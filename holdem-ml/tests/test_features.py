import random

import numpy as np

from holdem.engine import Action, ActionType, PlayerState, Table
from holdem.ml.features import (
    FEATURE_NAMES, NUM_FEATURES, NUM_PUBLIC_FEATURES, encode, encode_public,
    last_aggressor, preflop_aggressor, street_context,
)


def table(seed=2, n=4):
    players = [PlayerState(i, f"P{i}", 200) for i in range(n)]
    t = Table(players, sb=1, bb=2, button=0, rng=random.Random(seed))
    t.start_hand()
    return t


def test_encoding_shape_and_range():
    t = table()
    v = encode(t.observation())
    assert v.shape == (NUM_FEATURES,)
    assert len(FEATURE_NAMES) == NUM_FEATURES
    assert np.isfinite(v).all()
    assert v.min() >= -1.0 and v.max() <= 2.0, "features are meant to be roughly [0,1]"


def test_public_encoding_never_needs_hole_cards():
    t = table()
    obs = t.observation()
    stripped = obs.__class__(**{**obs.__dict__, "hole": []})
    v = encode_public(stripped)
    assert v.shape == (NUM_PUBLIC_FEATURES,)
    assert np.isfinite(v).all()


def test_encoding_is_deterministic_for_the_same_spot():
    t = table()
    obs = t.observation()
    a = encode(obs, equity_field=0.5, equity_heads_up=0.5)
    b = encode(obs, equity_field=0.5, equity_heads_up=0.5)
    assert np.array_equal(a, b)


def test_street_context_counts_the_action():
    t = table()
    t.apply(Action(ActionType.RAISE, 6))
    t.apply(Action(ActionType.CALL, 6))
    ctx = street_context(t.observation())
    assert ctx["raises"] == 1
    assert ctx["callers"] == 1
    assert preflop_aggressor(t.observation()) == 3
    assert last_aggressor(t.observation()) == 3


def test_equity_feature_tracks_hand_strength():
    from holdem.cards import parse_cards as P

    players = [PlayerState(i, f"P{i}", 200) for i in range(2)]
    t = Table(players, sb=1, bb=2, button=0, rng=random.Random(1))
    t.start_hand(hole_cards={0: P("AcAd"), 1: P("7c2d")})
    strong = encode(t.observation(0))
    weak = encode(t.observation(1))
    i = FEATURE_NAMES.index("equity_vs_field")
    assert strong[i] > weak[i]

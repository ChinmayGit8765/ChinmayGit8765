import numpy as np
import pytest

from holdem.ml.difficulty import (
    AdaptiveDifficulty, Difficulty, LEVELS, LEVELS_BY_NAME, SkillTracker,
    difficulty_at, resolve_difficulty,
)


def test_levels_are_monotonic():
    for a, b in zip(LEVELS, LEVELS[1:]):
        assert a.level < b.level
        assert a.temperature > b.temperature, "stronger levels follow the policy more closely"
        assert a.blunder_rate >= b.blunder_rate
        assert a.exploit_weight <= b.exploit_weight
        assert a.equity_iters <= b.equity_iters


def test_interpolation_between_presets():
    mid = difficulty_at(0.375)
    casual, regular = LEVELS_BY_NAME["casual"], LEVELS_BY_NAME["regular"]
    assert casual.temperature > mid.temperature > regular.temperature
    assert difficulty_at(0.0).temperature == LEVELS[0].temperature
    assert difficulty_at(1.0).temperature == LEVELS[-1].temperature


def test_resolve_accepts_names_numbers_and_objects():
    assert resolve_difficulty("pro").level == 1.0
    assert resolve_difficulty(0.5).level == pytest.approx(0.5)
    assert resolve_difficulty("0.25").level == pytest.approx(0.25)
    d = Difficulty("x", 0.3, 1, 0, 0, 1, 10, 0)
    assert resolve_difficulty(d) is d
    with pytest.raises(ValueError):
        resolve_difficulty("impossible")


def test_skill_tracker_separates_good_and_bad_play():
    reference = np.zeros(7)
    reference[[1, 2, 4]] = [0.6, 0.3, 0.1]
    good, bad = SkillTracker("good"), SkillTracker("bad")
    for _ in range(200):
        good.observe_decision(reference, 1)   # always the model's top choice
        bad.observe_decision(reference, 4)    # always the model's least likely
        good.observe_hand(4, 2)
        bad.observe_hand(-4, 2)
    healthy = {"vpip": 0.24, "pfr": 0.19, "fold_to_bet": 0.45,
               "aggression_factor": 2.0, "wtsd": 0.27}
    leaky = {"vpip": 0.62, "pfr": 0.05, "fold_to_bet": 0.78,
             "aggression_factor": 0.3, "wtsd": 0.5}
    assert good.raw_rating(healthy) > bad.raw_rating(leaky) + 0.2
    assert 0.0 <= bad.raw_rating(leaky) <= 1.0

    # The smoothed rating converges toward the instantaneous one, and reading
    # it must not move it.
    for _ in range(60):
        good.update(healthy)
        bad.update(leaky)
    assert good.rating > bad.rating + 0.2
    assert good.rating == good.rating
    assert good.leak_score(healthy) > good.leak_score(leaky)
    assert "skill" in good.report(healthy)


def test_confidence_grows_with_evidence():
    t = SkillTracker("x")
    assert t.confidence == 0.0
    reference = np.full(7, 1 / 7)
    for _ in range(120):
        t.observe_decision(reference, 0)
    assert t.confidence == 1.0


def test_adaptive_difficulty_tracks_the_player():
    weak = AdaptiveDifficulty(start=0.5)
    for _ in range(80):
        weak.update(0.15, confidence=1.0)
    assert weak.level < 0.3, "bots ease off against a weaker player"

    strong = AdaptiveDifficulty(start=0.3)
    for _ in range(80):
        strong.update(0.9, confidence=1.0)
    assert strong.level > 0.8, "bots step up against a stronger player"
    assert strong.current().level == pytest.approx(strong.level)


def test_adaptive_difficulty_moves_gradually():
    ad = AdaptiveDifficulty(start=0.2, rate=0.06)
    before = ad.level
    ad.update(1.0, confidence=1.0)
    assert ad.level - before <= 0.061, "difficulty must not jump in one hand"

import random

import numpy as np

from holdem.bots.rule import CallingStation, EquityBot, LooseAggressive, TightRock
from holdem.game import Game
from holdem.ml.opponent import (
    ActionPredictor, DecayCounter, OpponentModel, OpponentTracker, PREDICTOR_INPUT,
    RatioCounter,
)


def test_decay_counter_tracks_a_rate():
    c = DecayCounter(decay=0.99, prior=0.5, prior_weight=1.0)
    for _ in range(200):
        c.observe(True)
    assert c.value > 0.95
    for _ in range(200):
        c.observe(False)
    # decay=0.99 has a ~69-observation half-life, so a fully reversed player
    # lands near 0.12 rather than at 0 — the point is that it moved.
    assert c.value < 0.15, "the counter follows a player who changes gear"


def test_ratio_counter():
    r = RatioCounter()
    for _ in range(50):
        r.observe_top()
    assert r.value > 3
    assert 0 <= r.normalised <= 1


def test_action_predictor_learns_a_rule():
    rng = np.random.default_rng(0)
    p = ActionPredictor(rng=rng)
    X = rng.random((500, PREDICTOR_INPUT)).astype(np.float32)
    y = (X[:, 6] > 0.5).astype(int) * 2
    for x, label in zip(X, y):
        p.observe(x, int(label), train_every=1)
    correct = np.mean([int(np.argmax(p.predict(x)) == label) for x, label in zip(X[:150], y[:150])])
    assert correct > 0.85, f"online predictor should learn the rule (got {correct:.2f})"
    assert p.trained


def test_tracker_profiles_distinct_playing_styles():
    rng = random.Random(3)
    bots = [CallingStation("station", rng), TightRock("rock", rng),
            LooseAggressive("lag", rng), EquityBot("equity", rng)]
    tracker = OpponentTracker(learn=False)
    game = Game(bots, rng=rng,
                on_hand_end=lambda r: tracker.on_hand_end(r))
    original = game.play_hand

    for _ in range(220):
        result = original(keep_history=False)
        from holdem.analysis.replay import replay_hand
        for dp in replay_hand(result):
            tracker.on_action(dp.record, dp.obs)

    station = tracker.get("station").stats()
    rock = tracker.get("rock").stats()
    lag = tracker.get("lag").stats()
    assert station["vpip"] > rock["vpip"], "the station plays far more hands than the rock"
    assert station["fold_to_bet"] < rock["fold_to_bet"], "the station does not fold"
    assert lag["aggression_factor"] > station["aggression_factor"]
    assert tracker.get("station").label().startswith("loose")
    assert "player" in tracker.report()


def test_model_predictions_are_a_distribution():
    rng = random.Random(1)
    game = Game([EquityBot("a", rng), CallingStation("b", rng)], rng=rng)
    result = game.play_hand()
    from holdem.analysis.replay import replay_hand
    model = OpponentModel("b", rng=np.random.default_rng(0))
    for dp in replay_hand(result):
        if dp.name == "b":
            model.observe_action(dp.record, dp.obs)
            probs = model.predict_action(dp.obs)
            assert probs.shape == (3,)
            assert abs(probs.sum() - 1.0) < 0.05
            assert (probs >= 0).all()

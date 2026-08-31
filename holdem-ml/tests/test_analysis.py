import os
import random

import numpy as np
import pytest

from holdem.analysis.analyzer import Analyzer, LEAK_RULES
from holdem.analysis.corpus import (
    load_corpus, samples_from_histories, samples_from_results, save_corpus, stack_samples,
)
from holdem.analysis.handhistory import format_hand
from holdem.analysis.promodel import DEFAULT_PRO_MODEL, ProModel, load_pro_model, save_pro_model
from holdem.bots.rule import CallingStation, EquityBot, LooseAggressive, TightRock
from holdem.cards import parse_cards as P
from holdem.game import Game
from holdem.ml.abstraction import NUM_ACTIONS
from holdem.ml.features import NUM_FEATURES


def session(hands=60, seed=31):
    rng = random.Random(seed)
    bots = [EquityBot("Hero", rng), LooseAggressive("Villain", rng),
            TightRock("Nit", rng), CallingStation("Fish", rng)]
    game = Game(bots, rng=rng)
    for _ in range(hands):
        game.play_hand()
    return game


def test_corpus_extraction_shapes():
    game = session(hands=30)
    samples = samples_from_results(game.history, equity_iters=80)
    assert samples
    X, M, A, R = stack_samples(samples)
    assert X.shape == (len(samples), NUM_FEATURES)
    assert M.shape == (len(samples), NUM_ACTIONS)
    assert ((0 <= A) & (A < NUM_ACTIONS)).all()
    assert all(M[i, A[i]] for i in range(len(A))), "the action taken must be legal"


def test_corpus_save_and_load(tmp_path):
    game = session(hands=20)
    samples = samples_from_results(game.history, equity_iters=60)
    path = str(tmp_path / "corpus.npz")
    save_corpus(path, samples)
    X, M, A, R, street = load_corpus(path)
    assert len(X) == len(samples)
    assert street.max() <= 3


def test_corpus_from_real_hand_history_files(tmp_path):
    game = session(hands=25)
    text = "\n\n".join(format_hand(r, sb=1, bb=2) for r in game.history)
    path = tmp_path / "hh.txt"
    path.write_text(text)
    samples = samples_from_histories(str(tmp_path), equity_iters=60)
    assert samples, "imported hand histories should yield training samples"
    assert all(0 <= s.action < NUM_ACTIONS for s in samples)


def test_pro_model_heads():
    model = ProModel(rng=np.random.default_rng(0))
    x = np.random.default_rng(1).random((6, NUM_FEATURES)).astype(np.float32)
    policy, q, value = model.forward(x, training=False)
    assert policy.shape == (6, NUM_ACTIONS)
    assert q.shape == (6, NUM_ACTIONS)
    assert value.shape == (6, 1)
    mask = np.array([True, True, False, True, False, False, True])
    probs, values, spot = model.assess(x[0], mask)
    assert probs.sum() == pytest.approx(1.0)
    assert (probs[~mask] == 0).all()
    assert np.isneginf(values[~mask]).all()


def test_pro_model_save_load(tmp_path):
    model = ProModel(hidden=(32, 16), rng=np.random.default_rng(0))
    path = str(tmp_path / "pro.npz")
    save_pro_model(path, model, {"note": "t"})
    other = load_pro_model(path)
    x = np.random.default_rng(2).random((3, NUM_FEATURES)).astype(np.float32)
    assert np.allclose(model.forward(x, training=False)[0],
                       other.forward(x, training=False)[0])


def test_pro_model_learns_action_values():
    """The Q head must fit the returns of the action actually taken."""
    from holdem.train.train_analyst import train

    rng = np.random.default_rng(0)
    n = 400
    X = rng.random((n, NUM_FEATURES)).astype(np.float32)
    M = np.ones((n, NUM_ACTIONS), dtype=bool)
    A = rng.integers(0, NUM_ACTIONS, n)
    # Return depends on the action in a way the model can learn.
    R = (A * 2.0 - 6.0 + X[:, 0]).astype(np.float32)
    model = ProModel(hidden=(64, 32), rng=rng)
    train(model, X, M, A, R, epochs=40, lr=3e-3, rng=rng)
    _, q, _ = model.forward(X[:1], training=False)
    predicted = q[0]
    assert predicted.argmax() >= 5, "the highest-return action should score highest"
    assert predicted[0] < predicted[-1]


def test_analyzer_runs_without_a_trained_model():
    game = session(hands=25)
    analyzer = Analyzer(model=None)
    analyzer.model = None
    report = analyzer.analyse_session(game.history, "Hero")
    assert report.hands > 0
    assert report.decisions > 0
    assert "session report" in report.summary()


def test_analyzer_grades_a_hand():
    game = session(hands=25)
    analyzer = Analyzer(model=ProModel(rng=np.random.default_rng(0)))
    graded = 0
    for result in game.history:
        report = analyzer.analyse_hand(result, "Hero")
        for decision in report.decisions:
            graded += 1
            assert decision.name == "Hero"
            assert decision.ev_loss >= 0
            assert decision.pro_probs.sum() == pytest.approx(1.0)
            assert "chose" in decision.describe()
    assert graded > 10


def test_session_report_aggregates_by_street():
    game = session(hands=40)
    analyzer = Analyzer(model=ProModel(rng=np.random.default_rng(0)))
    report = analyzer.analyse_session(game.history, "Hero")
    assert report.ev_loss_total == pytest.approx(sum(report.ev_loss_by_street.values()))
    assert 0.0 <= report.agreement <= 1.0
    assert report.stats["hands"] > 0


def test_leak_detection_flags_a_maniac():
    analyzer = Analyzer(model=None)
    leaks = analyzer.find_leaks(
        {"vpip": 0.8, "pfr": 0.02, "fold_to_bet": 0.85, "aggression_factor": 0.2,
         "wtsd": 0.5, "three_bet": 0.0, "hands": 200}, [])
    assert len(leaks) >= 3
    assert any("too many hands" in leak for leak in leaks)
    assert any("easy to bluff" in leak for leak in leaks)


def test_leak_detection_is_quiet_for_solid_stats():
    analyzer = Analyzer(model=None)
    leaks = analyzer.find_leaks(
        {"vpip": 0.24, "pfr": 0.19, "fold_to_bet": 0.45, "aggression_factor": 2.2,
         "wtsd": 0.27, "three_bet": 0.07, "hands": 200}, [])
    assert leaks == []


def test_assess_cards_from_an_image_style_input():
    analyzer = Analyzer(model=None)
    strong = analyzer.assess_cards(P("AcAd"), P("Ah7c2d"), opponents=1)
    weak = analyzer.assess_cards(P("7c2d"), P("Ah8c3s"), opponents=1)
    assert strong["equity"] > weak["equity"]
    assert "three of a kind" in strong["made_hand"]
    assert "bet for value" in strong["advice"]
    assert "fold" in weak["advice"] or "check" in weak["advice"]


def test_advice_accounts_for_the_number_of_opponents():
    analyzer = Analyzer(model=None)
    heads_up = analyzer.assess_cards(P("AcAd"), (), opponents=1)
    six_way = analyzer.assess_cards(P("AcAd"), (), opponents=5)
    assert six_way["equity"] < heads_up["equity"]
    assert "value" in six_way["advice"], "aces are still a value bet six ways"


def test_pot_odds_drive_the_call_fold_decision():
    analyzer = Analyzer(model=None)
    cheap = analyzer.assess_cards(P("9h8h"), P("7c6d2s"), opponents=1, pot=100, to_call=5)
    expensive = analyzer.assess_cards(P("9h8h"), P("7c6d2s"), opponents=1, pot=10, to_call=50)
    assert "fold" not in cheap["advice"]
    assert "fold" in expensive["advice"]


@pytest.mark.skipif(not os.path.exists(DEFAULT_PRO_MODEL), reason="analyser not trained")
def test_trained_analyser_prefers_the_obvious_play():
    analyzer = Analyzer()
    assert analyzer.model is not None
    game = session(hands=15)
    report = analyzer.analyse_session(game.history, "Fish")
    assert report.decisions > 0
    assert report.ev_loss_total >= 0

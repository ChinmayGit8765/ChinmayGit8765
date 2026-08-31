"""Integration tests for the shipped, trained models.

They are skipped if the weights are absent, so a fresh clone that has not run
the training scripts still has a green suite.
"""

import os
import random

import numpy as np
import pytest

from holdem.analysis.analyzer import Analyzer
from holdem.analysis.promodel import DEFAULT_PRO_MODEL
from holdem.bots.neural import DEFAULT_POLICY, NeuralBot
from holdem.bots.rule import CallingStation, EquityBot, LooseAggressive, RandomBot, TightRock
from holdem.cards import parse_cards as P
from holdem.game import Game
from holdem.ml.policy import load_policy
from holdem.vision.cardnet import DEFAULT_CARDNET
from holdem.vision.detect import CardReader
from holdem.vision.render import HOLDOUT_STYLES, TRAIN_STYLES, render_board

requires_policy = pytest.mark.skipif(not os.path.exists(DEFAULT_POLICY),
                                     reason="policy not trained")
requires_cardnet = pytest.mark.skipif(not os.path.exists(DEFAULT_CARDNET),
                                      reason="card model not trained")
requires_analyst = pytest.mark.skipif(not os.path.exists(DEFAULT_PRO_MODEL),
                                      reason="analyser not trained")


def duel(hero, villain_cls, hands=1200, seed=101):
    rng = random.Random(seed)
    villain = villain_cls("villain", rng)
    game = Game([hero, villain], rng=rng)
    game.run(hands, keep_history=False)
    return game.stats.bb_per_100(0, game.bb)


@requires_policy
@pytest.mark.parametrize("villain", [EquityBot, TightRock, CallingStation,
                                     LooseAggressive, RandomBot])
def test_trained_bot_beats_every_baseline_heads_up(villain):
    rng = random.Random(7)
    hero = NeuralBot("hero", difficulty="pro", rng=rng)
    assert hero.policy is not None, "the shipped policy should have loaded"
    result = duel(hero, villain)
    assert result > 0, f"pro bot lost to {villain.__name__} ({result:+.0f} bb/100)"


@requires_policy
def test_pro_beats_novice():
    rng = random.Random(8)
    policy = load_policy(DEFAULT_POLICY)
    pro = NeuralBot("pro", policy=policy, difficulty="pro", rng=rng, load_defaults=False)
    novice = NeuralBot("novice", policy=policy, difficulty="novice", rng=rng,
                       load_defaults=False)
    game = Game([pro, novice], rng=rng)
    game.run(1500, keep_history=False)
    assert game.stats.net[0] > 0, "the difficulty ladder must actually be a ladder"


@requires_policy
def test_trained_bot_wins_a_multiway_table():
    rng = random.Random(9)
    hero = NeuralBot("hero", difficulty="pro", rng=rng)
    table = [hero, EquityBot("equity", rng), TightRock("rock", rng),
             CallingStation("station", rng), LooseAggressive("lag", rng)]
    game = Game(table, rng=rng)
    # No-limit swings are enormous; a few hundred hands says nothing.
    game.run(3000, keep_history=False)
    assert game.leaderboard()[0][0] == "hero", "the trained bot should top the table"
    assert game.stats.bb_per_100(0, game.bb) > 0


@requires_cardnet
def test_card_model_reads_clean_cards_it_has_never_seen():
    from holdem.vision.cardnet import load_cardnet
    from holdem.vision.dataset import clean_batch

    net = load_cardnet()
    cards = list(range(52))
    for style in HOLDOUT_STYLES:
        X = clean_batch(cards, style, random.Random(0))
        predicted, _ = net.predict(X)
        accuracy = float(np.mean([p == c for p, c in zip(predicted, cards)]))
        assert accuracy > 0.7, f"{style.name}: only {accuracy:.0%} of cards read"


@requires_cardnet
def test_end_to_end_board_reading():
    reader = CardReader()
    rng = random.Random(3)
    correct = total = 0
    for style in TRAIN_STYLES[:4] + HOLDOUT_STYLES:
        for _ in range(3):
            cards = rng.sample(range(52), 5)
            img = render_board(cards, style, jitter=4, rng=rng)
            found = reader.read_table(img)
            assert len(found) == 5, f"{style.name}: detected {len(found)} cards"
            correct += sum(1 for d, c in zip(found, cards) if d.card == c)
            total += 5
    assert correct / total > 0.9, f"read {correct}/{total} cards correctly"


@requires_cardnet
def test_reading_a_specific_board():
    reader = CardReader()
    cards = P("AsKhQdJcTs")
    img = render_board(cards, TRAIN_STYLES[0])
    found = reader.read_table(img)
    assert [d.card for d in found] == cards
    assert all(d.confidence > 0.5 for d in found)


@requires_analyst
def test_analyser_scores_a_bad_player_worse_than_a_good_one():
    rng = random.Random(11)
    bots = [EquityBot("Solid", rng), RandomBot("Wild", rng), TightRock("Nit", rng)]
    game = Game(bots, rng=rng)
    for _ in range(150):
        game.play_hand()
    analyzer = Analyzer()
    solid = analyzer.analyse_session(game.history, "Solid")
    wild = analyzer.analyse_session(game.history, "Wild")
    assert wild.ev_loss_per_100 > solid.ev_loss_per_100, \
        "a random player must lose more expected value than a solid one"
    assert wild.agreement < solid.agreement, \
        "a random player must agree with the model less often"


@requires_analyst
def test_analyser_reports_leaks_for_a_maniac():
    rng = random.Random(12)
    game = Game([LooseAggressive("Maniac", rng), EquityBot("Solid", rng)], rng=rng)
    for _ in range(120):
        game.play_hand()
    report = Analyzer().analyse_session(game.history, "Maniac")
    assert report.leaks, "a maniac should trip at least one leak rule"
    assert report.decisions > 50
    assert "session report" in report.summary()

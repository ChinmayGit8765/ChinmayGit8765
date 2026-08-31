import os
import random

import numpy as np
import pytest

from holdem.bots.blueprint import BlueprintBot, DEFAULT_BLUEPRINT
from holdem.bots.neural import NeuralBot
from holdem.bots.rule import (
    CallingStation, EquityBot, HonestBot, LooseAggressive, RandomBot, TightRock,
)
from holdem.engine import ActionType
from holdem.game import Game
from holdem.ml.abstraction import legal_mask

ALL_RULE_BOTS = [RandomBot, CallingStation, TightRock, LooseAggressive, EquityBot, HonestBot]


@pytest.mark.parametrize("bot_cls", ALL_RULE_BOTS)
def test_rule_bots_only_make_legal_moves(bot_cls):
    rng = random.Random(5)
    game = Game([bot_cls("a", rng), bot_cls("b", rng), EquityBot("c", rng)], rng=rng)
    game.run(120, keep_history=False)   # the engine raises on an illegal action
    assert game.stats.hands == 120
    assert sum(game.stats.net.values()) == 0


def test_equity_bot_beats_the_random_bot():
    rng = random.Random(2)
    game = Game([EquityBot("equity", rng), RandomBot("random", rng)], rng=rng)
    game.run(1200, keep_history=False)
    assert game.stats.net[0] > 0, "a pot-odds bot must beat a random one"


def test_calling_station_calls_and_the_rock_folds():
    rng = random.Random(4)
    game = Game([CallingStation("station", rng), TightRock("rock", rng)], rng=rng)
    folds = {"station": 0, "rock": 0}
    total = {"station": 0, "rock": 0}
    for _ in range(250):
        result = game.play_hand()
        for record in result.history:
            total[record.name] += 1
            if record.action == ActionType.FOLD:
                folds[record.name] += 1
    station_rate = folds["station"] / max(1, total["station"])
    rock_rate = folds["rock"] / max(1, total["rock"])
    assert station_rate < rock_rate, "the station should fold far less than the rock"


def test_neural_bot_plays_without_trained_models():
    rng = random.Random(6)
    bot = NeuralBot("neural", difficulty="regular", rng=rng, load_defaults=False)
    game = Game([bot, EquityBot("eq", rng), CallingStation("st", rng)], rng=rng)
    game.run(150, keep_history=False)
    assert bot.decisions > 0


def test_neural_bot_probabilities_are_valid():
    rng = random.Random(7)
    bot = NeuralBot("neural", difficulty="pro", rng=rng, load_defaults=False)
    game = Game([bot, EquityBot("eq", rng)], rng=rng)
    from holdem.analysis.replay import replay_hand
    for _ in range(15):
        result = game.play_hand()
        for dp in replay_hand(result):
            if dp.name != "neural":
                continue
            probs = bot.policy_probs(dp.obs)
            mask = legal_mask(dp.obs)
            assert probs.sum() == pytest.approx(1.0, abs=1e-6)
            assert (probs[~mask] == 0).all(), "no weight on illegal actions"


def test_difficulty_changes_how_much_the_bot_blunders():
    rng = random.Random(8)
    novice = NeuralBot("novice", difficulty="novice", rng=rng, load_defaults=False)
    pro = NeuralBot("pro", difficulty="pro", rng=rng, load_defaults=False)
    assert novice.difficulty.blunder_rate > pro.difficulty.blunder_rate
    assert novice.difficulty.temperature > pro.difficulty.temperature
    assert novice.difficulty.exploit_weight < pro.difficulty.exploit_weight


def test_adaptive_bot_moves_its_level_over_a_session():
    rng = random.Random(9)
    bot = NeuralBot("adaptive", difficulty="adaptive", rng=rng, study="fish",
                    load_defaults=False)
    start = bot.adaptive_ctl.level
    game = Game([bot, CallingStation("fish", rng)], rng=rng)
    game.run(150, keep_history=False)
    assert bot.adaptive_ctl is not None
    assert len(bot.adaptive_ctl.trail) > 100, "difficulty is re-evaluated every hand"
    assert bot.adaptive_ctl.level != start
    assert "adaptive" in bot.status()


def test_bot_builds_a_read_on_its_opponent():
    rng = random.Random(10)
    bot = NeuralBot("neural", difficulty="strong", rng=rng, load_defaults=False)
    game = Game([bot, CallingStation("fish", rng), TightRock("nit", rng)], rng=rng)
    game.run(200, keep_history=False)
    fish = bot.tracker.models["fish"]
    nit = bot.tracker.models["nit"]
    assert fish.stats()["vpip"] > nit.stats()["vpip"]
    assert "fish" in bot.read_on("fish")
    assert bot.read_on("nobody").startswith("no read")


def test_bot_never_looks_at_hole_cards_it_should_not_see():
    rng = random.Random(11)
    bot = NeuralBot("neural", difficulty="pro", rng=rng, load_defaults=False)
    game = Game([bot, EquityBot("eq", rng)], rng=rng)
    seen = []
    original = bot.tracker.on_action

    def spy(record, obs):
        seen.append(list(obs.hole))
        original(record, obs)

    bot.tracker.on_action = spy
    game.run(30, keep_history=False)
    assert seen, "the tracker should have seen some actions"
    assert all(h == [] for h in seen), "opponent models must only get public information"


@pytest.mark.skipif(not os.path.exists(DEFAULT_BLUEPRINT), reason="blueprint not trained")
def test_blueprint_bot_finds_most_spots_in_its_table():
    rng = random.Random(12)
    bot = BlueprintBot("bp", rng=rng)
    game = Game([bot, EquityBot("eq", rng)], rng=rng)
    game.run(400, keep_history=False)
    assert bot.hit_rate > 0.7, "the solved table should cover most live spots"


@pytest.mark.skipif(not os.path.exists(DEFAULT_BLUEPRINT), reason="blueprint not trained")
def test_blueprint_bot_beats_the_weakest_opponents():
    rng = random.Random(13)
    bot = BlueprintBot("bp", rng=rng)
    game = Game([bot, CallingStation("station", rng)], rng=rng)
    game.run(1500, keep_history=False)
    assert game.stats.bb_per_100(0, game.bb) > 0

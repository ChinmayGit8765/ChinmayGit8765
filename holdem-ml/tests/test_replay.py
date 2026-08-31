import random

from holdem.bots.rule import CallingStation, EquityBot, LooseAggressive, TightRock
from holdem.game import Game
from holdem.analysis.replay import decisions_by, replay_hand


def session(hands=120, seed=17):
    rng = random.Random(seed)
    bots = [EquityBot("Alice", rng), LooseAggressive("Bob", rng),
            TightRock("Carol", rng), CallingStation("Dan", rng)]
    game = Game(bots, rng=rng)
    for _ in range(hands):
        game.play_hand()
    return game


def test_replay_reproduces_every_decision():
    game = session()
    for result in game.history:
        points = replay_hand(result)
        assert len(points) == len(result.history)
        assert [p.record.action for p in points] == [r.action for r in result.history]
        assert [p.seat for p in points] == [r.seat for r in result.history]


def test_replay_restores_the_original_cards():
    game = session(hands=40)
    for result in game.history:
        for point in replay_hand(result):
            assert point.obs.hole == result.hole[point.seat]
            assert point.obs.board == result.board[:len(point.obs.board)]


def test_replay_context_is_consistent():
    game = session(hands=40)
    for result in game.history:
        for point in replay_hand(result):
            assert point.obs.pot == point.record.pot_before
            assert point.obs.to_call == point.record.to_call


def test_decisions_by_filters_to_one_player():
    game = session(hands=30)
    for result in game.history:
        for point in decisions_by(result, "Alice"):
            assert point.name == "Alice"

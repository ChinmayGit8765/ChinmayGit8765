import random

import pytest

from holdem.bots.rule import CallingStation, EquityBot, RandomBot
from holdem.engine import Action, ActionType
from holdem.game import BaseAgent, Game


class Folder(BaseAgent):
    def act(self, obs):
        if obs.to_call > 0:
            return Action(ActionType.FOLD)
        return Action(ActionType.CHECK)


def test_button_rotates_each_hand():
    rng = random.Random(1)
    game = Game([Folder(f"P{i}") for i in range(4)], rng=rng)
    buttons = []
    for _ in range(8):
        buttons.append(game.button)
        game.play_hand(keep_history=False)
    assert buttons == [0, 1, 2, 3, 0, 1, 2, 3]


def test_chips_are_conserved_across_a_session():
    rng = random.Random(2)
    bots = [EquityBot("a", rng), RandomBot("b", rng), CallingStation("c", rng)]
    game = Game(bots, rng=rng, auto_rebuy=False)
    total = sum(p.stack for p in game.players)
    for _ in range(60):
        if not game.playable():
            break
        game.play_hand(keep_history=False)
        assert sum(p.stack for p in game.players) == total


def test_auto_rebuy_keeps_the_table_going():
    rng = random.Random(3)
    game = Game([RandomBot(f"P{i}", rng) for i in range(3)], starting_stack=20, rng=rng)
    game.run(200, keep_history=False)
    assert game.stats.hands == 200
    assert sum(game.stats.busts.values()) > 0, "short stacks should have busted"


def test_agents_receive_hand_end_and_action_hooks():
    rng = random.Random(4)
    seen = {"actions": 0, "hands": 0}

    class Watcher(Folder):
        def on_action(self, record, obs_public):
            seen["actions"] += 1

        def on_hand_end(self, result, seat):
            seen["hands"] += 1

    game = Game([Watcher("w"), EquityBot("e", rng)], rng=rng)
    game.run(10, keep_history=False)
    assert seen["hands"] == 10
    assert seen["actions"] > 10


def test_observers_never_see_hole_cards():
    rng = random.Random(5)
    holes = []

    class Peeker(Folder):
        def on_action(self, record, obs_public):
            if record.name != self.name:
                holes.append(list(obs_public.hole))

    game = Game([Peeker("p"), EquityBot("e", rng), CallingStation("c", rng)], rng=rng)
    game.run(20, keep_history=False)
    assert holes
    assert all(h == [] for h in holes)


def test_leaderboard_and_report():
    rng = random.Random(6)
    game = Game([EquityBot("winner", rng), RandomBot("loser", rng)], rng=rng)
    game.run(300, keep_history=False)
    board = game.leaderboard()
    assert board[0][1] >= board[-1][1]
    assert "bb/100" in game.report()


def test_needs_two_players():
    with pytest.raises(ValueError):
        Game([Folder("solo")])


def test_a_seeded_session_replays_identically():
    """Regression: Monte-Carlo equity used to draw from OS entropy, so two runs
    of the same seeded session produced different hands."""
    from holdem.bots.neural import NeuralBot
    from holdem.equity import _cached_equity_key

    def run():
        rng = random.Random(4242)
        bots = [NeuralBot("neural", difficulty="strong", rng=rng, load_defaults=False),
                EquityBot("equity", rng), CallingStation("station", rng)]
        game = Game(bots, rng=rng)
        game.run(40)
        return [(r.board, sorted(r.net.items()),
                 [(x.seat, x.action, x.to_amount) for x in r.history])
                for r in game.history]

    first = run()
    _cached_equity_key.cache_clear()
    second = run()
    assert first == second, "the same seed must produce the same session"

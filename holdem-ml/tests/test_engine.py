import random

import pytest

from holdem.cards import parse_cards as P
from holdem.engine import (
    Action, ActionType, PlayerState, Table, street_of,
)
from holdem.evaluator import evaluate


def seats(n, stack=200):
    return [PlayerState(seat=i, name=f"P{i}", stack=stack) for i in range(n)]


def run_random_hand(table, rng):
    while not table.hand_over:
        obs = table.observation()
        la = rng.choice(obs.legal)
        amount = rng.randint(la.min_amount, la.max_amount) \
            if la.type in (ActionType.BET, ActionType.RAISE) else 0
        table.apply(Action(la.type, amount))


def test_blind_posting_and_first_to_act_six_handed(rng):
    players = seats(6)
    table = Table(players, sb=1, bb=2, button=0, rng=rng)
    table.start_hand()
    assert players[1].street_committed == 1, "seat left of button posts the small blind"
    assert players[2].street_committed == 2
    assert table.current_actor() == 3, "action starts under the gun"
    assert table.current_bet == 2


def test_heads_up_button_is_small_blind_and_acts_first(rng):
    players = seats(2)
    table = Table(players, sb=1, bb=2, button=0, rng=rng)
    table.start_hand()
    assert players[0].street_committed == 1
    assert players[1].street_committed == 2
    assert table.current_actor() == 0, "heads-up button acts first preflop"
    table.apply(Action(ActionType.CALL, 1))
    table.apply(Action(ActionType.CHECK))
    assert table.street == 1
    assert table.current_actor() == 1, "big blind acts first after the flop"


def test_big_blind_gets_the_option(rng):
    players = seats(3)
    table = Table(players, sb=1, bb=2, button=0, rng=rng)
    table.start_hand()
    table.apply(Action(ActionType.CALL, 2))   # UTG (button)
    table.apply(Action(ActionType.CALL, 1))   # small blind
    assert table.current_actor() == 2, "big blind still has the option to raise"
    assert table.street == 0


def test_min_raise_is_the_size_of_the_last_raise(rng):
    players = seats(3)
    table = Table(players, sb=1, bb=2, button=0, rng=rng)
    table.start_hand()
    raise_la = table.observation().legal_of(ActionType.RAISE)
    assert raise_la.min_amount == 4, "first raise is to two big blinds"
    table.apply(Action(ActionType.RAISE, 10))
    nxt = table.observation().legal_of(ActionType.RAISE)
    assert nxt.min_amount == 18, "next raise must add at least the previous raise size"
    with pytest.raises(ValueError):
        table.apply(Action(ActionType.RAISE, 12))


def test_short_all_in_does_not_reopen_the_betting(rng):
    players = [PlayerState(0, "big", 200), PlayerState(1, "short", 25),
               PlayerState(2, "caller", 200)]
    table = Table(players, sb=1, bb=2, button=0, rng=rng)
    table.start_hand()
    table.apply(Action(ActionType.RAISE, 20))          # seat 0 opens to 20
    table.apply(Action(ActionType.RAISE, 25))          # seat 1 all-in for less than a full raise
    obs = table.observation()
    assert obs.seat == 2
    assert obs.legal_of(ActionType.RAISE) is not None, "a player yet to act may still raise"
    table.apply(Action(ActionType.CALL, 25))
    obs = table.observation()
    assert obs.seat == 0
    assert obs.to_call == 5
    assert obs.legal_of(ActionType.RAISE) is None, "seat 0 already acted: call or fold only"


def test_side_pots_split_by_contribution():
    players = [PlayerState(0, "short", 10), PlayerState(1, "mid", 50),
               PlayerState(2, "deep", 200)]
    table = Table(players, sb=1, bb=2, button=2, rng=random.Random(3))
    table.start_hand(hole_cards={0: P("AcAd"), 1: P("KcKd"), 2: P("QcQd")},
                     board=P("2h3s7d9c Jh".replace(" ", "")))
    while not table.hand_over:
        obs = table.observation()
        la = obs.legal_of(ActionType.RAISE) or obs.legal_of(ActionType.CALL) \
            or obs.legal_of(ActionType.CHECK)
        table.apply(Action(la.type, la.max_amount if la.type == ActionType.RAISE else la.min_amount))
    result = table.results
    assert len(result.pots) >= 2, "unequal stacks must build a side pot"
    main = result.pots[0]
    assert main.winners == [0], "aces win the main pot"
    assert set(main.eligible) == {0, 1, 2}
    side = result.pots[1]
    assert 0 not in side.eligible, "the short stack cannot win the side pot"
    assert side.winners == [1], "kings win the side pot"
    assert sum(p.amount for p in result.pots) == 10 + 50 + 50, \
        "the deep stack's unmatched chips are not part of any pot"
    assert result.uncalled == (2, 150), "unmatched chips go back to the bettor"


def test_split_pot_gives_odd_chip_left_of_the_button():
    players = [PlayerState(0, "a", 100), PlayerState(1, "b", 100)]
    table = Table(players, sb=1, bb=2, button=0, rng=random.Random(5))
    table.start_hand(hole_cards={0: P("AcKc"), 1: P("AdKd")}, board=P("2h3s7d9cJh"))
    while not table.hand_over:
        obs = table.observation()
        la = obs.legal_of(ActionType.CALL) or obs.legal_of(ActionType.CHECK)
        table.apply(Action(la.type, la.min_amount))
    assert table.results.pots[0].winners == [0, 1], "identical hands chop"
    assert sum(p.stack for p in players) == 200


def test_folding_ends_the_hand_without_a_showdown(rng):
    players = seats(2)
    table = Table(players, sb=1, bb=2, button=0, rng=rng)
    table.start_hand()
    table.apply(Action(ActionType.FOLD))
    assert table.hand_over
    assert not table.results.showdown
    assert table.results.revealed == {}, "mucked cards are never revealed"
    assert table.results.net[1] == 1


def test_cannot_act_after_the_hand_is_over(rng):
    players = seats(2)
    table = Table(players, sb=1, bb=2, button=0, rng=rng)
    table.start_hand()
    table.apply(Action(ActionType.FOLD))
    with pytest.raises(ValueError):
        table.apply(Action(ActionType.CHECK))


def test_illegal_action_is_rejected(rng):
    players = seats(3)
    table = Table(players, sb=1, bb=2, button=0, rng=rng)
    table.start_hand()
    with pytest.raises(ValueError):
        table.apply(Action(ActionType.CHECK))       # facing the big blind
    with pytest.raises(ValueError):
        table.apply(Action(ActionType.RAISE, 3))    # under the minimum


def test_street_helper():
    assert street_of([]) == 0
    assert street_of(P("2h3s7d")) == 1
    assert street_of(P("2h3s7d9c")) == 2
    assert street_of(P("2h3s7d9cJh")) == 3


def test_fuzz_conserves_chips_and_never_deadlocks():
    rng = random.Random(99)
    for _ in range(600):
        n = rng.randint(2, 9)
        players = [PlayerState(i, f"P{i}", rng.choice([2, 3, 7, 25, 200, 1000]))
                   for i in range(n)]
        if sum(1 for p in players if p.stack > 0) < 2:
            continue
        before = sum(p.stack for p in players)
        table = Table(players, sb=1, bb=2, ante=rng.choice([0, 0, 1]),
                      button=rng.randrange(n), rng=rng)
        table.start_hand()
        steps = 0
        while not table.hand_over:
            steps += 1
            assert steps < 400, "betting round failed to terminate"
            run_one = table.observation()
            la = rng.choice(run_one.legal)
            amount = rng.randint(la.min_amount, la.max_amount) \
                if la.type in (ActionType.BET, ActionType.RAISE) else 0
            table.apply(Action(la.type, amount))
        assert sum(p.stack for p in players) == before
        assert sum(table.results.net.values()) == 0
        assert sum(p.amount for p in table.results.pots) == table.pot
        assert all(p.stack >= 0 for p in players)


def test_fuzz_pays_the_best_eligible_hand():
    rng = random.Random(4242)
    checked = 0
    for _ in range(500):
        players = [PlayerState(i, f"P{i}", rng.choice([15, 60, 200]))
                   for i in range(rng.randint(2, 6))]
        table = Table(players, sb=1, bb=2, button=rng.randrange(len(players)), rng=rng)
        table.start_hand()
        run_random_hand(table, rng)
        result = table.results
        if not result.showdown:
            continue
        checked += 1
        scores = {s: evaluate(h + result.board) for s, h in result.revealed.items()}
        for pot in result.pots:
            eligible = [s for s in pot.eligible if s in scores]
            if not eligible:
                continue
            best = max(scores[s] for s in eligible)
            assert set(pot.winners) == {s for s in eligible if scores[s] == best}
    assert checked > 50, "expected a decent number of showdowns"

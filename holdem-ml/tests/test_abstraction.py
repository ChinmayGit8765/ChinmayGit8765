import random

import numpy as np
import pytest

from holdem.cards import parse_cards as P
from holdem.engine import Action, ActionType, PlayerState, Table
from holdem.ml.abstraction import (
    A_ALLIN, A_CALL, A_FOLD, A_RAISE_66, A_RAISE_100, NUM_ACTIONS,
    card_bucket, draw_class, legal_mask, raise_target, strength_percentile,
    to_abstract, to_engine_action,
)


def spot(stacks=(200, 200, 200), button=0, seed=1):
    players = [PlayerState(i, f"P{i}", s) for i, s in enumerate(stacks)]
    table = Table(players, sb=1, bb=2, button=button, rng=random.Random(seed))
    table.start_hand()
    return table


def test_mask_matches_the_engine(rng):
    for seed in range(40):
        table = spot(seed=seed)
        while not table.hand_over:
            obs = table.observation()
            mask = legal_mask(obs)
            assert mask[A_CALL], "check or call is always available"
            assert mask[A_FOLD] == (obs.to_call > 0), "folding for free is not offered"
            for index in range(NUM_ACTIONS):
                if not mask[index]:
                    continue
                action = to_engine_action(obs, index)
                table_legal = {la.type for la in obs.legal}
                assert action.type in table_legal
            table.apply(to_engine_action(obs, A_CALL))


def test_every_masked_action_is_accepted_by_the_engine():
    r = random.Random(4)
    for seed in range(60):
        table = spot(stacks=(200, 60, 25), seed=seed)
        while not table.hand_over:
            obs = table.observation()
            mask = legal_mask(obs)
            choices = [i for i in range(NUM_ACTIONS) if mask[i]]
            table.apply(to_engine_action(obs, r.choice(choices)))


def test_abstract_roundtrip():
    table = spot()
    obs = table.observation()
    for index in [A_FOLD, A_CALL, A_RAISE_66, A_ALLIN]:
        if not legal_mask(obs)[index]:
            continue
        action = to_engine_action(obs, index)
        assert to_abstract(obs, action) == index


def test_raise_target_is_a_pot_fraction():
    table = spot()
    obs = table.observation()          # UTG facing the big blind: pot 3, to call 2
    assert raise_target(obs, 1.0) == obs.current_bet + obs.pot + obs.to_call


def test_all_in_is_recognised_however_it_is_sized():
    table = spot(stacks=(200, 200, 12))
    while table.current_actor() != 2:
        table.apply(Action(ActionType.CALL, table.observation().to_call))
    obs = table.observation()
    la = obs.legal_of(ActionType.RAISE) or obs.legal_of(ActionType.BET)
    assert to_abstract(obs, Action(la.type, la.max_amount)) == A_ALLIN


def test_strength_percentile_orders_hands():
    strong = strength_percentile(P("AcAd"), P("Ah7c2d"))
    medium = strength_percentile(P("Ac9d"), P("Ah7c2d"))
    weak = strength_percentile(P("5c4d"), P("Ah7c2d"))
    assert strong > medium > weak
    assert 0.0 <= weak and strong <= 1.0


def test_draw_classes():
    assert draw_class(P("AsKs"), P("QsJs2d")) == 3, "four to a flush"
    assert draw_class(P("9h8h"), P("7c6d2s")) == 2, "open-ended straight draw"
    assert draw_class(P("Ac2d"), P("Kh8c3s")) == 0, "nothing"
    assert draw_class(P("AcKd"), P("Qh7c2s")) == 0


def test_bucket_is_stable_and_in_range():
    for hole, board in [("AcAd", "Kh7c2d"), ("7c2d", "Kh8c3d"), ("AsKs", "QsJs2d")]:
        b = card_bucket(P(hole), P(board))
        assert 0 <= b < 40
        assert card_bucket(P(hole), P(board)) == b


def test_to_abstract_never_returns_a_masked_action():
    """Regression: sizes that collapse onto all-in are dropped from the mask,
    and mapping a real action onto one of them produced a training target the
    policy was forbidden to predict (a -1e9 cross-entropy term with no gradient).
    """
    from holdem.analysis.replay import replay_hand
    from holdem.bots.rule import CallingStation, EquityBot, LooseAggressive, RandomBot
    from holdem.game import Game

    r = random.Random(5)
    bots = [EquityBot("eq", r), LooseAggressive("lag", r),
            RandomBot("rnd", r), CallingStation("st", r)]
    game = Game(bots, rng=r)
    checked = 0
    for _ in range(120):
        result = game.play_hand()
        for dp in replay_hand(result):
            mask = legal_mask(dp.obs)
            index = to_abstract(dp.obs, dp.action)
            assert mask[index], (
                f"{dp.record.describe()} mapped to a masked action {index}")
            checked += 1
    assert checked > 500


def test_to_abstract_snaps_a_short_stack_raise_to_all_in():
    table = spot(stacks=(200, 200, 9))
    while table.current_actor() != 2:
        obs = table.observation()
        table.apply(Action(ActionType.CALL, obs.to_call))
    obs = table.observation()
    la = obs.legal_of(ActionType.RAISE) or obs.legal_of(ActionType.BET)
    if la is not None:
        # Every raise this player can make is effectively a shove.
        mask = legal_mask(obs)
        assert mask[A_ALLIN]
        index = to_abstract(obs, Action(la.type, la.min_amount))
        assert mask[index]

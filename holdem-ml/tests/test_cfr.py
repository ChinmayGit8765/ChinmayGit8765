import os
import random
import tempfile

import numpy as np
import pytest

from holdem.cards import Deck
from holdem.ml.abstraction import A_FOLD, NUM_ACTIONS
from holdem.ml.cfr import (
    Blueprint, MCCFRTrainer, apply_action, legal_actions, new_hand, payoff,
)

BLUEPRINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "blueprint.npz")


def test_hand_setup_posts_blinds():
    cards = list(range(9))
    st = new_hand(cards, stack=200, sb=1, bb=2)
    assert st.committed == [1, 2]
    assert st.stack == [199, 198]
    assert st.to_act == 0, "heads-up button acts first preflop"
    assert st.to_call(0) == 1


def test_folding_pays_the_other_player():
    st = new_hand(list(range(9)))
    st = apply_action(st, A_FOLD)
    assert st.terminal()
    assert payoff(st, 0) == pytest.approx(-0.5)   # lost the small blind
    assert payoff(st, 1) == pytest.approx(0.5)


def test_payoffs_are_zero_sum():
    rng = random.Random(3)
    for _ in range(200):
        st = new_hand(Deck(rng).deal(9))
        while not st.terminal():
            legal = legal_actions(st)
            st = apply_action(st, rng.choice(legal))
        assert payoff(st, 0) + payoff(st, 1) == pytest.approx(0.0)


def test_a_hand_always_terminates():
    rng = random.Random(9)
    for _ in range(200):
        st = new_hand(Deck(rng).deal(9))
        steps = 0
        while not st.terminal():
            steps += 1
            assert steps < 100
            st = apply_action(st, rng.choice(legal_actions(st)))


def test_regret_matching_is_a_distribution_over_legal_actions():
    bp = Blueprint()
    key = "AA|0200"
    bp.regret[key] = np.array([-5.0, 3.0, 0.0, 7.0, 0.0, 0.0, 0.0])
    probs = bp.strategy(key, [0, 1, 3])
    assert probs[0] == 0.0, "negative regret gets no weight"
    assert probs[1] == pytest.approx(0.3)
    assert probs[3] == pytest.approx(0.7)
    assert probs.sum() == pytest.approx(1.0)
    uniform = bp.strategy("unseen", [0, 1])
    assert uniform[0] == uniform[1] == 0.5


def test_training_learns_to_fold_the_worst_hands():
    trainer = MCCFRTrainer(rng=random.Random(0))
    trainer.iterate(4000)
    bp = trainer.bp
    assert len(bp) > 500
    trash = [k for k in bp.strategy_sum if k.startswith(("72o|", "32o|", "82o|"))]
    assert trash, "expected to have visited some trash hands"
    fold_weight = np.mean([bp.average_strategy(k, range(NUM_ACTIONS))[A_FOLD]
                           for k in trash if k.endswith("0200")] or [0])
    assert fold_weight > 0.3, "CFR should learn to fold the worst preflop hands"


def test_blueprint_save_and_load_roundtrip():
    trainer = MCCFRTrainer(rng=random.Random(1))
    trainer.iterate(400)
    path = os.path.join(tempfile.mkdtemp(), "bp.npz")
    kept = trainer.bp.save(path, min_visits=0.0)
    loaded = Blueprint.load(path)
    assert len(loaded) == kept
    assert loaded.iterations == trainer.bp.iterations
    key = next(iter(loaded.strategy_sum))
    assert np.allclose(loaded.average_strategy(key, range(NUM_ACTIONS)),
                       trainer.bp.average_strategy(key, range(NUM_ACTIONS)), atol=1e-6)


def test_merge_adds_deltas():
    bp = Blueprint()
    bp.merge_({"k": np.ones(NUM_ACTIONS)}, {"k": np.full(NUM_ACTIONS, 2.0)})
    bp.merge_({"k": np.ones(NUM_ACTIONS)}, {"k": np.full(NUM_ACTIONS, 2.0)})
    assert bp.regret["k"][0] == 2.0
    assert bp.strategy_sum["k"][0] == 4.0


@pytest.mark.skipif(not os.path.exists(BLUEPRINT_PATH), reason="blueprint not trained")
def test_shipped_blueprint_folds_trash_and_plays_premiums():
    bp = Blueprint.load(BLUEPRINT_PATH)
    assert len(bp) > 2000
    trash = bp.average_strategy("72o|0200", range(NUM_ACTIONS))
    premium = bp.average_strategy("AA|0200", range(NUM_ACTIONS))
    assert trash[A_FOLD] > 0.6, "the solved strategy folds the worst hand"
    assert premium[A_FOLD] < 0.05, "and never folds the best one"

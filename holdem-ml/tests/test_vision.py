import os
import random

import numpy as np
import pytest
from PIL import Image

from holdem.cards import parse_cards as P, rank_of, suit_of
from holdem.ml.nn import SoftmaxCrossEntropy, numeric_grad_check
from holdem.vision.cardnet import CardNet, DEFAULT_CARDNET, load_cardnet, save_cardnet
from holdem.vision.dataset import IMG_H, IMG_W, augment, clean_batch, make_batch
from holdem.vision.detect import CardReader, card_mask, connected_components, find_cards
from holdem.vision.render import HOLDOUT_STYLES, TRAIN_STYLES, render_board, render_card


def test_renderer_draws_every_card():
    rng = random.Random(0)
    for card in range(52):
        img = render_card(card, TRAIN_STYLES[0], (48, 68), rng)
        assert img.size == (48, 68)
        assert img.mode == "RGB"


def test_styles_are_visually_distinct():
    a = np.asarray(render_card(0, TRAIN_STYLES[0], (48, 68)), dtype=float)
    b = np.asarray(render_card(0, TRAIN_STYLES[4], (48, 68)), dtype=float)
    assert np.abs(a - b).mean() > 2.0, "deck styles must actually differ"


def test_train_and_holdout_styles_are_disjoint():
    train = {s.name for s in TRAIN_STYLES}
    holdout = {s.name for s in HOLDOUT_STYLES}
    assert train and holdout and not (train & holdout)


def test_red_suits_render_red():
    hearts = np.asarray(render_card(P("Ah")[0], TRAIN_STYLES[0], (48, 68)), dtype=float)
    spades = np.asarray(render_card(P("As")[0], TRAIN_STYLES[0], (48, 68)), dtype=float)
    redness = lambda a: (a[:, :, 0] - a[:, :, 2]).mean()
    assert redness(hearts) > redness(spades) + 2


def test_augmentation_shape_and_range():
    rng = random.Random(1)
    base = render_card(7, TRAIN_STYLES[0], (96, 136), rng)
    for _ in range(20):
        arr = augment(base, rng)
        assert arr.shape == (3, IMG_H, IMG_W)
        assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_make_batch_labels_match_the_cards():
    rng = random.Random(2)
    X, yr, ys = make_batch(24, TRAIN_STYLES, rng)
    assert X.shape == (24, 3, IMG_H, IMG_W)
    assert set(np.unique(yr)) <= set(range(13))
    assert set(np.unique(ys)) <= set(range(4))


def test_connected_components_finds_separate_blobs():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 2:6] = True
    mask[10:18, 12:19] = True
    boxes = sorted(connected_components(mask))
    assert len(boxes) == 2
    assert boxes[0] == (2, 2, 6, 6)


@pytest.mark.parametrize("style_index", range(len(TRAIN_STYLES)))
def test_detection_finds_the_right_number_of_cards(style_index):
    rng = random.Random(style_index)
    style = TRAIN_STYLES[style_index]
    for n in (2, 5):
        cards = rng.sample(range(52), n)
        img = render_board(cards, style, jitter=5, rng=rng)
        assert len(find_cards(img)) == n, f"{style.name}: expected {n} cards"


def test_detection_works_on_unseen_deck_styles():
    rng = random.Random(3)
    for style in HOLDOUT_STYLES:
        cards = rng.sample(range(52), 4)
        img = render_board(cards, style, jitter=6, rng=rng)
        assert len(find_cards(img)) == 4


def test_detection_orders_cards_left_to_right():
    img = render_board(P("AsKhQdJc9s"), TRAIN_STYLES[0])
    boxes = find_cards(img)
    assert boxes == sorted(boxes, key=lambda b: b[0])


def test_card_mask_separates_cards_from_felt():
    img = render_board(P("AsKh"), TRAIN_STYLES[0])
    mask = card_mask(img)
    assert 0.05 < mask.mean() < 0.6


def test_cardnet_shapes_and_prediction():
    net = CardNet(channels=(6, 8, 12), hidden=24, rng=np.random.default_rng(0))
    X = np.random.default_rng(1).random((5, 3, IMG_H, IMG_W)).astype(np.float32)
    rank, suit = net.forward(X, training=False)
    assert rank.shape == (5, 13) and suit.shape == (5, 4)
    cards, conf = net.predict(X)
    assert cards.shape == (5,)
    assert ((0 <= conf) & (conf <= 1)).all()
    assert all(0 <= int(c) < 52 for c in cards)


def test_cardnet_gradients():
    net = CardNet(channels=(4, 6, 8), hidden=12, dropout=0.0,
                  rng=np.random.default_rng(1))
    rng = np.random.default_rng(2)
    for p in net.parameters():          # avoid the ReLU kink at zero bias
        if p.value.ndim == 1:
            p.value = (rng.standard_normal(p.value.shape) * 0.3).astype(np.float32)
    X = rng.random((3, 3, IMG_H, IMG_W)).astype(np.float32)
    y = rng.integers(0, 13, 3)

    class RankHeadOnly:
        def forward(self, x, training=True):
            return net.forward(x, training)[0]

        def backward(self, grad):
            return net.backward(grad, np.zeros((grad.shape[0], 4)))

        def parameters(self):
            return net.parameters()

        def zero_grad(self):
            net.zero_grad()

    err = numeric_grad_check(RankHeadOnly(), SoftmaxCrossEntropy(), X, y,
                             eps=1e-6, samples=50)
    assert err < 1e-5


def test_cardnet_can_memorise_a_small_clean_set():
    """A sanity check that the architecture learns at all."""
    from holdem.ml.nn import Adam

    cards = list(range(0, 52, 4))    # one card per rank
    X = clean_batch(cards, TRAIN_STYLES[0])
    yr = np.array([rank_of(c) for c in cards])
    ys = np.array([suit_of(c) for c in cards])
    net = CardNet(channels=(10, 16, 24), hidden=48, dropout=0.0,
                  rng=np.random.default_rng(0))
    opt = Adam(net.parameters(), lr=3e-3)
    ce = SoftmaxCrossEntropy()
    for _ in range(120):
        opt.zero_grad()
        rl, sl = net.forward(X, training=True)
        l1, g1 = ce(rl, yr)
        l2, g2 = ce(sl, ys)
        net.backward(g1, g2)
        opt.step()
    pred_r, _ = net.forward(X, training=False)
    assert (pred_r.argmax(1) == yr).mean() > 0.9


def test_cardnet_save_load_roundtrip(tmp_path):
    net = CardNet(channels=(6, 8, 12), hidden=24, rng=np.random.default_rng(0))
    X = np.random.default_rng(3).random((4, 3, IMG_H, IMG_W)).astype(np.float32)
    net.forward(X, training=True)          # populate batch-norm running stats
    path = str(tmp_path / "cardnet.npz")
    save_cardnet(path, net, {"tag": "test"})
    other = load_cardnet(path)
    assert np.allclose(net.forward(X, training=False)[0],
                       other.forward(X, training=False)[0])


@pytest.mark.skipif(not os.path.exists(DEFAULT_CARDNET), reason="card model not trained")
def test_trained_model_reads_a_rendered_board():
    reader = CardReader()
    cards = P("AsKhQdJc9s")
    for style in TRAIN_STYLES[:3] + HOLDOUT_STYLES[:1]:
        img = render_board(cards, style, jitter=4, rng=random.Random(1))
        found = reader.read_table(img)
        assert len(found) == 5
        correct = sum(1 for d, c in zip(found, cards) if d.card == c)
        assert correct >= 4, f"{style.name}: only read {correct}/5 correctly"

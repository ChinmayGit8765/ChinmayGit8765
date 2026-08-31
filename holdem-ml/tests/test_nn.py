import os
import tempfile

import numpy as np
import pytest

from holdem.ml.nn import (
    Adam, Dropout, HuberLoss, LayerNorm, LeakyReLU, Linear, MSELoss, ReLU, SGD,
    Sequential, Sigmoid, SoftmaxCrossEntropy, Tanh, load, log_softmax, minibatches,
    mlp, numeric_grad_check, one_hot, save, softmax,
)


@pytest.fixture
def np_rng():
    return np.random.default_rng(0)


def smooth_stack(rng):
    """Tanh instead of ReLU: finite differences are exact on a smooth network."""
    return Sequential(Linear(6, 10, rng=rng), LayerNorm(10), Tanh(),
                      Linear(10, 8, rng=rng), Sigmoid(), Linear(8, 4, rng=rng))


def test_cross_entropy_gradients_match_finite_differences(np_rng):
    x = np_rng.standard_normal((12, 6)).astype(np.float32)
    y = np_rng.integers(0, 4, 12)
    err = numeric_grad_check(smooth_stack(np_rng), SoftmaxCrossEntropy(), x, y, samples=60)
    assert err < 1e-6, f"analytic gradient disagrees with numeric ({err:.2e})"


@pytest.mark.parametrize("loss", [MSELoss(), HuberLoss(0.8)])
def test_regression_gradients_match_finite_differences(np_rng, loss):
    x = np_rng.standard_normal((12, 6)).astype(np.float32)
    y = np_rng.standard_normal((12, 4)).astype(np.float32)
    err = numeric_grad_check(smooth_stack(np_rng), loss, x, y, samples=60)
    assert err < 1e-6


def test_relu_gradients_are_correct_away_from_the_kink(np_rng):
    # ReLU is not differentiable at zero, and zero-initialised biases put many
    # units exactly there, so the check uses randomised biases and a small step.
    model = Sequential(Linear(6, 10, rng=np_rng), ReLU(),
                       Linear(10, 8, rng=np_rng), LeakyReLU(0.1), Linear(8, 4, rng=np_rng))
    for p in model.parameters():
        if p.value.ndim == 1:
            p.value = np_rng.standard_normal(p.value.shape).astype(np.float32)
    x = np_rng.standard_normal((12, 6)).astype(np.float32)
    y = np_rng.integers(0, 4, 12)
    err = numeric_grad_check(model, SoftmaxCrossEntropy(), x, y, eps=1e-6, samples=60)
    assert err < 1e-5


def test_softmax_is_stable_and_normalised():
    x = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1000.0, -1000.0]])
    p = softmax(x)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.isfinite(p).all()
    assert np.allclose(np.log(p), log_softmax(x), atol=1e-6)


def test_masked_cross_entropy_ignores_illegal_classes(np_rng):
    logits = np_rng.standard_normal((5, 4)).astype(np.float32)
    mask = np.array([[True, True, False, False]] * 5)
    y = np.zeros(5, dtype=int)
    _, grad = SoftmaxCrossEntropy()(logits, y, mask=mask)
    assert np.allclose(grad[:, 2:], 0.0), "masked classes must not receive gradient"


def test_dropout_only_fires_in_training(np_rng):
    d = Dropout(0.5, rng=np_rng)
    x = np.ones((100, 20), dtype=np.float32)
    assert np.array_equal(d.forward(x, training=False), x)
    out = d.forward(x, training=True)
    assert 0.2 < (out == 0).mean() < 0.8
    assert abs(out.mean() - 1.0) < 0.15, "inverted dropout keeps the scale"


@pytest.mark.parametrize("opt_cls", [Adam, SGD])
def test_optimisers_learn_xor(opt_cls):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([0, 1, 1, 0])
    net = mlp([2, 16, 16, 2], rng=np.random.default_rng(1))
    opt = opt_cls(net.parameters(), lr=0.05 if opt_cls is Adam else 0.5)
    ce = SoftmaxCrossEntropy()
    for _ in range(900):
        opt.zero_grad()
        loss, grad = ce(net.forward(X), Y)
        net.backward(grad)
        opt.step()
    assert loss < 0.05
    assert (net.forward(X, training=False).argmax(1) == Y).all()


def test_gradient_clipping_bounds_the_norm(np_rng):
    net = mlp([4, 8, 2], rng=np_rng)
    opt = Adam(net.parameters())
    for p in net.parameters():
        p.grad = np.full_like(p.value, 100.0)
    opt.clip_grad_norm(1.0)
    total = np.sqrt(sum(float((p.grad ** 2).sum()) for p in net.parameters()))
    assert total == pytest.approx(1.0, rel=1e-5)


def test_save_and_load_roundtrip(np_rng):
    def build():
        return mlp([5, 7, 3], rng=np.random.default_rng(2))

    net = build()
    x = np_rng.standard_normal((4, 5)).astype(np.float32)
    path = os.path.join(tempfile.mkdtemp(), "m.npz")
    save(path, net, {"note": "hello"})
    other = build()
    for p in other.parameters():
        p.value = np.zeros_like(p.value)
    meta = load(path, other)
    assert meta["note"] == "hello"
    assert np.allclose(net.forward(x, training=False), other.forward(x, training=False))


def test_load_rejects_a_mismatched_model(np_rng):
    path = os.path.join(tempfile.mkdtemp(), "m.npz")
    save(path, mlp([5, 7, 3], rng=np_rng))
    with pytest.raises(ValueError):
        load(path, mlp([5, 9, 3], rng=np_rng))


def test_minibatches_cover_every_index():
    seen = sorted(i for idx in minibatches(37, 8, np.random.default_rng(0)) for i in idx)
    assert seen == list(range(37))


def test_one_hot():
    assert (one_hot(np.array([0, 2]), 3) == np.array([[1, 0, 0], [0, 0, 1]])).all()

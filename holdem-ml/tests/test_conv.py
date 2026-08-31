import numpy as np
import pytest

from holdem.ml.conv import BatchNorm2D, Conv2D, Flatten, GlobalAvgPool, MaxPool2D
from holdem.ml.nn import Linear, MSELoss, Sequential, Tanh, numeric_grad_check


@pytest.fixture
def np_rng():
    return np.random.default_rng(3)


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1)])
def test_conv_backward_matches_a_naive_implementation(np_rng, stride, padding):
    conv = Conv2D(3, 4, 3, stride=stride, padding=padding, rng=np_rng)
    x = np_rng.standard_normal((4, 3, 11, 9))
    out = conv.forward(x)
    grad = np_rng.standard_normal(out.shape)
    conv.zero_grad()
    dx = conv.backward(grad)

    xp = np.pad(x, ((0, 0), (0, 0), (padding,) * 2, (padding,) * 2)) if padding else x
    dxp = np.zeros_like(xp)
    dW = np.zeros_like(conv.W.value, dtype=np.float64)
    db = np.zeros_like(conv.b.value, dtype=np.float64)
    for n in range(x.shape[0]):
        for oc in range(4):
            for y in range(out.shape[2]):
                for z in range(out.shape[3]):
                    g = grad[n, oc, y, z]
                    db[oc] += g
                    patch = xp[n, :, y * stride:y * stride + 3, z * stride:z * stride + 3]
                    dW[oc] += g * patch
                    dxp[n, :, y * stride:y * stride + 3, z * stride:z * stride + 3] += \
                        g * conv.W.value[oc]
    expected_dx = dxp[:, :, padding:padding + 11, padding:padding + 9] if padding else dxp
    assert np.allclose(dx, expected_dx, atol=1e-9)
    assert np.allclose(conv.W.grad, dW, rtol=1e-4, atol=1e-4)
    assert np.allclose(conv.b.grad, db, rtol=1e-4, atol=1e-4)


def test_conv_forward_shapes(np_rng):
    conv = Conv2D(3, 8, 3, stride=2, padding=1, rng=np_rng)
    assert conv.forward(np_rng.standard_normal((2, 3, 48, 32))).shape == (2, 8, 24, 16)


def test_maxpool_routes_gradient_to_the_maximum(np_rng):
    pool = MaxPool2D(2)
    x = np_rng.standard_normal((3, 2, 8, 6))
    out = pool.forward(x)
    grad = np_rng.standard_normal(out.shape)
    dx = pool.backward(grad)
    expected = np.zeros_like(x)
    for n in range(3):
        for c in range(2):
            for y in range(4):
                for z in range(3):
                    window = x[n, c, 2 * y:2 * y + 2, 2 * z:2 * z + 2]
                    a = np.unravel_index(np.argmax(window), window.shape)
                    expected[n, c, 2 * y + a[0], 2 * z + a[1]] += grad[n, c, y, z]
    assert np.allclose(dx, expected)


def test_batchnorm_normalises_in_training_and_uses_running_stats_after(np_rng):
    bn = BatchNorm2D(3)
    x = np_rng.standard_normal((16, 3, 5, 5)) * 4 + 7
    out = bn.forward(x, training=True)
    assert np.allclose(out.mean(axis=(0, 2, 3)), 0.0, atol=1e-5)
    assert np.allclose(out.std(axis=(0, 2, 3)), 1.0, atol=1e-3)
    assert not np.allclose(bn.running_mean, 0.0), "running stats must track the data"
    eval_out = bn.forward(x, training=False)
    assert not np.allclose(eval_out, out), "eval mode uses running stats, not batch stats"


def test_conv_stack_gradients(np_rng):
    model = Sequential(
        Conv2D(2, 4, 3, padding=1, rng=np_rng), BatchNorm2D(4), Tanh(), MaxPool2D(2),
        Conv2D(4, 5, 3, stride=2, padding=1, rng=np_rng), Tanh(),
        GlobalAvgPool(), Linear(5, 3, rng=np_rng),
    )
    x = np_rng.standard_normal((3, 2, 12, 12)).astype(np.float32)
    y = np_rng.standard_normal((3, 3)).astype(np.float32)
    assert numeric_grad_check(model, MSELoss(), x, y, samples=50) < 1e-6


def test_flatten_roundtrip(np_rng):
    f = Flatten()
    x = np_rng.standard_normal((4, 3, 5, 5))
    out = f.forward(x)
    assert out.shape == (4, 75)
    assert f.backward(out).shape == x.shape

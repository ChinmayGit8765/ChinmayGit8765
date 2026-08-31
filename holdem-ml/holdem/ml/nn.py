"""A small neural-network framework written from scratch on NumPy.

No PyTorch, no TensorFlow, no autograd package — every layer implements its own
forward and backward pass explicitly.  That is a deliberate choice: the whole
point of this project is a *custom* model, and hand-written backprop keeps the
poker-specific pieces (regret matching, self-play updates, the two-headed card
CNN) transparent and dependency-free.

Conventions
-----------
* Tensors are ``float32`` arrays, batch-major: ``(batch, features)``.
* ``Module.forward(x, training=True)`` caches what the backward pass needs.
* ``Module.backward(grad)`` returns the gradient w.r.t. the module's input and
  accumulates parameter gradients into ``Parameter.grad``.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class Parameter:
    __slots__ = ("value", "grad", "name")

    def __init__(self, value: np.ndarray, name: str = ""):
        self.value = value.astype(np.float32)
        self.grad = np.zeros_like(self.value)
        self.name = name

    @property
    def shape(self):
        return self.value.shape

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class Module:
    """Base class: a differentiable block."""

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        return []

    def modules(self) -> List["Module"]:
        return [self]

    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        return self.forward(x, training=training)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()


# --- core layers ------------------------------------------------------------

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init: str = "he", rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        if init == "he":
            scale = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            scale = np.sqrt(1.0 / in_features)
        elif init == "zero":
            scale = 0.0
        else:
            raise ValueError(f"unknown init {init!r}")
        self.W = Parameter(rng.standard_normal((in_features, out_features)) * scale, "W")
        self.b = Parameter(np.zeros(out_features), "b") if bias else None
        self._x: Optional[np.ndarray] = None

    def forward(self, x, training=True):
        self._x = x
        out = x @ self.W.value
        if self.b is not None:
            out = out + self.b.value
        return out

    def backward(self, grad):
        self.W.grad += self._x.T @ grad
        if self.b is not None:
            self.b.grad += grad.sum(axis=0)
        return grad @ self.W.value.T

    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])


class ReLU(Module):
    def forward(self, x, training=True):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad):
        return grad * self._mask


class LeakyReLU(Module):
    def __init__(self, slope: float = 0.01):
        self.slope = slope

    def forward(self, x, training=True):
        self._mask = np.where(x > 0, 1.0, self.slope).astype(np.float32)
        return x * self._mask

    def backward(self, grad):
        return grad * self._mask


class Tanh(Module):
    def forward(self, x, training=True):
        self._y = np.tanh(x)
        return self._y

    def backward(self, grad):
        return grad * (1.0 - self._y ** 2)


class Sigmoid(Module):
    def forward(self, x, training=True):
        self._y = 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
        return self._y

    def backward(self, grad):
        return grad * self._y * (1.0 - self._y)


class Dropout(Module):
    def __init__(self, p: float = 0.1, rng: Optional[np.random.Generator] = None):
        self.p = p
        self.rng = rng or np.random.default_rng()

    def forward(self, x, training=True):
        if not training or self.p <= 0:
            self._mask = None
            return x
        self._mask = (self.rng.random(x.shape) >= self.p).astype(np.float32) / (1 - self.p)
        return x * self._mask

    def backward(self, grad):
        return grad if self._mask is None else grad * self._mask


class LayerNorm(Module):
    def __init__(self, features: int, eps: float = 1e-5):
        self.g = Parameter(np.ones(features), "gamma")
        self.b = Parameter(np.zeros(features), "beta")
        self.eps = eps

    def forward(self, x, training=True):
        self._mu = x.mean(axis=-1, keepdims=True)
        self._var = x.var(axis=-1, keepdims=True)
        self._inv = 1.0 / np.sqrt(self._var + self.eps)
        self._xhat = (x - self._mu) * self._inv
        return self._xhat * self.g.value + self.b.value

    def backward(self, grad):
        n = grad.shape[-1]
        self.g.grad += (grad * self._xhat).sum(axis=0)
        self.b.grad += grad.sum(axis=0)
        dxhat = grad * self.g.value
        return self._inv / n * (
            n * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - self._xhat * (dxhat * self._xhat).sum(axis=-1, keepdims=True)
        )

    def parameters(self):
        return [self.g, self.b]


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def modules(self):
        out = [self]
        for layer in self.layers:
            out.extend(layer.modules())
        return out

    def append(self, layer: Module) -> None:
        self.layers.append(layer)


def mlp(sizes: Sequence[int], activation=ReLU, out_activation: Optional[type] = None,
        dropout: float = 0.0, layernorm: bool = False,
        rng: Optional[np.random.Generator] = None) -> Sequential:
    """Build a plain feed-forward stack, e.g. ``mlp([64, 128, 128, 5])``."""
    layers: List[Module] = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], rng=rng))
        last = i == len(sizes) - 2
        if not last:
            if layernorm:
                layers.append(LayerNorm(sizes[i + 1]))
            layers.append(activation())
            if dropout:
                layers.append(Dropout(dropout, rng=rng))
        elif out_activation is not None:
            layers.append(out_activation())
    return Sequential(*layers)


# --- losses -----------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


class SoftmaxCrossEntropy:
    """Cross-entropy over logits; targets are class indices or a probability matrix."""

    def __call__(self, logits: np.ndarray, targets: np.ndarray,
                 weights: Optional[np.ndarray] = None,
                 mask: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        if mask is not None:
            logits = np.where(mask, logits, -1e9)
        logp = log_softmax(logits)
        n = logits.shape[0]
        if targets.ndim == 1:
            onehot = np.zeros_like(logp)
            onehot[np.arange(n), targets.astype(int)] = 1.0
        else:
            onehot = targets
        per_sample = -(onehot * logp).sum(axis=1)
        if weights is not None:
            per_sample = per_sample * weights
        loss = float(per_sample.mean())
        grad = (softmax(logits) - onehot)
        if weights is not None:
            grad = grad * weights[:, None]
        if mask is not None:
            grad = np.where(mask, grad, 0.0)
        return loss, (grad / n).astype(logits.dtype, copy=False)


class MSELoss:
    def __call__(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        diff = pred - target
        loss = float(np.mean(diff ** 2))
        return loss, (2.0 * diff / diff.size).astype(pred.dtype, copy=False)


class HuberLoss:
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        diff = pred - target
        absd = np.abs(diff)
        quad = absd <= self.delta
        loss = float(np.mean(np.where(quad, 0.5 * diff ** 2, self.delta * (absd - 0.5 * self.delta))))
        grad = np.where(quad, diff, self.delta * np.sign(diff)) / diff.size
        return loss, grad.astype(pred.dtype, copy=False)


# --- optimisers -------------------------------------------------------------

class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float):
        self.params = list(params)
        self.lr = lr
        self.steps = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def clip_grad_norm(self, max_norm: float) -> float:
        total = np.sqrt(sum(float((p.grad ** 2).sum()) for p in self.params))
        if total > max_norm and total > 0:
            scale = max_norm / total
            for p in self.params:
                p.grad *= scale
        return total


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = [np.zeros_like(p.value) for p in self.params]

    def step(self):
        self.steps += 1
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.value
            self.v[i] = self.momentum * self.v[i] + g
            p.value -= self.lr * self.v[i]


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 decoupled: bool = True):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled = decoupled  # AdamW-style when True
        self.m = [np.zeros_like(p.value) for p in self.params]
        self.v = [np.zeros_like(p.value) for p in self.params]

    def step(self):
        self.steps += 1
        t = self.steps
        bc1 = 1 - self.b1 ** t
        bc2 = 1 - self.b2 ** t
        for i, p in enumerate(self.params):
            g = p.grad
            if self.weight_decay and not self.decoupled:
                g = g + self.weight_decay * p.value
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / bc1
            vhat = self.v[i] / bc2
            if self.weight_decay and self.decoupled:
                p.value -= self.lr * self.weight_decay * p.value
            p.value -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# --- persistence ------------------------------------------------------------

def state_dict(model: Module) -> Dict[str, np.ndarray]:
    return {f"p{i}": p.value for i, p in enumerate(model.parameters())}


def load_state_dict(model: Module, state: Dict[str, np.ndarray]) -> None:
    params = model.parameters()
    if len(params) != len(state):
        raise ValueError(f"checkpoint has {len(state)} tensors, model wants {len(params)}")
    for i, p in enumerate(params):
        arr = state[f"p{i}"]
        if arr.shape != p.value.shape:
            raise ValueError(f"shape mismatch for p{i}: {arr.shape} vs {p.value.shape}")
        p.value = arr.astype(np.float32)


def buffer_dict(model: Module) -> Dict[str, np.ndarray]:
    """Non-learned state (e.g. batch-norm running stats) that must survive a save."""
    out: Dict[str, np.ndarray] = {}
    for i, m in enumerate(model.modules()):
        getter = getattr(m, "buffers", None)
        if getter:
            for key, value in getter().items():
                out[f"__buf__{i}__{key}"] = np.asarray(value)
    return out


def load_buffers(model: Module, state: Dict[str, np.ndarray]) -> None:
    mods = model.modules()
    for key, value in state.items():
        rest = key[len("__buf__"):]
        idx_str, name = rest.split("__", 1)
        mod = mods[int(idx_str)]
        setattr(mod, name, value)


def save(path: str, model: Module, meta: Optional[dict] = None) -> None:
    payload = state_dict(model)
    payload.update(buffer_dict(model))
    payload["__meta__"] = np.frombuffer(
        json.dumps(meta or {}).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(path, **payload)


def load(path: str, model: Module) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = {}
        state = {}
        buffers = {}
        for k in data.files:
            if k == "__meta__":
                meta = json.loads(bytes(data[k]).decode("utf-8"))
            elif k.startswith("__buf__"):
                buffers[k] = data[k]
            else:
                state[k] = data[k]
    load_state_dict(model, state)
    load_buffers(model, buffers)
    return meta


def read_meta(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if "__meta__" in data.files:
            return json.loads(bytes(data["__meta__"]).decode("utf-8"))
    return {}


# --- helpers ----------------------------------------------------------------

def minibatches(n: int, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(idx), n), dtype=np.float32)
    out[np.arange(len(idx)), np.asarray(idx, dtype=int)] = 1.0
    return out


def numeric_grad_check(model: Module, loss_fn, x: np.ndarray, y: np.ndarray,
                       eps: float = 1e-4, samples: int = 40,
                       rng: Optional[np.random.Generator] = None) -> float:
    """Max relative error between analytic and finite-difference gradients.

    Used by the test-suite to prove the hand-written backward passes are right.
    """
    rng = rng or np.random.default_rng(0)
    # Finite differences in float32 are dominated by rounding noise, so the
    # check runs the whole model in float64 and restores precision afterwards.
    params = model.parameters()
    saved = [p.value for p in params]
    for p in params:
        p.value = p.value.astype(np.float64)
        p.grad = p.grad.astype(np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.dtype.kind == "f":
        y = np.asarray(y, dtype=np.float64)
    try:
        model.zero_grad()
        out = model.forward(x, training=False)
        _, grad = loss_fn(out, y)
        model.backward(grad)
        worst = 0.0
        for _ in range(samples):
            p = params[rng.integers(len(params))]
            idx = tuple(rng.integers(s) for s in p.value.shape)
            original = float(p.value[idx])
            p.value[idx] = original + eps
            loss_plus, _ = loss_fn(model.forward(x, training=False), y)
            p.value[idx] = original - eps
            loss_minus, _ = loss_fn(model.forward(x, training=False), y)
            p.value[idx] = original
            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = float(p.grad[idx])
            denom = max(1e-9, abs(numeric) + abs(analytic))
            worst = max(worst, abs(numeric - analytic) / denom)
    finally:
        for p, value in zip(params, saved):
            p.value = value
            p.grad = np.zeros_like(value)
    return worst

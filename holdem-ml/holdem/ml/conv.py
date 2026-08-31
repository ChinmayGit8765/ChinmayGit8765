"""Convolutional layers for the card-vision CNN — also written from scratch.

Images are ``(batch, channels, height, width)`` float32.  Convolution uses the
classic im2col trick so the heavy lifting is one matrix multiply per layer,
which keeps pure-NumPy training practical.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .nn import Module, Parameter


def _im2col_indices(shape: Tuple[int, int, int, int], kh: int, kw: int,
                    stride: int, pad: int):
    _, c, h, w = shape
    out_h = (h + 2 * pad - kh) // stride + 1
    out_w = (w + 2 * pad - kw) // stride + 1
    i0 = np.tile(np.repeat(np.arange(kh), kw), c)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(kw), kh * c)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), kh * kw).reshape(-1, 1)
    return k, i, j, out_h, out_w


class Conv2D(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3,
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        fan_in = in_channels * kernel * kernel
        scale = np.sqrt(2.0 / fan_in)
        self.W = Parameter(
            rng.standard_normal((out_channels, in_channels, kernel, kernel)) * scale, "convW"
        )
        self.b = Parameter(np.zeros(out_channels), "convb") if bias else None
        self.stride, self.padding, self.kernel = stride, padding, kernel
        self.in_channels, self.out_channels = in_channels, out_channels
        self._scatter = None

    def forward(self, x, training=True):
        self._shape = x.shape
        n = x.shape[0]
        p = self.padding
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x
        k, i, j, oh, ow = _im2col_indices(x.shape, self.kernel, self.kernel,
                                          self.stride, self.padding)
        self._idx = (k, i, j)
        self._oh, self._ow = oh, ow
        col = xp[:, k, i, j]                      # (n, C*kh*kw, oh*ow)
        self._padded_shape = xp.shape
        wf = self.W.value.reshape(self.out_channels, -1)
        # One GEMM over the whole batch: (oc, D) @ (D, n*L).  The flattened
        # layout is cached because backward needs exactly the same matrix.
        d, l = col.shape[1], col.shape[2]
        self._d, self._l = d, l
        flat = np.ascontiguousarray(col.transpose(1, 0, 2).reshape(d, n * l))
        self._col_flat = flat
        out = (wf @ flat).reshape(self.out_channels, n, l).transpose(1, 0, 2)
        if self.b is not None:
            out = out + self.b.value[None, :, None]
        return np.ascontiguousarray(out.reshape(n, self.out_channels, oh, ow)).astype(
            x.dtype, copy=False)

    def backward(self, grad):
        n, oc, oh, ow = grad.shape
        d, l = self._d, self._l
        g = grad.reshape(n, oc, oh * ow)
        wf = self.W.value.reshape(oc, -1)

        # dW: (oc, n*L) @ (n*L, D)
        g_flat = np.ascontiguousarray(g.transpose(1, 0, 2).reshape(oc, n * l))
        self.W.grad += (g_flat @ self._col_flat.T).reshape(self.W.shape)
        if self.b is not None:
            self.b.grad += g.sum(axis=(0, 2))

        dcol = (wf.T @ g_flat).reshape(d, n, l).transpose(1, 0, 2)
        dcol = np.ascontiguousarray(dcol).reshape(n, d * l)

        # col2im.  np.add.at is unbuffered and painfully slow, so the scatter
        # is done once as a sort + reduceat over precomputed indices, which
        # accumulates the whole batch in two vectorised passes.
        _, c, h, w = self._shape
        p = self.padding
        k, i, j = self._idx
        _, _, hp, wp = self._padded_shape
        size = c * hp * wp
        if self._scatter is None or self._scatter[0].size != d * l:
            flat_idx = ((k * hp + i) * wp + j).ravel()
            order = np.argsort(flat_idx, kind="stable")
            sorted_idx = flat_idx[order]
            starts = np.flatnonzero(
                np.concatenate(([True], sorted_idx[1:] != sorted_idx[:-1]))
            )
            self._scatter = (order, starts, sorted_idx[starts])
        order, starts, uniq = self._scatter
        dx = np.zeros((n, size), dtype=dcol.dtype)
        dx[:, uniq] = np.add.reduceat(dcol[:, order], starts, axis=1)
        dx = dx.reshape(n, c, hp, wp)
        if p:
            dx = dx[:, :, p:p + h, p:p + w]
        return dx.astype(grad.dtype, copy=False)

    def parameters(self) -> List[Parameter]:
        return [self.W] + ([self.b] if self.b is not None else [])


class MaxPool2D(Module):
    def __init__(self, size: int = 2, stride: Optional[int] = None):
        self.size = size
        self.stride = stride or size

    def forward(self, x, training=True):
        n, c, h, w = x.shape
        s, k = self.stride, self.size
        oh, ow = (h - k) // s + 1, (w - k) // s + 1
        # (n, c, oh, ow, k, k) view of every pooling window
        strides = x.strides
        windows = np.lib.stride_tricks.as_strided(
            x,
            shape=(n, c, oh, ow, k, k),
            strides=(strides[0], strides[1], strides[2] * s, strides[3] * s,
                     strides[2], strides[3]),
            writeable=False,
        )
        flat = windows.reshape(n, c, oh, ow, k * k)
        self._arg = flat.argmax(axis=-1)
        self._shape = x.shape
        self._oh, self._ow = oh, ow
        return flat.max(axis=-1)

    def backward(self, grad):
        n, c, h, w = self._shape
        s, k = self.stride, self.size
        oh, ow = self._oh, self._ow
        ii, jj = np.divmod(self._arg, k)
        hi = np.arange(oh)[None, None, :, None] * s + ii
        wi = np.arange(ow)[None, None, None, :] * s + jj
        base = (np.arange(n)[:, None, None, None] * c + np.arange(c)[None, :, None, None]) * (h * w)
        flat = (base + hi * w + wi).ravel()
        dx = np.bincount(flat, weights=grad.ravel(), minlength=n * c * h * w)
        return dx.reshape(self._shape).astype(grad.dtype, copy=False)

    def parameters(self):
        return []


class Flatten(Module):
    def forward(self, x, training=True):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self._shape)


class BatchNorm2D(Module):
    """Per-channel batch norm with running statistics for inference."""

    def __init__(self, channels: int, momentum: float = 0.9, eps: float = 1e-5):
        self.g = Parameter(np.ones(channels), "bn_gamma")
        self.b = Parameter(np.zeros(channels), "bn_beta")
        self.running_mean = np.zeros(channels, dtype=np.float32)
        self.running_var = np.ones(channels, dtype=np.float32)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, training=True):
        if training:
            mu = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))
            self.running_mean = (self.momentum * self.running_mean
                                 + (1 - self.momentum) * mu).astype(np.float32)
            self.running_var = (self.momentum * self.running_var
                                + (1 - self.momentum) * var).astype(np.float32)
        else:
            mu, var = self.running_mean, self.running_var
        self._inv = 1.0 / np.sqrt(var + self.eps)
        self._xhat = (x - mu[None, :, None, None]) * self._inv[None, :, None, None]
        self._training = training
        return self._xhat * self.g.value[None, :, None, None] + self.b.value[None, :, None, None]

    def backward(self, grad):
        n = grad.shape[0] * grad.shape[2] * grad.shape[3]
        self.g.grad += (grad * self._xhat).sum(axis=(0, 2, 3))
        self.b.grad += grad.sum(axis=(0, 2, 3))
        dxhat = grad * self.g.value[None, :, None, None]
        if not self._training:
            return dxhat * self._inv[None, :, None, None]
        return (self._inv[None, :, None, None] / n) * (
            n * dxhat
            - dxhat.sum(axis=(0, 2, 3))[None, :, None, None]
            - self._xhat * (dxhat * self._xhat).sum(axis=(0, 2, 3))[None, :, None, None]
        )

    def parameters(self):
        return [self.g, self.b]

    def buffers(self):
        return {"running_mean": self.running_mean, "running_var": self.running_var}


class GlobalAvgPool(Module):
    def forward(self, x, training=True):
        self._shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, grad):
        n, c, h, w = self._shape
        return np.broadcast_to(grad[:, :, None, None] / (h * w), self._shape).copy()

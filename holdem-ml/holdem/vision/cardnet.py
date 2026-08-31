"""The card-recognition CNN: one trunk, two heads (rank and suit).

Rank and suit are separate 13-way and 4-way problems that share almost all of
their visual evidence, so a shared trunk with two heads trains faster and
generalises better than 52 independent classes — and a wrong suit no longer
costs you the rank.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..cards import make_card
from ..ml import nn as _nn
from ..ml.conv import BatchNorm2D, Conv2D, Flatten, MaxPool2D
from ..ml.nn import Dropout, Linear, Module, Parameter, ReLU, Sequential, softmax
from .dataset import IMG_H, IMG_W, NUM_RANKS, NUM_SUITS

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"
)
DEFAULT_CARDNET = os.path.join(MODEL_DIR, "cardnet.npz")


class CardNet(Module):
    def __init__(self, channels: Sequence[int] = (16, 32, 64), hidden: int = 160,
                 dropout: float = 0.15, rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng(0)
        c1, c2, c3 = channels
        self.channels = list(channels)
        self.hidden = hidden
        # Downsampling is done with strided convolutions rather than pooling:
        # same receptive field for roughly a third of the arithmetic, which
        # matters when every matrix multiply is pure NumPy.
        self.trunk = Sequential(
            Conv2D(3, c1, 3, stride=2, padding=1, rng=rng),   # 48x32 -> 24x16
            BatchNorm2D(c1), ReLU(),
            Conv2D(c1, c2, 3, stride=2, padding=1, rng=rng),  # -> 12x8
            BatchNorm2D(c2), ReLU(),
            Conv2D(c2, c3, 3, stride=2, padding=1, rng=rng),  # -> 6x4
            BatchNorm2D(c3), ReLU(),
            Flatten(),
            Linear(c3 * (IMG_H // 8) * (IMG_W // 8), hidden, rng=rng),
            ReLU(), Dropout(dropout, rng=rng),
        )
        self.rank_head = Linear(hidden, NUM_RANKS, rng=rng)
        self.suit_head = Linear(hidden, NUM_SUITS, rng=rng)

    def forward(self, x: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        h = self.trunk.forward(x, training=training)
        return self.rank_head.forward(h, training), self.suit_head.forward(h, training)

    def backward(self, grad_rank: np.ndarray, grad_suit: np.ndarray) -> np.ndarray:
        dh = self.rank_head.backward(grad_rank) + self.suit_head.backward(grad_suit)
        return self.trunk.backward(dh)

    def parameters(self) -> List[Parameter]:
        return (self.trunk.parameters() + self.rank_head.parameters()
                + self.suit_head.parameters())

    def modules(self) -> List[Module]:
        return [self] + self.trunk.modules() + [self.rank_head, self.suit_head]

    # -- inference -----------------------------------------------------------

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(cards, confidence)`` for a batch of ``(n, 3, H, W)`` images."""
        rank_logits, suit_logits = self.forward(x, training=False)
        rp, sp = softmax(rank_logits), softmax(suit_logits)
        ranks, suits = rp.argmax(1), sp.argmax(1)
        cards = np.array([make_card(int(r), int(s)) for r, s in zip(ranks, suits)])
        confidence = rp.max(1) * sp.max(1)
        return cards, confidence

    @property
    def config(self) -> dict:
        return {"channels": self.channels, "hidden": self.hidden,
                "img_h": IMG_H, "img_w": IMG_W}


def save_cardnet(path: str, net: CardNet, meta: Optional[dict] = None) -> None:
    payload = dict(meta or {})
    payload.update(net.config)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _nn.save(path, net, payload)


def load_cardnet(path: str = DEFAULT_CARDNET) -> CardNet:
    meta = _nn.read_meta(path)
    net = CardNet(channels=tuple(meta.get("channels", (16, 32, 64))),
                  hidden=int(meta.get("hidden", 160)))
    _nn.load(path, net)
    return net

"""The bot's neural brain: a shared trunk with a policy head and a value head.

* **Policy head** — a distribution over the 7 abstract actions.
* **Value head** — the expected result of the hand from here, in big blinds.
  It is the baseline that makes the self-play policy-gradient updates
  low-variance, and it doubles as the "how good is this spot" number the
  analyser reports.

Training happens in two stages (see :mod:`holdem.train.selfplay`): supervised
distillation of the CFR blueprint, then policy-gradient self-play.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .abstraction import NUM_ACTIONS
from .features import NUM_FEATURES
from .nn import Linear, Module, Parameter, ReLU, mlp


class PolicyValueNet(Module):
    def __init__(self, in_features: int = NUM_FEATURES,
                 hidden: Sequence[int] = (192, 160, 96),
                 rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng(0)
        self.in_features = in_features
        self.hidden = list(hidden)
        self.trunk = mlp([in_features, *hidden], activation=ReLU,
                         out_activation=ReLU, layernorm=True, rng=rng)
        self.policy_head = Linear(hidden[-1], NUM_ACTIONS, rng=rng)
        self.value_head = Linear(hidden[-1], 1, rng=rng)

    def forward(self, x: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        h = self.trunk.forward(x, training=training)
        return self.policy_head.forward(h, training), self.value_head.forward(h, training)

    def backward(self, grad_policy: np.ndarray, grad_value: Optional[np.ndarray] = None):
        dh = self.policy_head.backward(grad_policy)
        if grad_value is not None:
            dh = dh + self.value_head.backward(grad_value)
        return self.trunk.backward(dh)

    def parameters(self) -> List[Parameter]:
        return (self.trunk.parameters() + self.policy_head.parameters()
                + self.value_head.parameters())

    def modules(self) -> List[Module]:
        return [self] + self.trunk.modules() + [self.policy_head, self.value_head]

    # -- inference -----------------------------------------------------------

    def action_probs(self, features: np.ndarray, mask: np.ndarray,
                     temperature: float = 1.0) -> np.ndarray:
        """Masked, temperature-scaled policy for a single state."""
        logits, _ = self.forward(features[None, :].astype(np.float32), training=False)
        return mask_softmax(logits[0], mask, temperature)

    def evaluate_state(self, features: np.ndarray) -> float:
        _, value = self.forward(features[None, :].astype(np.float32), training=False)
        return float(value[0, 0])

    @property
    def config(self) -> dict:
        return {"in_features": self.in_features, "hidden": self.hidden,
                "num_actions": NUM_ACTIONS}


def mask_softmax(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax restricted to legal actions.  ``temperature`` > 1 flattens."""
    t = max(1e-3, temperature)
    z = np.where(mask, logits / t, -1e9)
    z = z - z.max()
    e = np.exp(z) * mask
    total = e.sum()
    if total <= 0:  # pragma: no cover - all-masked shouldn't happen
        out = mask.astype(np.float64)
        return out / out.sum()
    return e / total


def save_policy(path: str, net: PolicyValueNet, meta: Optional[dict] = None) -> None:
    from . import nn as _nn
    payload = dict(meta or {})
    payload.update(net.config)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _nn.save(path, net, payload)


def load_policy(path: str) -> PolicyValueNet:
    from . import nn as _nn
    meta = _nn.read_meta(path)
    net = PolicyValueNet(
        in_features=int(meta.get("in_features", NUM_FEATURES)),
        hidden=tuple(meta.get("hidden", (192, 160, 96))),
    )
    _nn.load(path, net)
    return net

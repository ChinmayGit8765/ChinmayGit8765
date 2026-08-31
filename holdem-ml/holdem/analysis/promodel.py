"""The analyser's model: three heads over one shared understanding of a spot.

* **policy** — what a strong player does here (7-way distribution).
* **action values** — what each action is worth, in big blinds.  Regressed onto
  the realised result of the action that was actually taken, so it learns the
  empirical value of a line rather than a theoretical one.
* **value** — the worth of the spot itself, used as the baseline that turns a
  raw result into "how much did this decision cost".

``ev_loss = max_a Q(a) - Q(chosen)`` is the number the analyser reports as the
cost of a mistake.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..ml import nn as _nn
from ..ml.abstraction import NUM_ACTIONS
from ..ml.features import NUM_FEATURES
from ..ml.nn import Linear, Module, Parameter, ReLU, mlp, softmax
from ..ml.policy import mask_softmax

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"
)
DEFAULT_PRO_MODEL = os.path.join(MODEL_DIR, "promodel.npz")


class ProModel(Module):
    def __init__(self, in_features: int = NUM_FEATURES,
                 hidden: Sequence[int] = (224, 160, 96),
                 rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng(0)
        self.in_features = in_features
        self.hidden = list(hidden)
        self.trunk = mlp([in_features, *hidden], activation=ReLU, out_activation=ReLU,
                         layernorm=True, rng=rng)
        self.policy_head = Linear(hidden[-1], NUM_ACTIONS, rng=rng)
        self.q_head = Linear(hidden[-1], NUM_ACTIONS, rng=rng)
        self.value_head = Linear(hidden[-1], 1, rng=rng)

    def forward(self, x, training: bool = True):
        h = self.trunk.forward(x, training=training)
        return (self.policy_head.forward(h, training),
                self.q_head.forward(h, training),
                self.value_head.forward(h, training))

    def backward(self, grad_policy, grad_q, grad_value):
        dh = self.policy_head.backward(grad_policy)
        dh = dh + self.q_head.backward(grad_q)
        dh = dh + self.value_head.backward(grad_value)
        return self.trunk.backward(dh)

    def parameters(self) -> List[Parameter]:
        return (self.trunk.parameters() + self.policy_head.parameters()
                + self.q_head.parameters() + self.value_head.parameters())

    def modules(self):
        return [self] + self.trunk.modules() + [self.policy_head, self.q_head,
                                                self.value_head]

    # -- inference -----------------------------------------------------------

    def assess(self, features: np.ndarray, mask: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, float]:
        """``(pro policy, action values in bb, spot value in bb)`` for one state."""
        logits, q, v = self.forward(features[None, :].astype(np.float32), training=False)
        probs = mask_softmax(logits[0], mask, 1.0)
        values = np.where(mask, q[0], -np.inf)
        return probs, values, float(v[0, 0])

    @property
    def config(self) -> dict:
        return {"in_features": self.in_features, "hidden": self.hidden}


def save_pro_model(path: str, model: ProModel, meta: Optional[dict] = None) -> None:
    payload = dict(meta or {})
    payload.update(model.config)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _nn.save(path, model, payload)


def load_pro_model(path: str = DEFAULT_PRO_MODEL) -> ProModel:
    meta = _nn.read_meta(path)
    model = ProModel(in_features=int(meta.get("in_features", NUM_FEATURES)),
                     hidden=tuple(meta.get("hidden", (224, 160, 96))))
    _nn.load(path, model)
    return model

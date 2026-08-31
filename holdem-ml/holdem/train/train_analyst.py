"""Train the pro-game analyser.

    # from strong self-play
    python -m holdem.train.train_analyst --hands 6000

    # from real hand histories you already have
    python -m holdem.train.train_analyst --histories ~/pokerstars/HandHistory --hands 0

Corpus sources can be combined.  Every hand — bot or human, generated or
imported — becomes ``(state features, action taken, big blinds won)`` triples,
and the model learns a pro policy, per-action values and a spot value from them.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Optional, Sequence

import numpy as np

from ..analysis.corpus import (
    Sample,
    load_corpus,
    samples_from_histories,
    samples_from_results,
    save_corpus,
    stack_samples,
)
from ..analysis.promodel import DEFAULT_PRO_MODEL, ProModel, save_pro_model
from ..bots.blueprint import BlueprintBot
from ..bots.neural import NeuralBot
from ..bots.rule import EquityBot, HonestBot, LooseAggressive, TightRock
from ..game import Game
from ..ml.nn import Adam, HuberLoss, SoftmaxCrossEntropy, minibatches


def generate_corpus(hands: int, rng: random.Random, seats: int = 6,
                    equity_iters: int = 200) -> List[Sample]:
    """Play strong bots against each other and harvest every decision."""
    bots = [
        BlueprintBot("Blueprint", rng=rng),
        NeuralBot("Neural-pro", difficulty="pro", rng=rng),
        EquityBot("Equity", rng),
        HonestBot("Honest", rng),
        TightRock("Rock", rng),
        LooseAggressive("Lag", rng),
    ][:seats]
    game = Game(bots, rng=rng)
    samples: List[Sample] = []
    start = time.time()
    for h in range(hands):
        result = game.play_hand(keep_history=False)
        samples.extend(samples_from_results([result], game.sb, game.bb,
                                            equity_iters=equity_iters))
        if (h + 1) % 500 == 0:
            print(f"  {h + 1}/{hands} hands, {len(samples)} samples "
                  f"({time.time() - start:.0f}s)", flush=True)
    return samples


def train(model: ProModel, X, M, A, R, epochs: int = 25, lr: float = 1e-3,
          batch_size: int = 256, policy_weight: float = 1.0,
          q_weight: float = 0.6, value_weight: float = 0.3,
          clip_bb: float = 40.0, rng: Optional[np.random.Generator] = None) -> None:
    rng = rng or np.random.default_rng(0)
    R = np.clip(R, -clip_bb, clip_bb).astype(np.float32)
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ce = SoftmaxCrossEntropy()
    huber = HuberLoss(delta=4.0)
    n = len(X)

    for epoch in range(epochs):
        pol_total = q_total = val_total = 0.0
        batches = 0
        for idx in minibatches(n, batch_size, rng):
            xb, mb, ab, rb = X[idx], M[idx], A[idx], R[idx]
            opt.zero_grad()
            logits, q, v = model.forward(xb, training=True)

            pol_loss, pol_grad = ce(logits, ab, mask=mb)

            # The action value head is only supervised on the action that was
            # actually played — that is the only return we observed.
            taken = q[np.arange(len(idx)), ab]
            q_loss, q_grad_taken = huber(taken, rb)
            q_grad = np.zeros_like(q)
            q_grad[np.arange(len(idx)), ab] = q_grad_taken

            val_loss, val_grad = huber(v[:, 0], rb)
            val_grad = val_grad[:, None]

            model.backward((policy_weight * pol_grad).astype(np.float32),
                           (q_weight * q_grad).astype(np.float32),
                           (value_weight * val_grad).astype(np.float32))
            opt.clip_grad_norm(5.0)
            opt.step()
            pol_total += pol_loss
            q_total += q_loss
            val_total += val_loss
            batches += 1
        print(f"epoch {epoch + 1}/{epochs}  policy CE {pol_total / batches:.4f}  "
              f"Q huber {q_total / batches:.3f}  value huber {val_total / batches:.3f}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=6000, help="self-play hands to generate")
    ap.add_argument("--histories", default=None, help="directory of real hand histories")
    ap.add_argument("--corpus", default="data/corpus.npz", help="where to cache the corpus")
    ap.add_argument("--reuse-corpus", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=DEFAULT_PRO_MODEL)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.reuse_corpus and os.path.exists(args.corpus):
        X, M, A, R, street = load_corpus(args.corpus)
        print(f"loaded {len(X)} samples from {args.corpus}")
    else:
        samples: List[Sample] = []
        if args.histories:
            print(f"importing hand histories from {args.histories}")
            imported = samples_from_histories(args.histories)
            print(f"  {len(imported)} samples from real hands")
            samples.extend(imported)
        if args.hands:
            print(f"generating {args.hands} self-play hands with the strongest bots")
            samples.extend(generate_corpus(args.hands, rng))
        if not samples:
            raise SystemExit("no corpus: pass --hands and/or --histories")
        save_corpus(args.corpus, samples)
        X, M, A, R = stack_samples(samples)
        print(f"corpus: {len(X)} decisions saved to {args.corpus}")

    model = ProModel(rng=np.random.default_rng(args.seed))
    train(model, X, M, A, R, epochs=args.epochs, lr=args.lr,
          rng=np.random.default_rng(args.seed))
    save_pro_model(args.out, model, {"samples": int(len(X)), "epochs": args.epochs})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

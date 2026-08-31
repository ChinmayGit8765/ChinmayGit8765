"""Train the card-recognition CNN on synthetic, heavily augmented cards.

    python -m holdem.train.train_vision --train 30000 --epochs 14

Validation uses deck styles that never appear in training, so the reported
accuracy is generalisation to an unseen deck, not memorisation of the renderer.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from typing import Sequence, Tuple

import numpy as np

from ..ml.nn import Adam, SoftmaxCrossEntropy, minibatches
from ..vision.cardnet import CardNet, DEFAULT_CARDNET, save_cardnet
from ..vision.dataset import make_batch
from ..vision.render import HOLDOUT_STYLES, TRAIN_STYLES


def _gen(payload) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed, count, holdout, strength = payload
    styles = HOLDOUT_STYLES if holdout else TRAIN_STYLES
    return make_batch(count, styles, random.Random(seed), strength=strength)


def generate(count: int, holdout: bool, seed: int, workers: int,
             strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if workers <= 1:
        return _gen((seed, count, holdout, strength))
    chunk = (count + workers - 1) // workers
    payloads = [(seed + i, min(chunk, count - i * chunk), holdout, strength)
                for i in range(workers)]
    payloads = [p for p in payloads if p[1] > 0]
    ctx = mp.get_context("fork")
    with ctx.Pool(len(payloads)) as pool:
        parts = pool.map(_gen, payloads)
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def accuracy(net: CardNet, X: np.ndarray, yr: np.ndarray, ys: np.ndarray,
             batch: int = 256) -> Tuple[float, float, float]:
    rank_ok = suit_ok = card_ok = 0
    for start in range(0, len(X), batch):
        xb = X[start:start + batch]
        rl, sl = net.forward(xb, training=False)
        rp, sp = rl.argmax(1), sl.argmax(1)
        r_hit = rp == yr[start:start + batch]
        s_hit = sp == ys[start:start + batch]
        rank_ok += int(r_hit.sum())
        suit_ok += int(s_hit.sum())
        card_ok += int((r_hit & s_hit).sum())
    n = len(X)
    return rank_ok / n, suit_ok / n, card_ok / n


def clean_holdout(rng: random.Random):
    """Un-augmented renders of all 52 cards in every held-out deck style.

    The augmented score is the robustness number; this is the one that matters
    for reading a clean screenshot.
    """
    from ..cards import rank_of, suit_of
    from ..vision.dataset import clean_batch

    Xs, yr, ys = [], [], []
    cards = list(range(52))
    for style in HOLDOUT_STYLES:
        Xs.append(clean_batch(cards, style, rng))
        yr.extend(rank_of(c) for c in cards)
        ys.extend(suit_of(c) for c in cards)
    return np.concatenate(Xs), np.array(yr), np.array(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=30000)
    ap.add_argument("--val", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--refresh", type=int, default=4,
                    help="regenerate the training pool every N epochs")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=DEFAULT_CARDNET)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start = time.time()
    print(f"generating {args.val} validation samples from held-out decks "
          f"{[s.name for s in HOLDOUT_STYLES]}")
    Xv, yrv, ysv = generate(args.val, True, args.seed + 999, args.workers)
    print(f"generating {args.train} training samples ({time.time() - start:.0f}s)")
    X, yr, ys = generate(args.train, False, args.seed, args.workers)
    print(f"data ready in {time.time() - start:.0f}s")

    Xc, yrc, ysc = clean_holdout(random.Random(args.seed))

    net = CardNet(rng=rng)
    opt = Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = SoftmaxCrossEntropy()
    best = 0.0

    for epoch in range(args.epochs):
        if epoch and args.refresh and epoch % args.refresh == 0:
            X, yr, ys = generate(args.train, False, args.seed + 100 * epoch, args.workers)
            print("  refreshed training pool")
        # Cosine decay keeps the last epochs stable.
        opt.lr = args.lr * (0.5 * (1 + np.cos(np.pi * epoch / max(1, args.epochs))))
        total, batches = 0.0, 0
        t0 = time.time()
        for idx in minibatches(len(X), args.batch, rng):
            opt.zero_grad()
            rl, sl = net.forward(X[idx], training=True)
            l1, g1 = loss_fn(rl, yr[idx])
            l2, g2 = loss_fn(sl, ys[idx])
            net.backward(g1, g2)
            opt.clip_grad_norm(5.0)
            opt.step()
            total += l1 + l2
            batches += 1
        r, s, c = accuracy(net, Xv, yrv, ysv)
        rc, sc, cc = accuracy(net, Xc, yrc, ysc)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / batches:.4f} "
              f"| augmented rank {r:.3f} suit {s:.3f} card {c:.3f} "
              f"| clean card {cc:.3f} | {time.time() - t0:.0f}s", flush=True)
        score = c + cc
        if score >= best:
            best = score
            save_cardnet(args.out, net, {
                "holdout_card_accuracy": round(c, 4),
                "holdout_rank_accuracy": round(r, 4),
                "holdout_suit_accuracy": round(s, 4),
                "clean_card_accuracy": round(cc, 4),
                "clean_rank_accuracy": round(rc, 4),
                "clean_suit_accuracy": round(sc, 4),
                "epoch": epoch + 1,
            })
    print(f"best combined score {best:.3f}; saved {args.out} "
          f"({time.time() - start:.0f}s total)")


if __name__ == "__main__":
    main()

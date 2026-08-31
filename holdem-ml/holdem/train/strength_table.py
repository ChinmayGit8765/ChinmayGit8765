"""Precompute made-hand score quantiles per street.

    python -m holdem.train.strength_table --samples 400000
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..evaluator_np import evaluate_batch
from ..ml.abstraction import DATA_DIR, PERCENTILE_PATH


def sample_scores(n_board: int, samples: int, rng: np.random.Generator) -> np.ndarray:
    total = 2 + n_board
    keys = rng.random((samples, 52))
    idx = np.argpartition(keys, total - 1, axis=1)[:, :total]
    return evaluate_batch(idx.astype(np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=400_000)
    ap.add_argument("--quantiles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tables = [[]]
    for n_board in (3, 4, 5):
        scores = np.sort(sample_scores(n_board, args.samples, rng))
        qs = np.linspace(0, len(scores) - 1, args.quantiles).astype(int)
        tables.append([int(v) for v in scores[qs]])
        print(f"board {n_board}: {len(scores)} samples, "
              f"median score {scores[len(scores) // 2]}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PERCENTILE_PATH, "w") as fh:
        json.dump(tables, fh)
    print("wrote", PERCENTILE_PATH)


if __name__ == "__main__":
    main()

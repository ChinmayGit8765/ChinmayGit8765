"""Generate the cached 169 x 8 preflop equity table.

    python -m holdem.train.preflop_table --iters 20000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ..cards import RANK_CHARS, make_card
from ..equity import DATA_DIR, PREFLOP_TABLE_PATH, equity


def combo_for_label(label: str):
    r1 = RANK_CHARS.index(label[0])
    r2 = RANK_CHARS.index(label[1])
    if len(label) == 2:  # pocket pair
        return [make_card(r1, 0), make_card(r2, 1)]
    if label[2] == "s":
        return [make_card(r1, 0), make_card(r2, 0)]
    return [make_card(r1, 0), make_card(r2, 1)]


def all_labels():
    out = []
    for i in range(12, -1, -1):
        for j in range(i, -1, -1):
            if i == j:
                out.append(RANK_CHARS[i] * 2)
            else:
                out.append(RANK_CHARS[i] + RANK_CHARS[j] + "s")
                out.append(RANK_CHARS[i] + RANK_CHARS[j] + "o")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--max-opponents", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    labels = all_labels()
    table = {}
    start = time.time()
    for n, label in enumerate(labels):
        hole = combo_for_label(label)
        table[label] = [
            round(equity(hole, (), opp, iters=args.iters, rng=rng), 5)
            for opp in range(1, args.max_opponents + 1)
        ]
        if n % 20 == 0:
            print(f"{n + 1}/{len(labels)} {label} {table[label][0]:.3f} "
                  f"({time.time() - start:.0f}s)", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PREFLOP_TABLE_PATH, "w") as fh:
        json.dump(table, fh, indent=0, sort_keys=True)
    print(f"wrote {PREFLOP_TABLE_PATH} ({len(table)} hands, {time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()

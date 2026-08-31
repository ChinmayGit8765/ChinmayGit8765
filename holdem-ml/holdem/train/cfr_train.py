"""Train the CFR blueprint, optionally across several cores.

    python -m holdem.train.cfr_train --iters 400000 --workers 4 --out models/blueprint.npz

Parallel MCCFR: every round each worker runs the same number of iterations from
a shared table, then returns only the *deltas* to its regret and average-strategy
counters.  Regret is additive across iterations, so summing deltas approximates
a longer sequential run while using every core.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from typing import Dict, Tuple

import numpy as np

from ..ml.cfr import Blueprint, MCCFRTrainer

Table = Dict[str, np.ndarray]


def _snapshot(table: Table) -> Table:
    return {k: v.copy() for k, v in table.items()}


def _delta(before: Table, after: Table) -> Table:
    out: Table = {}
    for key, value in after.items():
        prev = before.get(key)
        diff = value if prev is None else value - prev
        if np.any(diff):
            out[key] = diff.astype(np.float32)
    return out


def _worker(payload) -> Tuple[Table, Table]:
    seed, iters, regret, strategy, weight, stack, sb, bb = payload
    bp = Blueprint()
    bp.regret = {k: v.astype(np.float64) for k, v in regret.items()}
    bp.strategy_sum = {k: v.astype(np.float64) for k, v in strategy.items()}
    before_r, before_s = _snapshot(bp.regret), _snapshot(bp.strategy_sum)
    trainer = MCCFRTrainer(bp, stack=stack, sb=sb, bb=bb, rng=random.Random(seed),
                           discount=False, base_weight=weight)
    trainer.iterate(iters)
    return _delta(before_r, bp.regret), _delta(before_s, bp.strategy_sum)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--round-iters", type=int, default=4000, help="iterations per worker per round")
    ap.add_argument("--stack", type=int, default=200)
    ap.add_argument("--sb", type=int, default=1)
    ap.add_argument("--bb", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default="models/blueprint.npz")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    bp = Blueprint.load(args.resume) if args.resume and os.path.exists(args.resume) else Blueprint()
    rng = random.Random(args.seed)
    done = 0
    round_no = 0
    start = time.time()

    if args.workers <= 1:
        trainer = MCCFRTrainer(bp, stack=args.stack, sb=args.sb, bb=args.bb,
                               rng=random.Random(args.seed))
        trainer.iterate(args.iters, log_every=max(1, args.iters // 20))
        done = args.iters
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            while done < args.iters:
                round_no += 1
                per = min(args.round_iters, max(1, (args.iters - done) // args.workers) or 1)
                payloads = [
                    (rng.randrange(1 << 30), per,
                     {k: v.astype(np.float32) for k, v in bp.regret.items()},
                     {k: v.astype(np.float32) for k, v in bp.strategy_sum.items()},
                     float(round_no), args.stack, args.sb, args.bb)
                    for _ in range(args.workers)
                ]
                for regret_delta, strategy_delta in pool.map(_worker, payloads):
                    bp.merge_(regret_delta, strategy_delta)
                done += per * args.workers
                bp.iterations += per * args.workers
                bp.discount_regrets(done / (done + per * args.workers))
                elapsed = time.time() - start
                print(f"round {round_no}: {done}/{args.iters} iters, "
                      f"{len(bp.strategy_sum)} infosets, {done / elapsed:.0f} it/s",
                      flush=True)

    kept = bp.save(args.out, min_visits=1.0)
    print(f"wrote {args.out}: {kept} infosets after {bp.iterations} iterations "
          f"in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()

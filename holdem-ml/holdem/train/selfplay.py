"""Train the neural policy.

Two stages, run in order:

**distil** — play hands with the CFR blueprint driving the action and record
``(features, blueprint strategy)`` pairs, then fit the policy network to them
by cross-entropy.  This transplants the solved heads-up strategy into a network
that takes the *full* continuous state, so it generalises to table sizes and
stack depths the abstraction never enumerated.

**rl** — REINFORCE with the value head as a baseline, played out on the real
engine against a pool of frozen snapshots and rule bots.  Each decision is
credited with the chips the hand ultimately won, in big blinds; the value head
is regressed onto the same target, and an entropy bonus stops the policy
collapsing onto one action.

    python -m holdem.train.selfplay --stage distil --hands 4000
    python -m holdem.train.selfplay --stage rl --hands 20000
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..bots.neural import DEFAULT_BLUEPRINT, DEFAULT_POLICY, NeuralBot
from ..bots.rule import CallingStation, EquityBot, HonestBot, LooseAggressive, RandomBot, TightRock
from ..engine import Action, HandResult, Observation
from ..game import Game
from ..ml.abstraction import NUM_ACTIONS, legal_mask, to_engine_action
from ..ml.cfr import Blueprint, observation_infoset
from ..ml.features import encode
from ..ml.nn import Adam, SoftmaxCrossEntropy, minibatches, softmax
from ..ml.policy import PolicyValueNet, load_policy, save_policy


# Heads-up appears most often: it is the hardest spot and the one every
# evaluation is measured in.
SEAT_MIX = [2, 2, 2, 3, 4, 5, 6]


# --- shared plumbing --------------------------------------------------------

@dataclass
class Step:
    features: np.ndarray
    mask: np.ndarray
    action: int
    target: Optional[np.ndarray] = None  # distillation target
    ret: float = 0.0                     # realised return, big blinds


class RecordingBot(NeuralBot):
    """A NeuralBot that remembers every decision it made this hand."""

    def __init__(self, *args, equity_iters: int = 140, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace: List[Step] = []
        self.completed: List[Step] = []
        self.equity_iters = equity_iters

    def act(self, obs: Observation) -> Action:
        mask = legal_mask(obs)
        opponents = max(1, obs.live_opponents)
        equity = self._equity(obs, opponents, self.equity_iters)
        stats = self.tracker.aggregate_stats(self._opponent_names(obs))
        features = encode(obs, equity_field=equity, opponent_stats=stats)

        if self.policy is not None:
            probs = self.policy.action_probs(features, mask, self.difficulty.temperature)
        else:
            probs = self._prior_probs(equity) * mask
            probs = probs / max(1e-9, probs.sum())
        probs = np.where(mask, probs, 0.0)
        probs = probs / max(1e-9, probs.sum())
        index = int(self.rng.choices(range(NUM_ACTIONS), weights=probs, k=1)[0])
        self.trace.append(Step(features=features, mask=mask, action=index))
        return to_engine_action(obs, index)

    def on_hand_end(self, result: HandResult, seat: int) -> None:
        self.tracker.on_hand_end(result)
        ret = result.net.get(seat, 0) / 2.0  # big blinds
        for step in self.trace:
            step.ret = ret
        self.completed.extend(self.trace)
        self.trace = []

    def drain(self) -> List[Step]:
        out = self.completed
        self.completed = []
        return out


def build_pool(rng: random.Random, policy: Optional[PolicyValueNet],
               blueprint: Optional[Blueprint], size: int,
               snapshots: Sequence[PolicyValueNet] = ()) -> List:
    """Opponents for a training table: rule bots plus frozen policy snapshots."""
    pool = [
        EquityBot("equity", rng),
        TightRock("rock", rng),
        LooseAggressive("lag", rng),
        CallingStation("station", rng),
        HonestBot("honest", rng),
        RandomBot("random", rng),
    ]
    out = []
    for i in range(size):
        if snapshots and rng.random() < 0.45:
            snap = snapshots[rng.randrange(len(snapshots))]
            out.append(NeuralBot(f"snap{i}", policy=snap, blueprint=blueprint,
                                 difficulty=rng.choice(["regular", "strong", "pro"]),
                                 rng=rng, load_defaults=False))
        else:
            base = pool[rng.randrange(len(pool))]
            cls = type(base)
            out.append(cls(f"{base.name}{i}", rng))
    return out


# --- stage 1: distil the blueprint -----------------------------------------

class _DistilBot:
    """Blueprint bot that records the solved strategy at every spot it hits."""

    def __init__(self, name: str, blueprint: Blueprint, rng: random.Random,
                 equity_iters: int = 200):
        from ..bots.blueprint import BlueprintBot

        self.inner = BlueprintBot(name, blueprint=blueprint, rng=rng)
        self.name = name
        self.blueprint = blueprint
        self.equity_iters = equity_iters
        self.steps: List[Step] = []

    def act(self, obs: Observation) -> Action:
        mask = legal_mask(obs)
        key = observation_infoset(obs)
        if key in self.blueprint.strategy_sum:
            legal = [i for i in range(NUM_ACTIONS) if mask[i]]
            target = self.blueprint.average_strategy(key, legal) * mask
            total = target.sum()
            if total > 0:
                target = target / total
                self.steps.append(Step(
                    features=encode(obs, equity_iters=self.equity_iters),
                    mask=mask, action=int(np.argmax(target)), target=target,
                ))
        return self.inner.act(obs)


def collect_distillation(blueprint: Blueprint, hands: int, rng: random.Random,
                         seats: int = 2, equity_iters: int = 200,
                         log_every: int = 500) -> List[Step]:
    """Play hands with the blueprint driving, recording its strategy as labels."""
    bots = [_DistilBot(f"bp{i}", blueprint, rng, equity_iters) for i in range(seats)]
    game = Game(bots, rng=rng)
    start = time.time()
    for h in range(hands):
        game.play_hand(keep_history=False)
        if log_every and (h + 1) % log_every == 0:
            total = sum(len(b.steps) for b in bots)
            print(f"  distil hand {h + 1}/{hands}: {total} states "
                  f"({time.time() - start:.0f}s)", flush=True)
    steps: List[Step] = []
    for b in bots:
        steps.extend(b.steps)
    return steps


def fit_distillation(net: PolicyValueNet, steps: Sequence[Step], epochs: int = 12,
                     lr: float = 1e-3, batch_size: int = 256,
                     rng: Optional[np.random.Generator] = None) -> float:
    rng = rng or np.random.default_rng(0)
    X = np.stack([s.features for s in steps])
    M = np.stack([s.mask for s in steps])
    Y = np.stack([s.target for s in steps])
    opt = Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = SoftmaxCrossEntropy()
    last = 0.0
    for epoch in range(epochs):
        total, batches = 0.0, 0
        for idx in minibatches(len(X), batch_size, rng):
            opt.zero_grad()
            logits, _ = net.forward(X[idx], training=True)
            loss, grad = loss_fn(logits, Y[idx], mask=M[idx])
            net.backward(grad, None)
            opt.clip_grad_norm(5.0)
            opt.step()
            total += loss
            batches += 1
        last = total / max(1, batches)
        print(f"  epoch {epoch + 1}/{epochs}  cross-entropy {last:.4f}", flush=True)
    return last


# --- stage 2: policy-gradient self-play ------------------------------------

def rl_update(net: PolicyValueNet, steps: Sequence[Step], opt: Adam,
              entropy_beta: float = 0.01, value_weight: float = 0.5,
              return_scale: float = 12.0, clip: float = 3.0) -> Dict[str, float]:
    """One REINFORCE-with-baseline update over a batch of decisions."""
    X = np.stack([s.features for s in steps])
    M = np.stack([s.mask for s in steps])
    A = np.array([s.action for s in steps], dtype=int)
    R = np.clip(np.array([s.ret for s in steps], dtype=np.float32) / return_scale,
                -clip, clip)

    opt.zero_grad()
    logits, values = net.forward(X, training=True)
    masked = np.where(M, logits, -1e9)
    probs = softmax(masked) * M
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-9)

    baseline = values[:, 0]
    advantage = R - baseline
    n = len(steps)

    onehot = np.zeros_like(probs)
    onehot[np.arange(n), A] = 1.0
    grad_policy = (probs - onehot) * advantage[:, None] / n

    # Entropy bonus: dH/dz = -p (log p + H)
    logp = np.log(np.maximum(probs, 1e-9))
    entropy = -(probs * logp).sum(axis=1)
    grad_policy -= entropy_beta * (-probs * (logp + entropy[:, None])) / n
    grad_policy = np.where(M, grad_policy, 0.0)

    grad_value = (2.0 * value_weight * (baseline - R) / n)[:, None]

    net.backward(grad_policy.astype(np.float32), grad_value.astype(np.float32))
    norm = opt.clip_grad_norm(5.0)
    opt.step()

    return {
        "return": float(R.mean() * return_scale),
        "advantage": float(np.abs(advantage).mean()),
        "entropy": float(entropy.mean()),
        "value_mse": float(np.mean((baseline - R) ** 2)),
        "grad_norm": float(norm),
    }


def evaluate(net: PolicyValueNet, blueprint: Optional[Blueprint], hands: int,
             rng: random.Random, difficulty: str = "pro") -> Dict[str, float]:
    """bb/100 for the trained policy against each rule baseline, heads-up."""
    out = {}
    for maker in (lambda r: EquityBot("equity", r),
                  lambda r: TightRock("rock", r),
                  lambda r: CallingStation("station", r),
                  lambda r: LooseAggressive("lag", r)):
        local = random.Random(rng.randrange(1 << 30))
        hero = NeuralBot("hero", policy=net, blueprint=blueprint, difficulty=difficulty,
                         rng=local, load_defaults=False)
        villain = maker(local)
        game = Game([hero, villain], rng=local)
        game.run(hands, keep_history=False)
        out[villain.name] = game.stats.bb_per_100(0, game.bb)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["distil", "rl", "both"], default="both")
    ap.add_argument("--hands", type=int, default=8000)
    ap.add_argument("--distil-hands", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-hands", type=int, default=120, help="hands per RL update")
    ap.add_argument("--seats", type=int, default=6,
                    help="largest table size to train on (sizes are mixed)")
    ap.add_argument("--entropy", type=float, default=0.012)
    ap.add_argument("--eval-hands", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--blueprint", default=DEFAULT_BLUEPRINT)
    ap.add_argument("--out", default=DEFAULT_POLICY)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    blueprint = Blueprint.load(args.blueprint) if os.path.exists(args.blueprint) else None
    if blueprint:
        print(f"blueprint: {len(blueprint)} info-sets, {blueprint.iterations} iterations")

    net = load_policy(args.resume) if args.resume and os.path.exists(args.resume) \
        else PolicyValueNet(rng=np_rng)

    if args.stage in ("distil", "both") and blueprint is not None:
        print(f"stage 1: distilling blueprint over {args.distil_hands} hands")
        steps = collect_distillation(blueprint, args.distil_hands, rng)
        print(f"  collected {len(steps)} labelled states")
        if steps:
            fit_distillation(net, steps, epochs=args.epochs, rng=np_rng)
            save_policy(args.out, net, {"stage": "distil", "states": len(steps)})
            print(f"  saved {args.out}")

    if args.stage in ("rl", "both"):
        print(f"stage 2: self-play RL for {args.hands} hands")
        opt = Adam(net.parameters(), lr=args.lr, weight_decay=1e-6)
        snapshots: List[PolicyValueNet] = []
        played = 0
        start = time.time()
        while played < args.hands:
            # Table size is randomised every batch.  Training only 6-handed
            # produces a bot that folds far too much heads-up, where you have
            # to play most hands; mixing the seat count fixes that.
            seat_count = rng.choice(SEAT_MIX[:max(1, args.seats - 1)])
            hero = RecordingBot("hero", policy=net, blueprint=blueprint,
                                difficulty="pro", rng=rng, load_defaults=False)
            others = build_pool(rng, net, blueprint, seat_count - 1, snapshots)
            game = Game([hero, *others], rng=rng)
            batch = min(args.batch_hands, args.hands - played)
            game.run(batch, keep_history=False)
            played += batch
            steps = hero.drain()
            if not steps:
                continue
            stats = rl_update(net, steps, opt, entropy_beta=args.entropy)
            if played % (args.batch_hands * 10) < args.batch_hands:
                print(f"  {played}/{args.hands} hands | return {stats['return']:+.2f}bb "
                      f"| entropy {stats['entropy']:.2f} | value mse {stats['value_mse']:.3f} "
                      f"| {played / max(1e-6, time.time() - start):.1f} hands/s", flush=True)
            if played % 2000 < args.batch_hands:
                snap = PolicyValueNet(rng=np_rng)
                for dst, src in zip(snap.parameters(), net.parameters()):
                    dst.value = src.value.copy()
                snapshots.append(snap)
                snapshots[:] = snapshots[-4:]
                save_policy(args.out, net, {"stage": "rl", "hands": played})

        save_policy(args.out, net, {"stage": "rl", "hands": played})
        print(f"saved {args.out} after {played} hands in {time.time() - start:.0f}s")

    print("evaluation (bb/100, heads-up):")
    for name, bb100 in evaluate(net, blueprint, args.eval_hands, rng).items():
        print(f"  vs {name:<10}{bb100:+8.1f}")


if __name__ == "__main__":
    main()

"""Reproduce every number quoted in the README.

    python -m holdem.train.benchmark              # the standard run
    python -m holdem.train.benchmark --quick      # a fast smoke version

Nothing here is cached: each table is measured when you run it.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Sequence

import numpy as np

from ..analysis.analyzer import Analyzer
from ..analysis.promodel import DEFAULT_PRO_MODEL
from ..bots.blueprint import BlueprintBot, DEFAULT_BLUEPRINT
from ..bots.neural import DEFAULT_POLICY, NeuralBot
from ..bots.rule import (
    CallingStation, EquityBot, HonestBot, LooseAggressive, RandomBot, TightRock,
)
from ..cards import parse_cards
from ..equity import equity
from ..game import Game
from ..vision.cardnet import DEFAULT_CARDNET, load_cardnet
from ..vision.dataset import clean_batch
from ..vision.detect import CardReader
from ..vision.render import HOLDOUT_STYLES, TRAIN_STYLES, render_board

BASELINES = [
    ("EquityBot", EquityBot), ("TightRock", TightRock),
    ("CallingStation", CallingStation), ("LooseAggressive", LooseAggressive),
    ("RandomBot", RandomBot), ("HonestBot", HonestBot),
]


def rule(title: str) -> None:
    print(f"\n{title}\n" + "─" * len(title))


def heads_up(hero_factory, hands: int, seeds: Sequence[int]) -> List[float]:
    """Mean bb/100 against each baseline, averaged over several seeds.

    No-limit swings are large enough that a single seed is close to
    meaningless, so every figure here is a mean over independent sessions.
    """
    out = []
    for _, villain_cls in BASELINES:
        per_seed = []
        for seed in seeds:
            rng = random.Random(seed)
            hero = hero_factory(rng)
            game = Game([hero, villain_cls("villain", rng)], rng=rng)
            game.run(hands, keep_history=False)
            per_seed.append(game.stats.bb_per_100(0, game.bb))
        out.append(float(np.mean(per_seed)))
    return out


def print_row(label: str, values: List[float], width: int = 17) -> None:
    print(f"  {label:<{width}}" + "".join(f"{v:>+11.0f}" for v in values))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--hands", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None,
                    help="independent sessions to average over")
    args = ap.parse_args()
    hands = args.hands or (400 if args.quick else 2000)
    seeds = list(range(101, 101 + (args.seeds or (2 if args.quick else 4))))
    start = time.time()

    rule("Equity against published all-in numbers")
    reference = [("AsAd", 1, 0.852), ("AsAd", 5, 0.490), ("AsKs", 1, 0.670),
                 ("7c2d", 1, 0.351), ("JhTh", 1, 0.577)]
    for hand, opponents, published in reference:
        got = equity(parse_cards(hand), (), opponents,
                     iters=4000 if args.quick else 40000,
                     rng=np.random.default_rng(0))
        print(f"  {hand} vs {opponents}: {got:.3f}   published {published:.3f}"
              f"   diff {abs(got - published):.3f}")

    rule(f"Heads-up, big blinds per 100 hands "
         f"(mean of {len(seeds)} sessions x {hands} hands)")
    print("  " + " " * 17 + "".join(f"{name[:10]:>11}" for name, _ in BASELINES))
    if os.path.exists(DEFAULT_POLICY):
        for level in ("pro", "strong", "regular", "casual", "novice"):
            print_row(f"neural ({level})", heads_up(
                lambda rng, level=level: NeuralBot("hero", difficulty=level, rng=rng),
                hands, seeds))
    else:
        print("  (no trained policy — run holdem.train.selfplay)")
    if os.path.exists(DEFAULT_BLUEPRINT):
        print_row("CFR blueprint", heads_up(
            lambda rng: BlueprintBot("bp", rng=rng), hands, seeds))
    print_row("EquityBot", heads_up(lambda rng: EquityBot("hero", rng), hands, seeds))

    if os.path.exists(DEFAULT_POLICY):
        rule(f"Five-handed table ({len(seeds)} sessions x {hands} hands)")
        totals: dict = {}
        placings: List[int] = []
        for seed in seeds:
            rng = random.Random(seed)
            hero = NeuralBot("neural(pro)", difficulty="pro", rng=rng)
            table = [hero, EquityBot("equity", rng), TightRock("rock", rng),
                     CallingStation("station", rng), LooseAggressive("lag", rng)]
            game = Game(table, rng=rng)
            game.run(hands, keep_history=False)
            board = game.leaderboard()
            placings.append([n for n, _, _ in board].index("neural(pro)") + 1)
            for name, _net, bb100 in board:
                totals.setdefault(name, []).append(bb100)
        for name, values in sorted(totals.items(), key=lambda kv: -np.mean(kv[1])):
            print(f"  {name:<17}{np.mean(values):>+9.0f} bb/100"
                  f"   (per session: {', '.join(f'{v:+.0f}' for v in values)})")
        print(f"  neural(pro) finished in position {placings} "
              f"across the {len(seeds)} sessions")

    if os.path.exists(DEFAULT_CARDNET):
        rule("Card recognition")
        net = load_cardnet()
        cards = list(range(52))
        for label, styles in (("training decks", TRAIN_STYLES),
                              ("HELD-OUT decks", HOLDOUT_STYLES)):
            scores = []
            for style in styles:
                X = clean_batch(cards, style, random.Random(0))
                predicted, _ = net.predict(X)
                scores.append(float(np.mean([p == c for p, c in zip(predicted, cards)])))
            print(f"  clean renders, {label}: {np.mean(scores):.1%} "
                  f"(" + ", ".join(f"{s.name} {a:.0%}"
                                   for s, a in zip(styles, scores)) + ")")
        reader = CardReader(net=net)
        rng = random.Random(3)
        correct = total = detected = boards = 0
        for style in TRAIN_STYLES + HOLDOUT_STYLES:
            for _ in range(2 if args.quick else 5):
                deal = rng.sample(range(52), 5)
                img = render_board(deal, style, jitter=5, rng=rng)
                found = reader.read_table(img)
                boards += 1
                detected += int(len(found) == 5)
                if len(found) == 5:
                    correct += sum(1 for d, c in zip(found, deal) if d.card == c)
                    total += 5
        print(f"  detection: {detected}/{boards} boards found all five cards")
        print(f"  end-to-end reading: {correct}/{total} cards correct "
              f"({correct / max(1, total):.1%})")

    if os.path.exists(DEFAULT_PRO_MODEL):
        rule("Analyser: expected value lost per 100 hands")
        rng = random.Random(11)
        players = [EquityBot("Solid", rng), TightRock("Nit", rng),
                   LooseAggressive("Maniac", rng), RandomBot("Wild", rng)]
        game = Game(players, rng=rng)
        for _ in range(60 if args.quick else 300):
            game.play_hand()
        analyzer = Analyzer()
        for player in ("Solid", "Nit", "Maniac", "Wild"):
            report = analyzer.analyse_session(game.history, player)
            print(f"  {player:<8}{report.ev_loss_per_100:>8.1f} bb/100 lost   "
                  f"agreement {report.agreement:.0%}   "
                  f"{len(report.leaks)} leaks flagged")

    print(f"\ndone in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()

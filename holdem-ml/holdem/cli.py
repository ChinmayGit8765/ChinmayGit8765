"""Command line entry point.

    python -m holdem play      --seats 6 --difficulty adaptive
    python -m holdem serve     --port 8000 --bots 4
    python -m holdem analyse   hands.txt --player Hero
    python -m holdem read      table.png
    python -m holdem bench     --hands 2000
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Optional

from .analysis.analyzer import Analyzer
from .bots.blueprint import BlueprintBot
from .bots.neural import NeuralBot
from .bots.rule import CallingStation, EquityBot, HonestBot, LooseAggressive, RandomBot, TightRock
from .game import Game
from .human import HumanAgent, QuitGame
from .ml.difficulty import LEVELS_BY_NAME
from .ml.opponent import OpponentTracker
from .ui import Painter

BOT_NAMES = ["Nova", "Rook", "Vega", "Juno", "Atlas", "Kite", "Onyx", "Sable"]
BOT_CLASSES = {
    "neural": NeuralBot,
    "blueprint": BlueprintBot,
    "equity": EquityBot,
    "rock": TightRock,
    "lag": LooseAggressive,
    "station": CallingStation,
    "honest": HonestBot,
    "random": RandomBot,
}


def _make_bots(count: int, difficulty: str, rng: random.Random, human: str,
               tracker: OpponentTracker, kind: str = "neural") -> List:
    bots = []
    for i in range(count):
        name = BOT_NAMES[i % len(BOT_NAMES)]
        if kind == "neural":
            bots.append(NeuralBot(name, difficulty=difficulty, rng=rng,
                                  tracker=tracker, study=human))
        elif kind == "blueprint":
            bots.append(BlueprintBot(name, rng=rng))
        else:
            bots.append(BOT_CLASSES[kind](name, rng))
    return bots


def cmd_play(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    painter = Painter()
    analyzer = Analyzer()
    tracker = OpponentTracker()
    bots = _make_bots(args.seats - 1, args.difficulty, rng, args.name, tracker, args.bots)

    def info() -> str:
        """What one bot has worked out about everyone at the table."""
        watcher = next((b for b in bots if hasattr(b, "read_on")), None)
        lines = [f"  reads (as seen by {watcher.name}):" if watcher else "  no reads yet"]
        if watcher is not None:
            for name in [args.name] + [b.name for b in bots if b is not watcher]:
                lines.append("    " + watcher.read_on(name))
            status = getattr(watcher, "status", None)
            if status:
                lines.append("    " + status())
        return "\n".join(lines)

    human = HumanAgent(args.name, painter=painter, analyzer=analyzer, info_fn=info)
    seats = [human, *bots]
    rng.shuffle(seats)
    game = Game(seats, starting_stack=args.stack, sb=args.sb, bb=args.bb, rng=rng)

    print(painter.rule("holdem-ml"))
    print(f"  {args.seats}-handed, {args.stack} chips, blinds {args.sb}/{args.bb}, "
          f"bots on '{args.difficulty}'")
    print("  type ? at any decision for the model's read, i for table stats, q to quit")

    try:
        for _ in range(args.hands):
            game.play_hand()
            if all(p.stack <= 0 for p in game.players if p.name == args.name) \
                    and not game.auto_rebuy:
                print("  you are out of chips")
                break
    except (QuitGame, KeyboardInterrupt):
        print("\n  leaving the table")

    print()
    print(painter.rule("session"))
    print(game.report())
    if game.history:
        report = analyzer.analyse_session(game.history, args.name, sb=args.sb, bb=args.bb)
        print()
        print(report.summary())
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    tracker = OpponentTracker()
    roster = []
    for spec in args.lineup.split(","):
        spec = spec.strip()
        if ":" in spec:
            kind, level = spec.split(":", 1)
        else:
            kind, level = spec, "pro"
        name = f"{kind}-{level}" if kind == "neural" else kind
        if kind == "neural":
            roster.append(NeuralBot(name, difficulty=level, rng=rng, tracker=tracker))
        elif kind == "blueprint":
            roster.append(BlueprintBot(name, rng=rng))
        elif kind in BOT_CLASSES:
            roster.append(BOT_CLASSES[kind](name, rng))
        else:
            raise SystemExit(f"unknown bot {kind!r}; choose from {sorted(BOT_CLASSES)}")
    if len(roster) < 2:
        raise SystemExit("need at least two bots in --lineup")
    game = Game(roster, starting_stack=args.stack, sb=args.sb, bb=args.bb, rng=rng)
    game.run(args.hands, keep_history=False)
    print(game.report())
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    from .analysis.handhistory import parse_directory, parse_file, to_hand_result

    analyzer = Analyzer()
    if analyzer.model is None:
        print("note: no trained analyser model found; reporting equity-based advice only.\n"
              "      train one with: python -m holdem.train.train_analyst", file=sys.stderr)
    hands = parse_directory(args.path) if os.path.isdir(args.path) else parse_file(args.path)
    if not hands:
        raise SystemExit(f"no hands found in {args.path}")
    results = [to_hand_result(h) for h in hands]
    sb, bb = hands[0].small_blind, hands[0].big_blind
    player = args.player
    if player is None:
        counts: dict = {}
        for h in hands:
            for name in h.seats.values():
                counts[name] = counts.get(name, 0) + 1
        player = max(counts, key=counts.get)
        print(f"analysing {player} (most frequent player in the file)")
    report = analyzer.analyse_session(results, player, sb=sb, bb=bb,
                                      top_mistakes=args.top)
    print(report.summary())
    if args.hand_by_hand:
        for result in results:
            hand_report = analyzer.analyse_hand(result, player, sb=sb, bb=bb)
            if hand_report.decisions:
                print()
                print(hand_report.summary())
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    from .vision.cardnet import DEFAULT_CARDNET
    from .vision.detect import CardReader

    if not os.path.exists(DEFAULT_CARDNET):
        raise SystemExit("no trained card model; run: python -m holdem.train.train_vision")
    reader = CardReader()
    detections = reader.read_file(args.image)
    if not detections:
        print("no cards found in the image")
        return 1
    painter = Painter()
    print("detected:")
    for d in detections:
        print(f"  {painter.card(d.card)}  confidence {d.confidence:.0%}  box {d.box}")
    cards = [d.card for d in detections]
    if args.hole:
        analyzer = Analyzer()
        assessment = analyzer.assess_cards(cards[:args.hole], cards[args.hole:],
                                           opponents=args.opponents)
        print()
        print(f"  hole {assessment['hole']}   board {assessment['board']}")
        if "made_hand" in assessment:
            print(f"  made hand: {assessment['made_hand']} "
                  f"({assessment['strength_percentile']:.0%} percentile)")
        print(f"  equity vs {args.opponents}: {assessment['equity']:.1%}")
        print(f"  advice: {assessment['advice']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server.app import serve

    serve(host=args.host, port=args.port, bots=args.bots, difficulty=args.difficulty,
          stack=args.stack, sb=args.sb, bb=args.bb, seats=args.seats)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="holdem", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="play a table against the bots")
    play.add_argument("--seats", type=int, default=6)
    play.add_argument("--name", default="You")
    play.add_argument("--difficulty", default="adaptive",
                      help=f"{sorted(LEVELS_BY_NAME)}, a number in [0,1], or 'adaptive'")
    play.add_argument("--bots", default="neural", choices=sorted(BOT_CLASSES))
    play.add_argument("--stack", type=int, default=200)
    play.add_argument("--sb", type=int, default=1)
    play.add_argument("--bb", type=int, default=2)
    play.add_argument("--hands", type=int, default=10_000)
    play.add_argument("--seed", type=int, default=None)
    play.set_defaults(func=cmd_play)

    bench = sub.add_parser("bench", help="run bots against each other")
    bench.add_argument("--lineup", default="neural:pro,blueprint,equity,rock,lag,station")
    bench.add_argument("--hands", type=int, default=2000)
    bench.add_argument("--stack", type=int, default=200)
    bench.add_argument("--sb", type=int, default=1)
    bench.add_argument("--bb", type=int, default=2)
    bench.add_argument("--seed", type=int, default=1)
    bench.set_defaults(func=cmd_bench)

    analyse = sub.add_parser("analyse", help="grade hands from a hand-history file")
    analyse.add_argument("path")
    analyse.add_argument("--player", default=None)
    analyse.add_argument("--top", type=int, default=5)
    analyse.add_argument("--hand-by-hand", action="store_true")
    analyse.set_defaults(func=cmd_analyse)

    read = sub.add_parser("read", help="read cards from an image of a table")
    read.add_argument("image")
    read.add_argument("--hole", type=int, default=0,
                      help="treat the first N detected cards as your hole cards")
    read.add_argument("--opponents", type=int, default=1)
    read.set_defaults(func=cmd_read)

    serve = sub.add_parser("serve", help="run the multiplayer server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--bots", type=int, default=3)
    serve.add_argument("--seats", type=int, default=6)
    serve.add_argument("--difficulty", default="regular")
    serve.add_argument("--stack", type=int, default=200)
    serve.add_argument("--sb", type=int, default=1)
    serve.add_argument("--bb", type=int, default=2)
    serve.set_defaults(func=cmd_serve)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

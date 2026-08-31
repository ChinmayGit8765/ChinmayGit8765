import random

import pytest

from holdem.analysis.handhistory import format_hand
from holdem.bots.rule import CallingStation, EquityBot, LooseAggressive
from holdem.cli import build_parser, main
from holdem.engine import ActionType, PlayerState, Table
from holdem.game import Game
from holdem.human import HumanAgent, QuitGame
from holdem.ui import Painter, format_legal


def test_parser_exposes_every_command():
    parser = build_parser()
    for command in ("play", "bench", "analyse", "read", "serve"):
        args = parser.parse_args([command] + {"analyse": ["f.txt"], "read": ["i.png"]}
                                 .get(command, []))
        assert args.command == command
        assert callable(args.func)


def test_bench_runs_a_lineup(capsys):
    assert main(["bench", "--lineup", "equity,rock,station", "--hands", "40",
                 "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "bb/100" in out and "equity" in out


def test_bench_rejects_an_unknown_bot():
    with pytest.raises(SystemExit):
        main(["bench", "--lineup", "equity,nonsense", "--hands", "10"])


def test_analyse_reads_a_hand_history_file(tmp_path, capsys):
    rng = random.Random(12)
    game = Game([EquityBot("Hero", rng), LooseAggressive("Villain", rng),
                 CallingStation("Fish", rng)], rng=rng)
    for _ in range(30):
        game.play_hand()
    path = tmp_path / "hands.txt"
    path.write_text("\n\n".join(format_hand(r, sb=1, bb=2) for r in game.history))
    assert main(["analyse", str(path), "--player", "Hero"]) == 0
    out = capsys.readouterr().out
    assert "session report" in out and "Hero" in out


def test_analyse_picks_a_player_when_none_is_named(tmp_path, capsys):
    rng = random.Random(13)
    game = Game([EquityBot("Hero", rng), CallingStation("Fish", rng)], rng=rng)
    for _ in range(15):
        game.play_hand()
    path = tmp_path / "hands.txt"
    path.write_text("\n\n".join(format_hand(r, sb=1, bb=2) for r in game.history))
    assert main(["analyse", str(path)]) == 0
    assert "analysing" in capsys.readouterr().out


def test_painter_renders_cards_and_a_table():
    from holdem.cards import parse_cards as P

    painter = Painter(colour=False)
    assert painter.card(P("Ah")[0]) == "A♥"
    assert painter.cards(P("AhKs")) == "A♥ K♠"
    players = [PlayerState(i, f"P{i}", 200) for i in range(3)]
    table = Table(players, sb=1, bb=2, button=0, rng=random.Random(1))
    table.start_hand()
    text = painter.table(table.observation(), hero=0)
    assert "board" in text and "pot" in text and "P0" in text


def test_human_agent_parses_commands():
    players = [PlayerState(i, f"P{i}", 200) for i in range(3)]
    table = Table(players, sb=1, bb=2, button=0, rng=random.Random(2))
    table.start_hand()
    obs = table.observation()
    agent = HumanAgent("You", painter=Painter(colour=False),
                       input_fn=lambda _: "", output_fn=lambda _: None)

    assert agent._parse("f", obs).type == ActionType.FOLD
    assert agent._parse("c", obs).type == ActionType.CALL
    raise_action = agent._parse("r 12", obs)
    assert raise_action.type == ActionType.RAISE and raise_action.amount == 12
    assert agent._parse("a", obs).amount == obs.legal_of(ActionType.RAISE).max_amount
    assert agent._parse("pot", obs) is None
    pot_raise = agent._parse("r pot", obs)
    assert pot_raise.amount == obs.current_bet + obs.pot + obs.to_call
    half = agent._parse("r 50%", obs)
    assert half.amount == obs.current_bet + round(0.5 * (obs.pot + obs.to_call))
    assert agent._parse("r 1", obs).amount == obs.legal_of(ActionType.RAISE).min_amount, \
        "an undersized raise is clamped to the minimum"


def test_human_agent_quits_and_asks_for_advice():
    players = [PlayerState(i, f"P{i}", 200) for i in range(2)]
    table = Table(players, sb=1, bb=2, button=0, rng=random.Random(3))
    table.start_hand()
    lines = []
    replies = iter(["?", "i", "nonsense", "q"])
    agent = HumanAgent("You", painter=Painter(colour=False),
                       input_fn=lambda _: next(replies),
                       output_fn=lines.append,
                       info_fn=lambda: "  table info here")
    with pytest.raises(QuitGame):
        agent.act(table.observation())
    joined = "\n".join(lines)
    assert "would play" in joined, "'?' shows the model's read"
    assert "table info here" in joined
    assert "did not understand" in joined


def test_format_legal_lists_the_options():
    players = [PlayerState(i, f"P{i}", 200) for i in range(3)]
    table = Table(players, sb=1, bb=2, button=0, rng=random.Random(4))
    table.start_hand()
    text = format_legal(table.observation())
    assert "[f]old" in text and "[c]all" in text and "[a]ll-in" in text


def test_read_command_needs_a_trained_model(monkeypatch, tmp_path):
    from holdem.vision import cardnet

    monkeypatch.setattr(cardnet, "DEFAULT_CARDNET", str(tmp_path / "missing.npz"))
    import holdem.cli as cli_module
    with pytest.raises(SystemExit):
        cli_module.cmd_read(build_parser().parse_args(["read", "x.png"]))

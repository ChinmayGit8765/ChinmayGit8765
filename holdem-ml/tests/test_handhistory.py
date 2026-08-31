import random
import textwrap

from holdem.analysis.handhistory import (
    format_hand, parse_directory, parse_file, parse_hand, split_hands, to_hand_result,
)
from holdem.analysis.replay import replay_hand
from holdem.bots.rule import CallingStation, EquityBot, LooseAggressive, TightRock
from holdem.cards import cards_str
from holdem.game import Game

REAL_WORLD_SAMPLE = textwrap.dedent("""\
    PokerStars Hand #241234567890:  Hold'em No Limit ($0.50/$1.00 USD) - 2024/03/02 21:14:03 ET
    Table 'Sirius III' 6-max Seat #3 is the button
    Seat 1: Alice ($100.00 in chips)
    Seat 2: Bob ($121.50 in chips)
    Seat 3: Carol ($98.00 in chips)
    Alice: posts small blind $0.50
    Bob: posts big blind $1.00
    *** HOLE CARDS ***
    Dealt to Alice [Ah Kd]
    Carol: folds
    Alice: raises $2.00 to $3.00
    Bob: calls $2.00
    *** FLOP *** [7h 2d 9c]
    Alice: bets $4.00
    Bob: calls $4.00
    *** TURN *** [7h 2d 9c] [Ks]
    Alice: bets $12.00
    Bob: folds
    Uncalled bet ($12.00) returned to Alice
    Alice collected $13.75 from pot
    *** SUMMARY ***
    Total pot $14.00 | Rake $0.25
    Board [7h 2d 9c Ks]
    """)


def test_parses_a_pokerstars_style_hand():
    hand = parse_hand(REAL_WORLD_SAMPLE)
    assert hand is not None
    assert hand.small_blind == 50 and hand.big_blind == 100
    assert hand.seats == {0: "Alice", 1: "Bob", 2: "Carol"}
    assert hand.stacks[0] == 10000
    assert cards_str(hand.hole[0]) == "Ah Kd"
    assert cards_str(hand.board) == "7h 2d 9c Ks"
    assert hand.collected[0] == 1375
    kinds = [(a.name, a.action.name, a.to_amount, a.street) for a in hand.actions]
    assert ("Carol", "FOLD", 0, 0) in kinds
    assert ("Alice", "RAISE", 300, 0) in kinds
    assert ("Alice", "BET", 1200, 2) in kinds


def test_parsed_hand_can_be_replayed():
    hand = parse_hand(REAL_WORLD_SAMPLE)
    result = to_hand_result(hand)
    points = replay_hand(result, sb=hand.small_blind, bb=hand.big_blind)
    assert [p.name for p in points] == [a.name for a in hand.actions]
    assert points[1].obs.hole == hand.hole[0], "Alice's known cards survive the replay"


def test_write_then_read_reproduces_the_hand():
    rng = random.Random(21)
    game = Game([EquityBot("Alice", rng), LooseAggressive("Bob", rng),
                 TightRock("Carol", rng), CallingStation("Dan", rng)], rng=rng)
    for _ in range(80):
        game.play_hand()
    for result in game.history:
        text = format_hand(result, sb=1, bb=2)
        parsed = parse_hand(text)
        assert parsed is not None
        replayed = replay_hand(to_hand_result(parsed),
                               sb=parsed.small_blind, bb=parsed.big_blind)
        assert [p.record.action for p in replayed] == [r.action for r in result.history]
        assert [p.name for p in replayed] == [r.name for r in result.history]


def test_uncalled_bets_are_reported():
    rng = random.Random(8)
    game = Game([EquityBot("Alice", rng), TightRock("Bob", rng)], rng=rng)
    texts = [format_hand(game.play_hand(), sb=1, bb=2) for _ in range(60)]
    assert any("Uncalled bet" in t for t in texts)


def test_split_hands_and_files(tmp_path):
    blob = REAL_WORLD_SAMPLE + "\n" + REAL_WORLD_SAMPLE.replace("241234567890", "241234567891")
    assert len(split_hands(blob)) == 2
    path = tmp_path / "hands.txt"
    path.write_text(blob)
    assert len(parse_file(str(path))) == 2
    assert len(parse_directory(str(tmp_path))) == 2


def test_non_hand_text_is_ignored():
    assert parse_hand("just some notes\nnot a hand at all") is None

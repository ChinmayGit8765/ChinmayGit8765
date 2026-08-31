"""Read and write PokerStars-style hand histories.

Writing lets this project export its own games in a format every poker tool
understands.  Reading lets the analyser work on *real* hands — point it at a
directory of hand histories from an actual site and it will grade them with the
same machinery it uses on bot games.

Money is converted to integer cents so the engine can stay in whole chips.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..cards import card_str, cards_str, parse_card, parse_cards
from ..engine import (
    STREET_NAMES,
    ActionRecord,
    ActionType,
    HandResult,
    PotResult,
)

_AMOUNT = r"[$€£]?([0-9]+(?:\.[0-9]+)?)"
RE_HEADER = re.compile(
    r"(?:PokerStars|Holdem-ML)\s+(?:Hand|Game)\s+#(\d+):?\s+"
    r"(?:.*?)\(" + _AMOUNT + r"/" + _AMOUNT + r".*?\)"
)
RE_TABLE = re.compile(r"Table\s+'([^']*)'.*?Seat #(\d+) is the button")
RE_SEAT = re.compile(r"Seat (\d+): (.+?) \(" + _AMOUNT + r" in chips\)")
RE_POST = re.compile(r"^(.+?): posts (small blind|big blind|the ante) " + _AMOUNT)
RE_DEALT = re.compile(r"^Dealt to (.+?) \[(.+?)\]")
RE_ACTION = re.compile(
    r"^(.+?): (folds|checks|calls|bets|raises)"
    r"(?:\s+" + _AMOUNT + r")?(?:\s+to\s+" + _AMOUNT + r")?"
)
RE_BOARD_FLOP = re.compile(r"\*\*\* FLOP \*\*\* \[(.+?)\]")
RE_BOARD_TURN = re.compile(r"\*\*\* TURN \*\*\* \[.+?\] \[(.+?)\]")
RE_BOARD_RIVER = re.compile(r"\*\*\* RIVER \*\*\* \[.+?\] \[(.+?)\]")
RE_SHOW = re.compile(r"^(.+?): (?:shows|mucks hand) \[(.+?)\]")
RE_COLLECT = re.compile(r"^(.+?) collected " + _AMOUNT + r" from")
RE_UNCALLED = re.compile(r"Uncalled bet \(" + _AMOUNT + r"\) returned to (.+)$")
RE_SUMMARY_BOARD = re.compile(r"^Board \[(.+?)\]")

VERB_TO_ACTION = {
    "folds": ActionType.FOLD,
    "checks": ActionType.CHECK,
    "calls": ActionType.CALL,
    "bets": ActionType.BET,
    "raises": ActionType.RAISE,
}


def cents(value: str | float) -> int:
    return int(round(float(value) * 100))


def money(chips: int) -> str:
    return f"${chips / 100:.2f}"


@dataclass
class ParsedAction:
    name: str
    action: ActionType
    amount: int          # chips moved by this action
    to_amount: int       # resulting street total
    street: int


@dataclass
class ParsedHand:
    hand_id: str
    small_blind: int
    big_blind: int
    ante: int
    table: str
    button_seat: int
    seats: Dict[int, str]              # seat index (0-based) -> name
    stacks: Dict[int, int]
    hole: Dict[int, List[int]]
    board: List[int]
    actions: List[ParsedAction]
    collected: Dict[int, int] = field(default_factory=dict)
    shown: Dict[int, List[int]] = field(default_factory=dict)
    uncalled: Optional[Tuple[int, int]] = None   # (seat, chips) returned unmatched
    source: str = ""

    def seat_of(self, name: str) -> Optional[int]:
        for seat, player in self.seats.items():
            if player == name:
                return seat
        return None

    @property
    def names(self) -> Dict[int, str]:
        return dict(self.seats)


def format_hand(result: HandResult, sb: int = 1, bb: int = 2,
                hand_id: Optional[int] = None, table: str = "Holdem-ML",
                hero: Optional[str] = None, scale: float = 0.01,
                timestamp: Optional[datetime] = None) -> str:
    """Render a :class:`HandResult` as a PokerStars-style history."""
    def amt(chips: int) -> str:
        return f"${chips * scale:.2f}"

    ts = (timestamp or datetime(2024, 1, 1, 12, 0, 0)).strftime("%Y/%m/%d %H:%M:%S")
    seats = sorted(result.stacks_before)
    lines = [
        f"Holdem-ML Hand #{hand_id or result.hand_number}:  Hold'em No Limit "
        f"({amt(sb)}/{amt(bb)} USD) - {ts} ET",
        f"Table '{table}' {len(seats)}-max Seat #{result.button + 1} is the button",
    ]
    for seat in seats:
        lines.append(f"Seat {seat + 1}: {result.names[seat]} "
                     f"({amt(result.stacks_before[seat])} in chips)")

    posts = [r for r in result.history if False]  # blinds are implicit in the engine
    order = [(result.button + 1 + k) % len(seats) for k in range(len(seats))]
    live = [s for s in order if s in result.stacks_before]
    if len(live) == 2:
        sb_seat, bb_seat = result.button, [s for s in live if s != result.button][0]
    else:
        sb_seat, bb_seat = live[0], live[1]
    lines.append(f"{result.names[sb_seat]}: posts small blind {amt(sb)}")
    lines.append(f"{result.names[bb_seat]}: posts big blind {amt(bb)}")

    lines.append("*** HOLE CARDS ***")
    if hero and hero in result.names.values():
        seat = [s for s, n in result.names.items() if n == hero][0]
        if seat in result.hole:
            lines.append(f"Dealt to {hero} [{cards_str(result.hole[seat])}]")
    else:
        for seat in seats:
            if seat in result.hole:
                lines.append(f"Dealt to {result.names[seat]} "
                             f"[{cards_str(result.hole[seat])}]")

    street = 0
    for record in result.history:
        while record.street > street:
            street += 1
            if street == 1 and len(result.board) >= 3:
                lines.append(f"*** FLOP *** [{cards_str(result.board[:3])}]")
            elif street == 2 and len(result.board) >= 4:
                lines.append(f"*** TURN *** [{cards_str(result.board[:3])}] "
                             f"[{card_str(result.board[3])}]")
            elif street == 3 and len(result.board) >= 5:
                lines.append(f"*** RIVER *** [{cards_str(result.board[:4])}] "
                             f"[{card_str(result.board[4])}]")
        verb = {
            ActionType.FOLD: "folds",
            ActionType.CHECK: "checks",
            ActionType.CALL: f"calls {amt(record.amount)}",
            ActionType.BET: f"bets {amt(record.amount)}",
            ActionType.RAISE: f"raises {amt(record.amount)} to {amt(record.to_amount)}",
        }[record.action]
        lines.append(f"{record.name}: {verb}")

    if result.showdown:
        lines.append("*** SHOW DOWN ***")
        for seat, hole in result.revealed.items():
            lines.append(f"{result.names[seat]}: shows [{cards_str(hole)}]")

    if result.uncalled:
        seat, chips = result.uncalled
        lines.append(f"Uncalled bet ({amt(chips)}) returned to {result.names[seat]}")
    for pot in result.pots:
        for seat in pot.winners:
            lines.append(f"{result.names[seat]} collected {amt(pot.per_winner)} from pot")

    lines.append("*** SUMMARY ***")
    lines.append(f"Total pot {amt(sum(p.amount for p in result.pots))} | Rake $0.00")
    if result.board:
        lines.append(f"Board [{cards_str(result.board)}]")
    return "\n".join(lines)


def parse_hand(text: str, source: str = "") -> Optional[ParsedHand]:
    """Parse one hand history block.  Returns ``None`` if it is not a hand."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    header = RE_HEADER.search(lines[0])
    if not header:
        return None
    hand_id, sb, bb = header.group(1), cents(header.group(2)), cents(header.group(3))

    table_name, button_seat = "", 1
    seats: Dict[int, str] = {}
    stacks: Dict[int, int] = {}
    hole: Dict[int, List[int]] = {}
    shown: Dict[int, List[int]] = {}
    board: List[int] = []
    actions: List[ParsedAction] = []
    collected: Dict[int, int] = {}
    uncalled: Optional[Tuple[int, int]] = None
    ante = 0
    street = 0
    street_totals: Dict[str, int] = {}
    seat_numbers: List[int] = []

    for line in lines[1:]:
        m = RE_TABLE.search(line)
        if m:
            table_name, button_seat = m.group(1), int(m.group(2))
            continue
        m = RE_SEAT.match(line)
        if m:
            seat_numbers.append(int(m.group(1)))
            seats[int(m.group(1))] = m.group(2).strip()
            stacks[int(m.group(1))] = cents(m.group(3))
            continue
        m = RE_POST.match(line)
        if m:
            name, kind, value = m.group(1), m.group(2), cents(m.group(3))
            if kind == "the ante":
                ante = max(ante, value)
            else:
                street_totals[name] = street_totals.get(name, 0) + value
            continue
        m = RE_DEALT.match(line)
        if m:
            seat = _seat_by_name(seats, m.group(1))
            if seat is not None:
                hole[seat] = parse_cards(m.group(2))
            continue
        if line.startswith("*** FLOP"):
            street = 1
            street_totals = {}
            mm = RE_BOARD_FLOP.search(line)
            if mm:
                board = parse_cards(mm.group(1))
            continue
        if line.startswith("*** TURN"):
            street = 2
            street_totals = {}
            mm = RE_BOARD_TURN.search(line)
            if mm:
                board = board[:3] + parse_cards(mm.group(1))
            continue
        if line.startswith("*** RIVER"):
            street = 3
            street_totals = {}
            mm = RE_BOARD_RIVER.search(line)
            if mm:
                board = board[:4] + parse_cards(mm.group(1))
            continue
        if line.startswith("*** SHOW") or line.startswith("*** SUMMARY"):
            street = max(street, 3)
            m = RE_SUMMARY_BOARD.match(line)
            continue
        m = RE_SHOW.match(line)
        if m:
            seat = _seat_by_name(seats, m.group(1))
            if seat is not None:
                cards = parse_cards(m.group(2))
                shown[seat] = cards
                hole.setdefault(seat, cards)
            continue
        m = RE_UNCALLED.search(line)
        if m:
            seat = _seat_by_name(seats, m.group(2))
            if seat is not None:
                uncalled = (seat, cents(m.group(1)))
            continue
        m = RE_COLLECT.match(line)
        if m:
            seat = _seat_by_name(seats, m.group(1))
            if seat is not None:
                collected[seat] = collected.get(seat, 0) + cents(m.group(2))
            continue
        m = RE_SUMMARY_BOARD.match(line)
        if m:
            board = parse_cards(m.group(1))
            continue
        m = RE_ACTION.match(line)
        if m:
            name, verb = m.group(1), m.group(2)
            if _seat_by_name(seats, name) is None:
                continue
            first = m.group(3)
            second = m.group(4)
            action = VERB_TO_ACTION[verb]
            prev = street_totals.get(name, 0)
            if action in (ActionType.FOLD, ActionType.CHECK):
                amount, to_amount = 0, prev
            elif action == ActionType.RAISE:
                to_amount = cents(second) if second else prev + cents(first or 0)
                amount = to_amount - prev
            elif action == ActionType.BET:
                amount = cents(first or 0)
                to_amount = prev + amount
            else:  # call
                amount = cents(first or 0)
                to_amount = prev + amount
            street_totals[name] = to_amount
            actions.append(ParsedAction(name=name, action=action, amount=amount,
                                        to_amount=to_amount, street=street))

    if not seats:
        return None
    # Renumber seats to a dense 0-based ring, preserving order round the table.
    ordered = sorted(seats)
    remap = {old: i for i, old in enumerate(ordered)}
    return ParsedHand(
        hand_id=hand_id,
        small_blind=sb,
        big_blind=bb,
        ante=ante,
        table=table_name,
        button_seat=remap.get(button_seat, 0),
        seats={remap[s]: n for s, n in seats.items()},
        stacks={remap[s]: v for s, v in stacks.items()},
        hole={remap[s]: v for s, v in hole.items()},
        shown={remap[s]: v for s, v in shown.items()},
        board=board,
        actions=actions,
        collected={remap[s]: v for s, v in collected.items()},
        uncalled=(remap[uncalled[0]], uncalled[1]) if uncalled else None,
        source=source,
    )


def _seat_by_name(seats: Dict[int, str], name: str) -> Optional[int]:
    name = name.strip()
    for seat, player in seats.items():
        if player == name:
            return seat
    return None


def split_hands(text: str) -> List[str]:
    """Split a history file into individual hand blocks."""
    blocks, current = [], []
    for line in text.splitlines():
        if RE_HEADER.search(line) and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return [b for b in blocks if b.strip()]


def parse_file(path: str) -> List[ParsedHand]:
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    out = []
    for block in split_hands(text):
        hand = parse_hand(block, source=os.path.basename(path))
        if hand:
            out.append(hand)
    return out


def parse_directory(path: str, pattern: str = ".txt") -> List[ParsedHand]:
    hands: List[ParsedHand] = []
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            if name.endswith(pattern):
                hands.extend(parse_file(os.path.join(root, name)))
    return hands


def to_hand_result(parsed: ParsedHand) -> HandResult:
    """Convert a parsed hand into the engine's own result type.

    Replaying it (see :mod:`holdem.analysis.replay`) then recovers the exact
    decision context for every action, which is what the analyser grades.
    """
    history: List[ActionRecord] = []
    pot = sum(parsed.stacks.values()) * 0  # running pot, filled below
    committed: Dict[str, int] = {}
    street_committed: Dict[str, int] = {}
    current_street = 0
    running_pot = parsed.small_blind + parsed.big_blind + parsed.ante * len(parsed.seats)

    for act in parsed.actions:
        if act.street != current_street:
            current_street = act.street
            street_committed = {}
        seat = parsed.seat_of(act.name)
        if seat is None:
            continue
        prev = street_committed.get(act.name, 0)
        to_call = max(0, max(street_committed.values(), default=0) - prev)
        history.append(ActionRecord(
            street=act.street, seat=seat, name=act.name, action=act.action,
            amount=act.amount, to_amount=act.to_amount, pot_before=running_pot,
            to_call=to_call, stack_before=parsed.stacks.get(seat, 0),
        ))
        street_committed[act.name] = act.to_amount
        running_pot += act.amount
        committed[act.name] = committed.get(act.name, 0) + act.amount

    net: Dict[int, int] = {}
    for seat, name in parsed.seats.items():
        net[seat] = parsed.collected.get(seat, 0) - committed.get(name, 0)

    total = running_pot
    winners = [s for s, v in parsed.collected.items() if v > 0]
    pots = [PotResult(amount=total, winners=winners, eligible=list(parsed.seats),
                      per_winner=total // max(1, len(winners)),
                      description="from history")] if winners else []

    return HandResult(
        hand_number=int(re.sub(r"\D", "", parsed.hand_id) or 0),
        button=parsed.button_seat,
        board=list(parsed.board),
        hole=dict(parsed.hole),
        revealed=dict(parsed.shown),
        net=net,
        pots=pots,
        uncalled=parsed.uncalled,
        history=history,
        showdown=bool(parsed.shown),
        stacks_before=dict(parsed.stacks),
        names=dict(parsed.seats),
    )

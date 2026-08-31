"""Terminal rendering helpers shared by the CLI."""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

from .cards import RANK_CHARS, SUIT_CHARS, card_pretty, rank_of, suit_of
from .engine import Observation, STREET_NAMES

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
GREY = "\033[90m"


def colour_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


class Painter:
    def __init__(self, colour: Optional[bool] = None):
        self.colour = colour_enabled() if colour is None else colour

    def paint(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.colour else text

    def card(self, card: int) -> str:
        text = card_pretty(card)
        if not self.colour:
            return text
        suit = SUIT_CHARS[suit_of(card)]
        return self.paint(text, RED if suit in "dh" else BOLD)

    def cards(self, cards: Iterable[int]) -> str:
        return " ".join(self.card(c) for c in cards) or self.paint("--", GREY)

    def rule(self, label: str = "", width: int = 62) -> str:
        if not label:
            return self.paint("─" * width, GREY)
        pad = max(0, width - len(label) - 4)
        return self.paint(f"── {label} " + "─" * pad, GREY)

    def table(self, obs: Observation, hero: int, reveal: Optional[Dict[int, List[int]]] = None,
              difficulty: Optional[str] = None) -> str:
        reveal = reveal or {}
        lines = [self.rule(f"hand {obs.hand_number} · blinds {obs.small_blind}/{obs.big_blind}"
                           + (f" · bots: {difficulty}" if difficulty else ""))]
        for seat in range(obs.num_players):
            tags = []
            if seat == obs.button:
                tags.append(self.paint("BTN", YELLOW))
            if seat == hero:
                tags.append(self.paint("you", CYAN))
            if obs.folded[seat]:
                tags.append(self.paint("folded", GREY))
            elif obs.all_in[seat]:
                tags.append(self.paint("all-in", RED))
            hole = ""
            if seat == hero:
                hole = self.cards(obs.hole)
            elif seat in reveal:
                hole = self.cards(reveal[seat])
            committed = obs.street_committed[seat]
            bet = f"bet {committed}" if committed else ""
            name = obs.names[seat]
            body = f"  {name:<12}{obs.stacks[seat]:>6}  {bet:<9}{hole:<12}{' '.join(tags)}"
            lines.append(self.paint(body, GREY) if obs.folded[seat] else body)
        board = self.cards(obs.board) if obs.board else self.paint("(preflop)", GREY)
        lines.append(f"  board {board}    pot {self.paint(str(obs.pot), BOLD)}")
        return "\n".join(lines)


def format_legal(obs: Observation) -> str:
    from .engine import ActionType

    parts = []
    for la in obs.legal:
        if la.type == ActionType.FOLD:
            parts.append("[f]old")
        elif la.type == ActionType.CHECK:
            parts.append("[c]heck")
        elif la.type == ActionType.CALL:
            parts.append(f"[c]all {la.min_amount}")
        elif la.type in (ActionType.BET, ActionType.RAISE):
            verb = "bet" if la.type == ActionType.BET else "raise"
            parts.append(f"[r]{verb} {la.min_amount}-{la.max_amount}")
            parts.append("[a]ll-in")
    parts += ["[?]advice", "[i]nfo", "[q]uit"]
    return "  ".join(parts)

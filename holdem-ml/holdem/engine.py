"""No-limit Texas Hold'em rules engine.

Implements a full hand as an explicit state machine so the same code drives the
terminal game, the multiplayer server, the self-play trainer and the replayer:

    table = Table(players, sb=1, bb=2)
    table.start_hand()
    while not table.hand_over:
        obs = table.observation()
        table.apply(agent.act(obs))
    results = table.results

Covers blind posting (including heads-up button-is-small-blind), antes, correct
min-raise sizing, short all-in raises that do *not* reopen the betting, and
multi-way side pots with odd-chip distribution left of the button.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

from .cards import Deck, card_str, cards_str
from .evaluator import describe, evaluate

PREFLOP, FLOP, TURN, RIVER, SHOWDOWN = 0, 1, 2, 3, 4
STREET_NAMES = ["preflop", "flop", "turn", "river", "showdown"]


class ActionType(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4


@dataclass(frozen=True)
class Action:
    """A player decision.

    ``amount`` is the *total* street contribution the player moves to (a
    "raise to" amount), which makes min-raise arithmetic unambiguous.  It is
    ignored for fold/check and derived for call.
    """

    type: ActionType
    amount: int = 0

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        if self.type in (ActionType.BET, ActionType.RAISE):
            return f"{self.type.name.lower()} to {self.amount}"
        return self.type.name.lower()


@dataclass
class PlayerState:
    seat: int
    name: str
    stack: int
    hole: List[int] = field(default_factory=list)
    folded: bool = False
    all_in: bool = False
    street_committed: int = 0
    total_committed: int = 0
    acted: bool = False
    can_raise: bool = True
    in_hand: bool = False  # dealt in this hand
    net: int = 0           # chips won/lost this hand

    @property
    def live(self) -> bool:
        """Still contesting the pot (may be all-in)."""
        return self.in_hand and not self.folded

    @property
    def can_act(self) -> bool:
        return self.live and not self.all_in and self.stack > 0


@dataclass
class ActionRecord:
    street: int
    seat: int
    name: str
    action: ActionType
    amount: int          # chips actually moved into the pot by this action
    to_amount: int       # resulting street contribution
    pot_before: int
    to_call: int
    stack_before: int

    def describe(self) -> str:
        t = self.action
        if t == ActionType.FOLD:
            return f"{self.name} folds"
        if t == ActionType.CHECK:
            return f"{self.name} checks"
        if t == ActionType.CALL:
            return f"{self.name} calls {self.amount}"
        verb = "bets" if t == ActionType.BET else "raises to"
        return f"{self.name} {verb} {self.to_amount}"


@dataclass
class LegalAction:
    type: ActionType
    min_amount: int = 0
    max_amount: int = 0


@dataclass
class Observation:
    """Everything a decision-maker at ``seat`` is allowed to see."""

    seat: int
    name: str
    hole: List[int]
    board: List[int]
    street: int
    pot: int
    stacks: List[int]
    street_committed: List[int]
    total_committed: List[int]
    folded: List[bool]
    all_in: List[bool]
    in_hand: List[bool]
    names: List[str]
    button: int
    big_blind: int
    small_blind: int
    current_bet: int
    min_raise: int
    to_call: int
    legal: List[LegalAction]
    history: List[ActionRecord]
    hand_number: int

    @property
    def num_players(self) -> int:
        return len(self.stacks)

    @property
    def my_stack(self) -> int:
        return self.stacks[self.seat]

    @property
    def live_opponents(self) -> int:
        return sum(
            1 for i in range(self.num_players)
            if i != self.seat and self.in_hand[i] and not self.folded[i]
        )

    @property
    def pot_odds(self) -> float:
        """Fraction of the final pot you'd be paying to call."""
        if self.to_call <= 0:
            return 0.0
        return self.to_call / (self.pot + self.to_call)

    def has(self, t: ActionType) -> bool:
        return any(la.type == t for la in self.legal)

    def legal_of(self, t: ActionType) -> Optional[LegalAction]:
        for la in self.legal:
            if la.type == t:
                return la
        return None


@dataclass
class PotResult:
    amount: int
    winners: List[int]
    eligible: List[int]
    per_winner: int
    description: str


@dataclass
class HandResult:
    hand_number: int
    button: int
    board: List[int]
    hole: Dict[int, List[int]]      # ground truth (trainers/analysers only)
    revealed: Dict[int, List[int]]  # what opponents are actually allowed to see
    net: Dict[int, int]
    pots: List[PotResult]
    uncalled: Optional[Tuple[int, int]]  # (seat, chips) returned unmatched
    history: List[ActionRecord]
    showdown: bool
    stacks_before: Dict[int, int]
    names: Dict[int, str]

    @property
    def total_pot(self) -> int:
        return sum(p.amount for p in self.pots)


class Table:
    """One hand of no-limit Hold'em among the seated players."""

    def __init__(
        self,
        players: Sequence[PlayerState],
        sb: int = 1,
        bb: int = 2,
        ante: int = 0,
        button: int = 0,
        rng: Optional[random.Random] = None,
        hand_number: int = 0,
    ):
        self.players = list(players)
        self.sb = sb
        self.bb = bb
        self.ante = ante
        self.button = button % max(1, len(self.players))
        self.rng = rng or random.Random()
        self.hand_number = hand_number

        self.deck = Deck(self.rng)
        self.board: List[int] = []
        self.street = PREFLOP
        self.pot = 0
        self.current_bet = 0
        self.min_raise = bb
        self.history: List[ActionRecord] = []
        self.hand_over = False
        self.uncalled: Optional[Tuple[int, int]] = None
        self.results: Optional[HandResult] = None
        self._actor: Optional[int] = None
        self._stacks_before: Dict[int, int] = {}

    # -- setup ---------------------------------------------------------------

    def _seated(self) -> List[PlayerState]:
        return [p for p in self.players if p.stack > 0]

    def start_hand(self, hole_cards: Optional[Dict[int, List[int]]] = None,
                   board: Optional[Sequence[int]] = None) -> None:
        seated = self._seated()
        if len(seated) < 2:
            raise ValueError("need at least two players with chips")

        for p in self.players:
            p.hole = []
            p.folded = False
            p.all_in = False
            p.street_committed = 0
            p.total_committed = 0
            p.acted = False
            p.can_raise = True
            p.net = 0
            p.in_hand = p.stack > 0
        self._stacks_before = {p.seat: p.stack for p in self.players}

        # Pull every pinned card out of the deck *before* dealing, so randomly
        # dealt hole cards can never collide with a forced board.
        if hole_cards:
            flat = [c for cards in hole_cards.values() for c in cards]
            self.deck.remove(flat)
        if board:
            self.deck.remove(board)
        for p in seated:
            if hole_cards and p.seat in hole_cards:
                p.hole = list(hole_cards[p.seat])
            else:
                p.hole = self.deck.deal(2)

        if board:
            self.deck.stack_next(board)

        if self.ante:
            for p in seated:
                self._commit(p, min(self.ante, p.stack), street=False)

        order = self._seats_from(self.button, include_start=False)
        if len(seated) == 2:
            # Heads-up: the button is the small blind and acts first preflop.
            sb_seat, bb_seat = self.button, order[0]
        else:
            sb_seat, bb_seat = order[0], order[1]

        self._commit(self.players[sb_seat], min(self.sb, self.players[sb_seat].stack))
        self._commit(self.players[bb_seat], min(self.bb, self.players[bb_seat].stack))
        self.current_bet = max(p.street_committed for p in seated)
        self.min_raise = self.bb

        first = self._next_actor_after(bb_seat)
        self._actor = first
        if self._round_complete():
            self._advance_street()

    def _seats_from(self, seat: int, include_start: bool = True) -> List[int]:
        """Seats with chips, clockwise from ``seat``."""
        n = len(self.players)
        out = []
        rng = range(0, n) if include_start else range(1, n + 1)
        for k in rng:
            s = (seat + k) % n
            if self.players[s].in_hand:
                out.append(s)
        return out

    # -- chip movement -------------------------------------------------------

    def _commit(self, p: PlayerState, amount: int, street: bool = True) -> int:
        """Move chips from a stack into the pot.

        ``street=False`` is used for antes, which count toward side-pot
        eligibility but are not a bet anyone has to match.
        """
        amount = min(amount, p.stack)
        p.stack -= amount
        if street:
            p.street_committed += amount
        p.total_committed += amount
        self.pot += amount
        if p.stack == 0:
            p.all_in = True
        return amount

    # -- turn order ----------------------------------------------------------

    def _next_actor_after(self, seat: int) -> Optional[int]:
        n = len(self.players)
        for k in range(1, n + 1):
            s = (seat + k) % n
            p = self.players[s]
            if p.can_act and (not p.acted or p.street_committed < self.current_bet):
                return s
        return None

    def _round_complete(self) -> bool:
        live = [p for p in self.players if p.live]
        if len(live) <= 1:
            return True
        actionable = [p for p in self.players if p.can_act]
        if not actionable:
            return True
        if len(actionable) == 1 and actionable[0].street_committed >= self.current_bet:
            # Only one player can still act and they owe nothing: betting is
            # closed (there is nobody left to bet into).
            return True
        for p in actionable:
            if not p.acted or p.street_committed < self.current_bet:
                return False
        return True

    def current_actor(self) -> Optional[int]:
        return self._actor

    # -- legality ------------------------------------------------------------

    def legal_actions(self, seat: Optional[int] = None) -> List[LegalAction]:
        seat = self._actor if seat is None else seat
        if seat is None:
            return []
        p = self.players[seat]
        to_call = self.current_bet - p.street_committed
        max_to = p.street_committed + p.stack
        out: List[LegalAction] = []
        if to_call > 0:
            out.append(LegalAction(ActionType.FOLD))
            out.append(LegalAction(ActionType.CALL, min(to_call, p.stack), min(to_call, p.stack)))
        else:
            out.append(LegalAction(ActionType.CHECK))
        if p.can_raise and max_to > self.current_bet:
            kind = ActionType.BET if self.current_bet == 0 else ActionType.RAISE
            min_to = min(self.current_bet + self.min_raise, max_to)
            out.append(LegalAction(kind, min_to, max_to))
        return out

    def observation(self, seat: Optional[int] = None) -> Observation:
        seat = self._actor if seat is None else seat
        if seat is None:
            raise ValueError("no player to act")
        p = self.players[seat]
        return Observation(
            seat=seat,
            name=p.name,
            hole=list(p.hole),
            board=list(self.board),
            street=self.street,
            pot=self.pot,
            stacks=[q.stack for q in self.players],
            street_committed=[q.street_committed for q in self.players],
            total_committed=[q.total_committed for q in self.players],
            folded=[q.folded for q in self.players],
            all_in=[q.all_in for q in self.players],
            in_hand=[q.in_hand for q in self.players],
            names=[q.name for q in self.players],
            button=self.button,
            big_blind=self.bb,
            small_blind=self.sb,
            current_bet=self.current_bet,
            min_raise=self.min_raise,
            to_call=max(0, self.current_bet - p.street_committed),
            legal=self.legal_actions(seat),
            history=list(self.history),
            hand_number=self.hand_number,
        )

    # -- the move ------------------------------------------------------------

    def apply(self, action: Action) -> ActionRecord:
        if self.hand_over:
            raise ValueError("hand is over")
        seat = self._actor
        if seat is None:
            raise ValueError("no player to act")
        p = self.players[seat]
        to_call = self.current_bet - p.street_committed
        legal = {la.type: la for la in self.legal_actions(seat)}
        if action.type not in legal:
            raise ValueError(
                f"illegal action {action} for {p.name}; legal: {sorted(t.name for t in legal)}"
            )

        pot_before = self.pot
        stack_before = p.stack
        moved = 0
        to_amount = p.street_committed

        if action.type == ActionType.FOLD:
            p.folded = True
        elif action.type == ActionType.CHECK:
            pass
        elif action.type == ActionType.CALL:
            moved = self._commit(p, to_call)
            to_amount = p.street_committed
        else:  # BET / RAISE
            la = legal[action.type]
            target = int(action.amount)
            if target < la.min_amount or target > la.max_amount:
                raise ValueError(
                    f"{action.type.name} to {target} out of range "
                    f"[{la.min_amount}, {la.max_amount}]"
                )
            raise_size = target - self.current_bet
            moved = self._commit(p, target - p.street_committed)
            to_amount = p.street_committed
            full_raise = raise_size >= self.min_raise
            prev_bet = self.current_bet
            self.current_bet = max(self.current_bet, p.street_committed)
            if full_raise:
                self.min_raise = raise_size
                for q in self.players:
                    if q.seat != seat and q.can_act:
                        q.acted = False
                        q.can_raise = True
            else:
                # Short all-in: players who already acted owe a call, not a raise.
                for q in self.players:
                    if q.seat != seat and q.can_act and self.current_bet > prev_bet:
                        if q.acted:
                            q.acted = False
                            q.can_raise = False

        p.acted = True
        record = ActionRecord(
            street=self.street,
            seat=seat,
            name=p.name,
            action=action.type,
            amount=moved,
            to_amount=to_amount,
            pot_before=pot_before,
            to_call=to_call,
            stack_before=stack_before,
        )
        self.history.append(record)

        live = [q for q in self.players if q.live]
        if len(live) <= 1:
            self._finish()
            return record

        if self._round_complete():
            self._advance_street()
        else:
            self._actor = self._next_actor_after(seat)
            if self._actor is None:
                self._advance_street()
        return record

    # -- street transitions --------------------------------------------------

    def _advance_street(self) -> None:
        while True:
            if self.street == RIVER:
                self._finish()
                return
            self.street += 1
            if self.street == FLOP:
                self.board.extend(self.deck.deal(3))
            else:
                self.board.append(self.deck.deal_one())

            for p in self.players:
                p.street_committed = 0
                p.acted = False
                p.can_raise = True
            self.current_bet = 0
            self.min_raise = self.bb

            actionable = [p for p in self.players if p.can_act]
            if len(actionable) <= 1:
                # Everyone (or all but one) is all-in: run the board out.
                continue
            self._actor = self._next_actor_after(self.button)
            if self._actor is None:  # pragma: no cover - defensive
                continue
            return

    # -- payout --------------------------------------------------------------

    def _side_pots(self) -> List[Tuple[int, List[PlayerState]]]:
        contributors = [p for p in self.players if p.total_committed > 0]
        levels = sorted({p.total_committed for p in contributors})
        pots: List[Tuple[int, List[PlayerState]]] = []
        prev = 0
        for level in levels:
            amount = sum(
                min(p.total_committed, level) - min(p.total_committed, prev)
                for p in contributors
            )
            eligible = [p for p in contributors if not p.folded and p.total_committed >= level]
            if amount > 0:
                pots.append((amount, eligible))
            prev = level
        # Merge pots that no live player can win into the previous pot (dead money).
        merged: List[Tuple[int, List[PlayerState]]] = []
        carry = 0
        for amount, eligible in pots:
            if not eligible:
                carry += amount
                continue
            if merged and merged[-1][1] == eligible:
                merged[-1] = (merged[-1][0] + amount + carry, eligible)
            else:
                merged.append((amount + carry, eligible))
            carry = 0
        if carry and merged:
            merged[-1] = (merged[-1][0] + carry, merged[-1][1])
        return merged

    def _refund_uncalled(self) -> Optional[Tuple[int, int]]:
        """Return the part of the largest bet nobody matched.

        Without this the excess would be reported as a one-player side pot,
        which conserves chips but is not how a hand history reads.
        """
        contributions = sorted(
            ((p.total_committed, p.seat) for p in self.players if p.total_committed > 0),
            reverse=True,
        )
        if len(contributions) < 2:
            return None
        (top, seat), (second, _) = contributions[0], contributions[1]
        excess = top - second
        if excess <= 0:
            return None
        player = self.players[seat]
        player.stack += excess
        player.total_committed -= excess
        player.street_committed = max(0, player.street_committed - excess)
        self.pot -= excess
        if player.stack > 0:
            player.all_in = False
        return (seat, excess)

    def _finish(self) -> None:
        self.hand_over = True
        self.uncalled = self._refund_uncalled()
        live = [p for p in self.players if p.live]
        showdown = len(live) > 1
        if showdown:
            while len(self.board) < 5:
                self.board.append(self.deck.deal_one())
            self.street = SHOWDOWN

        scores: Dict[int, int] = {}
        if showdown:
            for p in live:
                scores[p.seat] = evaluate(p.hole + self.board)

        pot_results: List[PotResult] = []
        order = self._seats_from(self.button, include_start=False)
        for amount, eligible in self._side_pots():
            if len(eligible) == 1:
                winners = [eligible[0].seat]
                desc = "uncontested"
            else:
                best = max(scores[p.seat] for p in eligible)
                winners = [p.seat for p in eligible if scores[p.seat] == best]
                desc = describe(best)
            share, remainder = divmod(amount, len(winners))
            for seat in winners:
                self.players[seat].stack += share
            # Odd chips go to the first winner clockwise from the button.
            for seat in order:
                if remainder <= 0:
                    break
                if seat in winners:
                    self.players[seat].stack += 1
                    remainder -= 1
            pot_results.append(
                PotResult(amount=amount, winners=winners,
                          eligible=[p.seat for p in eligible],
                          per_winner=share, description=desc)
            )

        for p in self.players:
            p.net = p.stack - self._stacks_before.get(p.seat, p.stack)

        self.results = HandResult(
            hand_number=self.hand_number,
            button=self.button,
            board=list(self.board),
            hole={p.seat: list(p.hole) for p in self.players if p.in_hand},
            revealed=({p.seat: list(p.hole) for p in live} if showdown else {}),
            net={p.seat: p.net for p in self.players},
            pots=pot_results,
            uncalled=self.uncalled,
            history=list(self.history),
            showdown=showdown,
            stacks_before=dict(self._stacks_before),
            names={p.seat: p.name for p in self.players},
        )
        self._actor = None

    # -- helpers -------------------------------------------------------------

    def summary(self) -> str:  # pragma: no cover - debug aid
        lines = [f"board {cards_str(self.board)} pot {self.pot}"]
        for r in self.history:
            lines.append(f"  [{STREET_NAMES[r.street]}] {r.describe()}")
        if self.results:
            for pot in self.results.pots:
                who = ", ".join(self.results.names[s] for s in pot.winners)
                lines.append(f"  pot {pot.amount} -> {who} ({pot.description})")
        return "\n".join(lines)


def street_of(board: Sequence[int]) -> int:
    n = len(board)
    if n == 0:
        return PREFLOP
    if n == 3:
        return FLOP
    if n == 4:
        return TURN
    return RIVER

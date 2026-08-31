"""Multi-hand session driver: seats agents, rotates the button, tracks stacks."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from .engine import (
    Action,
    ActionRecord,
    HandResult,
    Observation,
    PlayerState,
    Table,
)


class Agent(Protocol):
    """Anything that can sit at the table."""

    name: str

    def act(self, obs: Observation) -> Action:  # pragma: no cover - protocol
        ...


class BaseAgent:
    """Convenience base with no-op hooks; bots override what they need."""

    def __init__(self, name: str):
        self.name = name
        self.seat: int = -1

    def act(self, obs: Observation) -> Action:  # pragma: no cover - abstract
        raise NotImplementedError

    def on_hand_start(self, seat: int, hand_number: int) -> None:
        self.seat = seat

    def on_action(self, record: ActionRecord, obs_public: Observation) -> None:
        """Called for every action at the table, including our own."""

    def on_hand_end(self, result: HandResult, seat: int) -> None:
        """Called once a hand is complete (``result.revealed`` is the legal view)."""

    def reset(self) -> None:
        """Forget any per-session state."""


@dataclass
class SessionStats:
    hands: int = 0
    net: Dict[int, int] = field(default_factory=dict)
    busts: Dict[int, int] = field(default_factory=dict)

    def bb_per_100(self, seat: int, bb: int) -> float:
        if not self.hands:
            return 0.0
        return 100.0 * self.net.get(seat, 0) / (bb * self.hands)


class Game:
    """A cash-game session: fixed blinds, optional auto-rebuy to the starting stack."""

    def __init__(
        self,
        agents: Sequence[Agent],
        starting_stack: int = 200,
        sb: int = 1,
        bb: int = 2,
        ante: int = 0,
        rng: Optional[random.Random] = None,
        auto_rebuy: bool = True,
        button: int = 0,
        on_hand_end: Optional[Callable[[HandResult], None]] = None,
    ):
        if len(agents) < 2:
            raise ValueError("need at least two agents")
        self.agents = list(agents)
        self.rng = rng or random.Random()
        self.sb, self.bb, self.ante = sb, bb, ante
        self.starting_stack = starting_stack
        self.auto_rebuy = auto_rebuy
        self.button = button
        self.hand_number = 0
        self.on_hand_end = on_hand_end
        self.players = [
            PlayerState(seat=i, name=getattr(a, "name", f"P{i}"), stack=starting_stack)
            for i, a in enumerate(self.agents)
        ]
        self.stats = SessionStats(net={i: 0 for i in range(len(agents))},
                                  busts={i: 0 for i in range(len(agents))})
        self.history: List[HandResult] = []

    # -- one hand ------------------------------------------------------------

    def _rebuy(self) -> None:
        for p in self.players:
            if p.stack <= 0:
                self.stats.busts[p.seat] = self.stats.busts.get(p.seat, 0) + 1
                if self.auto_rebuy:
                    p.stack = self.starting_stack

    def playable(self) -> bool:
        return sum(1 for p in self.players if p.stack > 0) >= 2

    def play_hand(self, keep_history: bool = True) -> HandResult:
        self._rebuy()
        if not self.playable():
            raise RuntimeError("not enough players with chips")
        while self.players[self.button].stack <= 0:
            self.button = (self.button + 1) % len(self.players)

        self.hand_number += 1
        table = Table(
            self.players,
            sb=self.sb,
            bb=self.bb,
            ante=self.ante,
            button=self.button,
            rng=self.rng,
            hand_number=self.hand_number,
        )
        for i, agent in enumerate(self.agents):
            hook = getattr(agent, "on_hand_start", None)
            if hook:
                hook(i, self.hand_number)
        table.start_hand()

        while not table.hand_over:
            seat = table.current_actor()
            obs = table.observation(seat)
            action = self.agents[seat].act(obs)
            record = table.apply(action)
            # Everyone else sees the same spot with the hole cards removed —
            # opponent models must learn from public information only.
            public = replace(obs, hole=[])
            for i, agent in enumerate(self.agents):
                hook = getattr(agent, "on_action", None)
                if hook:
                    hook(record, obs if i == seat else public)

        result = table.results
        assert result is not None
        for i, agent in enumerate(self.agents):
            hook = getattr(agent, "on_hand_end", None)
            if hook:
                hook(result, i)

        self.stats.hands += 1
        for seat, net in result.net.items():
            self.stats.net[seat] = self.stats.net.get(seat, 0) + net
        if keep_history:
            self.history.append(result)
        if self.on_hand_end:
            self.on_hand_end(result)

        self.button = (self.button + 1) % len(self.players)
        return result

    def run(self, hands: int, keep_history: bool = True) -> SessionStats:
        for _ in range(hands):
            if not self.playable() and not self.auto_rebuy:
                break
            self.play_hand(keep_history=keep_history)
        return self.stats

    # -- reporting -----------------------------------------------------------

    def leaderboard(self) -> List[tuple]:
        rows = [
            (self.agents[i].name, self.stats.net[i], self.stats.bb_per_100(i, self.bb))
            for i in range(len(self.agents))
        ]
        return sorted(rows, key=lambda r: -r[1])

    def report(self) -> str:
        width = max(len(a.name) for a in self.agents) + 2
        lines = [f"{self.stats.hands} hands, blinds {self.sb}/{self.bb}",
                 f"{'player':<{width}}{'net':>10}{'bb/100':>10}"]
        for name, net, bb100 in self.leaderboard():
            lines.append(f"{name:<{width}}{net:>10}{bb100:>10.1f}")
        return "\n".join(lines)

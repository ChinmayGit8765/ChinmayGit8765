"""The human seat: prompts a person for their action in the terminal."""

from __future__ import annotations

from typing import Callable, Optional

from .analysis.analyzer import Analyzer
from .engine import Action, ActionType, HandResult, Observation
from .equity import fast_equity
from .evaluator import describe, evaluate
from .game import BaseAgent
from .ui import Painter, format_legal


class QuitGame(Exception):
    """Raised when the player asks to leave the table."""


class HumanAgent(BaseAgent):
    def __init__(self, name: str, painter: Optional[Painter] = None,
                 analyzer: Optional[Analyzer] = None,
                 input_fn: Callable[[str], str] = input,
                 output_fn: Callable[[str], None] = print,
                 info_fn: Optional[Callable[[], str]] = None):
        super().__init__(name)
        self.painter = painter or Painter()
        self.analyzer = analyzer
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.info_fn = info_fn
        self.hands_played = 0

    def act(self, obs: Observation) -> Action:
        p = self.painter
        self.output_fn("")
        self.output_fn(p.table(obs, obs.seat))
        equity = fast_equity(obs.hole, obs.board, max(1, obs.live_opponents), iters=1200)
        made = ""
        if len(obs.board) >= 3:
            made = " · " + describe(evaluate(list(obs.hole) + list(obs.board)))
        self.output_fn(f"  your hand {p.cards(obs.hole)}  equity {equity:.0%}"
                       f" vs {max(1, obs.live_opponents)}{made}")
        if obs.to_call:
            self.output_fn(f"  to call {obs.to_call}   pot odds {obs.pot_odds:.0%}")

        while True:
            raw = self.input_fn(f"  {format_legal(obs)}\n  > ").strip().lower()
            if not raw:
                continue
            if raw in ("q", "quit", "exit"):
                raise QuitGame()
            if raw in ("?", "advice", "h", "help"):
                self.output_fn("  " + self._advice(obs, equity))
                continue
            if raw in ("i", "info", "stats"):
                self.output_fn(self.info_fn() if self.info_fn else "  no table info")
                continue
            action = self._parse(raw, obs)
            if action is not None:
                return action
            self.output_fn("  did not understand that — try f, c, r <amount>, or a")

    def _parse(self, raw: str, obs: Observation) -> Optional[Action]:
        legal = {la.type: la for la in obs.legal}
        head, _, rest = raw.partition(" ")
        rest = rest.strip()

        if head in ("f", "fold") and ActionType.FOLD in legal:
            return Action(ActionType.FOLD)
        if head in ("c", "call", "check", "k"):
            if ActionType.CHECK in legal:
                return Action(ActionType.CHECK)
            if ActionType.CALL in legal:
                return Action(ActionType.CALL, legal[ActionType.CALL].min_amount)
        raise_la = legal.get(ActionType.BET) or legal.get(ActionType.RAISE)
        if head in ("a", "allin", "all-in", "shove") and raise_la:
            return Action(raise_la.type, raise_la.max_amount)
        if head in ("r", "raise", "b", "bet") and raise_la:
            if not rest:
                return Action(raise_la.type, raise_la.min_amount)
            try:
                if rest.endswith("%"):
                    frac = float(rest[:-1]) / 100.0
                    amount = obs.current_bet + int(round(frac * (obs.pot + obs.to_call)))
                elif rest in ("pot", "p"):
                    amount = obs.current_bet + obs.pot + obs.to_call
                else:
                    amount = int(round(float(rest)))
            except ValueError:
                return None
            amount = max(raise_la.min_amount, min(amount, raise_la.max_amount))
            return Action(raise_la.type, amount)
        return None

    def _advice(self, obs: Observation, equity: float) -> str:
        if self.analyzer is None:
            self.analyzer = Analyzer()
        probs, values, spot, _ = self.analyzer.assess_spot(obs)
        from .ml.abstraction import ACTION_NAMES
        import numpy as np

        order = np.argsort(-probs)[:3]
        top = ", ".join(f"{ACTION_NAMES[i]} {probs[i]:.0%}" for i in order if probs[i] > 0.01)
        line = f"the model would play: {top}"
        finite = np.isfinite(values)
        if finite.any() and self.analyzer.model is not None:
            best = int(np.argmax(np.where(finite, values, -np.inf)))
            line += f" · best expected value: {ACTION_NAMES[best]} ({values[best]:+.2f}bb)"
        return line

    def on_hand_end(self, result: HandResult, seat: int) -> None:
        self.hands_played += 1
        p = self.painter
        net = result.net.get(seat, 0)
        lines = [p.rule("result")]
        if result.board:
            lines.append(f"  board {p.cards(result.board)}")
        for s, hole in sorted(result.revealed.items()):
            score = evaluate(list(hole) + result.board)
            lines.append(f"  {result.names[s]:<12}{p.cards(hole)}  {describe(score)}")
        for pot in result.pots:
            who = ", ".join(result.names[s] for s in pot.winners)
            lines.append(f"  pot {pot.amount} → {who} ({pot.description})")
        tag = p.paint(f"{net:+d}", "\033[32m" if net >= 0 else "\033[31m")
        lines.append(f"  you {tag}")
        self.output_fn("\n".join(lines))

"""Grade poker decisions — one hand, a whole session, or a spot from a photo.

The analyser answers three questions about every decision:

* *What would a strong player do here?*  — the pro model's policy.
* *What did this choice cost?*           — ``max Q - Q(chosen)``, in big blinds.
* *Is this a pattern?*                   — session aggregates by street and
  situation, cross-checked against the player's HUD stats.

It runs on hands played in this project and on hand histories imported from a
real site, and (via :mod:`holdem.vision`) on a photograph or screenshot of a
table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..cards import cards_str
from ..engine import HandResult, Observation, STREET_NAMES
from ..equity import fast_equity
from ..evaluator import describe, evaluate
from ..ml.abstraction import ACTION_NAMES, legal_mask, strength_percentile, to_abstract
from ..ml.features import encode
from ..ml.opponent import OpponentTracker
from .promodel import DEFAULT_PRO_MODEL, ProModel, load_pro_model
from .replay import DecisionPoint, replay_hand


@dataclass
class DecisionReport:
    hand_number: int
    street: int
    name: str
    hole: List[int]
    board: List[int]
    pot: int
    to_call: int
    chosen: int
    chosen_label: str
    best: int
    best_label: str
    pro_probs: np.ndarray
    action_values: np.ndarray
    ev_loss: float
    spot_value: float
    equity: float

    @property
    def is_mistake(self) -> bool:
        return self.ev_loss > 0.35

    def describe(self) -> str:
        board = cards_str(self.board) if self.board else "(preflop)"
        line = (f"[{STREET_NAMES[self.street]}] {cards_str(self.hole)} on {board} "
                f"pot {self.pot}, to call {self.to_call}: chose {self.chosen_label}")
        if self.is_mistake:
            line += (f" — costs {self.ev_loss:.2f}bb; "
                     f"{self.best_label} is better "
                     f"(pro plays it {self.pro_probs[self.best]:.0%} of the time)")
        else:
            line += f" — fine ({self.pro_probs[self.chosen]:.0%} pro agreement)"
        return line


@dataclass
class HandReport:
    hand_number: int
    player: str
    decisions: List[DecisionReport]
    net: int
    board: List[int]

    @property
    def total_ev_loss(self) -> float:
        return sum(d.ev_loss for d in self.decisions)

    @property
    def worst(self) -> Optional[DecisionReport]:
        return max(self.decisions, key=lambda d: d.ev_loss, default=None)

    def summary(self) -> str:
        lines = [f"hand #{self.hand_number} — {self.player} "
                 f"net {self.net:+d}, EV lost {self.total_ev_loss:.2f}bb"]
        lines += ["  " + d.describe() for d in self.decisions]
        return "\n".join(lines)


@dataclass
class SessionReport:
    player: str
    hands: int
    decisions: int
    net_bb: float
    ev_loss_total: float
    ev_loss_by_street: Dict[int, float]
    agreement: float
    stats: Dict[str, float]
    leaks: List[str]
    worst: List[DecisionReport]

    @property
    def ev_loss_per_100(self) -> float:
        return 100.0 * self.ev_loss_total / max(1, self.hands)

    def summary(self) -> str:
        lines = [
            f"session report — {self.player}",
            f"  {self.hands} hands, {self.decisions} decisions, "
            f"net {self.net_bb:+.1f}bb ({100 * self.net_bb / max(1, self.hands):+.1f} bb/100)",
            f"  EV lost {self.ev_loss_total:.1f}bb "
            f"({self.ev_loss_per_100:.1f}bb per 100 hands)",
            f"  agreement with the pro model: {self.agreement:.0%}",
            "  EV lost by street: " + ", ".join(
                f"{STREET_NAMES[s]} {v:.1f}bb" for s, v in sorted(self.ev_loss_by_street.items())
            ),
        ]
        if self.stats:
            lines.append(
                f"  VPIP {self.stats.get('vpip', 0):.0%}, PFR {self.stats.get('pfr', 0):.0%}, "
                f"AF {self.stats.get('aggression_factor', 0):.1f}, "
                f"folds to bets {self.stats.get('fold_to_bet', 0):.0%}, "
                f"WTSD {self.stats.get('wtsd', 0):.0%}"
            )
        if self.leaks:
            lines.append("  leaks:")
            lines += [f"    - {leak}" for leak in self.leaks]
        if self.worst:
            lines.append("  biggest mistakes:")
            lines += [f"    - {d.describe()}" for d in self.worst]
        return "\n".join(lines)


LEAK_RULES = [
    ("vpip", 0.36, "above", "playing too many hands preflop — tighten your opening range"),
    ("vpip", 0.14, "below", "playing too few hands — you are missing profitable spots"),
    ("pfr", 0.09, "below", "limping and calling too much preflop — raise or fold more often"),
    ("fold_to_bet", 0.60, "above", "folding too often to bets — you are easy to bluff"),
    ("fold_to_bet", 0.25, "below", "calling down too light — pay off strong hands less"),
    ("aggression_factor", 0.9, "below", "too passive after the flop — bet and raise your good hands"),
    ("aggression_factor", 4.0, "above", "over-aggressive after the flop — too many bluffs"),
    ("wtsd", 0.38, "above", "reaching showdown too often — fold marginal hands earlier"),
    ("three_bet", 0.02, "below", "almost never 3-betting — you are readable preflop"),
]


class Analyzer:
    def __init__(self, model: Optional[ProModel] = None,
                 path: Optional[str] = None, equity_iters: int = 400):
        if model is None:
            candidate = path or DEFAULT_PRO_MODEL
            model = load_pro_model(candidate) if os.path.exists(candidate) else None
        self.model = model
        self.equity_iters = equity_iters

    # -- one decision --------------------------------------------------------

    def assess_spot(self, obs: Observation) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """``(pro policy, action values, spot value, equity)`` for a live spot."""
        equity = fast_equity(obs.hole, obs.board, max(1, obs.live_opponents),
                             iters=self.equity_iters)
        mask = legal_mask(obs)
        if self.model is None:
            probs = mask / mask.sum()
            return probs, np.where(mask, 0.0, -np.inf), 0.0, equity
        features = encode(obs, equity_field=equity, equity_iters=self.equity_iters)
        probs, values, spot = self.model.assess(features, mask)
        return probs, values, spot, equity

    def analyse_decision(self, dp: DecisionPoint, hand_number: int = 0) -> DecisionReport:
        probs, values, spot, equity = self.assess_spot(dp.obs)
        chosen = to_abstract(dp.obs, dp.action)
        finite = np.where(np.isfinite(values), values, -np.inf)
        best = int(np.argmax(finite))
        ev_loss = float(max(0.0, finite[best] - finite[chosen])) if np.isfinite(finite[chosen]) else 0.0
        return DecisionReport(
            hand_number=hand_number, street=dp.street, name=dp.name,
            hole=list(dp.obs.hole), board=list(dp.obs.board), pot=dp.obs.pot,
            to_call=dp.obs.to_call, chosen=chosen, chosen_label=ACTION_NAMES[chosen],
            best=best, best_label=ACTION_NAMES[best], pro_probs=probs,
            action_values=finite, ev_loss=ev_loss, spot_value=spot, equity=equity,
        )

    # -- one hand ------------------------------------------------------------

    def analyse_hand(self, result: HandResult, player: str,
                     sb: int = 1, bb: int = 2) -> HandReport:
        decisions = []
        seat = next((s for s, n in result.names.items() if n == player), None)
        for dp in replay_hand(result, sb=sb, bb=bb):
            if dp.name != player or len(dp.obs.hole) != 2:
                continue
            decisions.append(self.analyse_decision(dp, result.hand_number))
        return HandReport(hand_number=result.hand_number, player=player,
                          decisions=decisions,
                          net=result.net.get(seat, 0) if seat is not None else 0,
                          board=list(result.board))

    # -- a session -----------------------------------------------------------

    def analyse_session(self, results: Sequence[HandResult], player: str,
                        sb: int = 1, bb: int = 2, top_mistakes: int = 5) -> SessionReport:
        tracker = OpponentTracker(learn=False)
        all_decisions: List[DecisionReport] = []
        net_bb = 0.0
        hands = 0

        for result in results:
            seat = next((s for s, n in result.names.items() if n == player), None)
            if seat is None:
                continue
            hands += 1
            net_bb += result.net.get(seat, 0) / bb
            for dp in replay_hand(result, sb=sb, bb=bb):
                tracker.on_action(dp.record, dp.obs)
                if dp.name == player and len(dp.obs.hole) == 2:
                    all_decisions.append(self.analyse_decision(dp, result.hand_number))
            tracker.on_hand_end(result)

        by_street: Dict[int, float] = {}
        for d in all_decisions:
            by_street[d.street] = by_street.get(d.street, 0.0) + d.ev_loss
        agreement = float(np.mean([d.pro_probs[d.chosen] for d in all_decisions])) \
            if all_decisions else 0.0

        model = tracker.models.get(player)
        stats = model.stats() if model else {}
        leaks = self.find_leaks(stats, all_decisions)
        worst = sorted(all_decisions, key=lambda d: -d.ev_loss)[:top_mistakes]
        worst = [d for d in worst if d.ev_loss > 0.15]

        return SessionReport(
            player=player, hands=hands, decisions=len(all_decisions), net_bb=net_bb,
            ev_loss_total=sum(d.ev_loss for d in all_decisions),
            ev_loss_by_street=by_street, agreement=agreement, stats=stats,
            leaks=leaks, worst=worst,
        )

    def find_leaks(self, stats: Dict[str, float],
                   decisions: Sequence[DecisionReport]) -> List[str]:
        out: List[str] = []
        for key, threshold, direction, message in LEAK_RULES:
            value = stats.get(key)
            if value is None or stats.get("hands", 0) < 25:
                continue
            if (direction == "above" and value > threshold) or \
               (direction == "below" and value < threshold):
                out.append(f"{message} ({key} {value:.2f})")

        # Situational leaks measured directly from the EV losses.
        folds = [d for d in decisions if d.chosen == 0]
        if folds:
            fold_loss = np.mean([d.ev_loss for d in folds])
            if fold_loss > 0.5:
                out.append(f"your folds are the expensive part "
                           f"({fold_loss:.2f}bb lost per fold on average)")
        bluffs = [d for d in decisions if d.chosen >= 2 and d.equity < 0.35]
        if len(bluffs) >= 8:
            loss = np.mean([d.ev_loss for d in bluffs])
            if loss > 0.6:
                out.append(f"bluffing too thin — {len(bluffs)} raises with under 35% equity "
                           f"cost {loss:.2f}bb each")
        thin = [d for d in decisions if d.chosen == 1 and d.equity > 0.7 and d.to_call == 0]
        if len(thin) >= 6:
            out.append(f"missing value — you checked {len(thin)} spots with over 70% equity")
        return out

    # -- from an image -------------------------------------------------------

    def analyse_image(self, path: str, hole: int = 2, reader=None,
                      opponents: int = 1) -> Dict[str, object]:
        """Read the cards out of a picture of a table and assess the spot."""
        from ..vision.detect import CardReader

        reader = reader or CardReader()
        detections = reader.read_file(path)
        cards = [d.card for d in detections]
        return self.assess_cards(cards[:hole], cards[hole:], opponents=opponents,
                                 detections=detections)

    def assess_cards(self, hole: Sequence[int], board: Sequence[int] = (),
                     opponents: int = 1, pot: int = 10, to_call: int = 0,
                     detections=None) -> Dict[str, object]:
        """Assess a spot given just the cards — the entry point for image input."""
        equity = fast_equity(list(hole), list(board), max(1, opponents), iters=2000)
        out: Dict[str, object] = {
            "hole": cards_str(hole),
            "board": cards_str(board),
            "opponents": opponents,
            "equity": equity,
            "pot_odds": to_call / (pot + to_call) if to_call else 0.0,
        }
        if len(hole) == 2 and len(board) >= 3:
            score = evaluate(list(hole) + list(board))
            out["made_hand"] = describe(score)
            out["strength_percentile"] = strength_percentile(hole, board)
        if detections is not None:
            out["detections"] = [(str(d), d.confidence) for d in detections]
            out["min_confidence"] = min((d.confidence for d in detections), default=0.0)
        out["advice"] = self._advice(equity, float(out.get("pot_odds", 0.0)),
                                     len(board), max(1, opponents))
        return out

    @staticmethod
    def _advice(equity: float, pot_odds: float, board_cards: int,
                opponents: int = 1) -> str:
        # Equity has to be read against the number of opponents: 49% against
        # five players is a monster, 49% heads-up is a coin flip.
        fair_share = 1.0 / (opponents + 1)
        if pot_odds <= 0:
            if equity > max(0.30, fair_share + 0.15):
                return "bet for value — you are well ahead of the range you face"
            if equity < fair_share * 0.85 and board_cards >= 3:
                return "check; the hand is not strong enough to build a pot"
            return "check or make a small probe bet"
        if equity > pot_odds + 0.18:
            return "raise — you have far more equity than the price demands"
        if equity > pot_odds + 0.03:
            return "call — the price is right"
        return "fold — you are not getting the odds"

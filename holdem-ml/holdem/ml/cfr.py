"""Monte-Carlo Counterfactual Regret Minimisation over the abstracted game.

This is the bot's *blueprint*: an unexploitable-by-construction baseline
strategy learned from nothing but self-play, with no hand-coded poker
knowledge.  External-sampling MCCFR walks the betting tree for one player at a
time, sampling chance (the deck) and the opponent's actions, and accumulates:

* ``regret_sum``   — how much better each action would have been in hindsight,
* ``strategy_sum`` — the visit-weighted strategy, whose average is what
  converges toward a Nash equilibrium of the abstraction.

Regret matching turns positive regrets into a probability distribution, which
is the whole learning rule.  The resulting table both plays directly (see
:class:`holdem.bots.blueprint.BlueprintBot`) and supplies training targets for
the neural policy.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..cards import Deck
from ..evaluator import evaluate
from .abstraction import (
    A_ALLIN,
    A_CALL,
    A_FOLD,
    A_RAISE_66,
    A_RAISE_200,
    NUM_ACTIONS,
    RAISE_FRACTIONS,
    info_bucket,
)

MAX_RAISES_PER_STREET = 3

# CFR explores a narrower sizing menu than the policy net: three raise sizes is
# enough to express the strategy while keeping the tree searchable.
CFR_RAISE_MENU = (A_RAISE_66, A_RAISE_200)

# Bucket edges for the betting-history abstraction.
_POT_EDGES = (0.05, 0.12, 0.25, 0.45, 0.7, 0.9)
_CALL_EDGES = (0.15, 0.28, 0.45)


@dataclass
class HUState:
    """Heads-up no-limit state, abstracted actions but real cards."""

    hole: List[List[int]]
    board: List[int]
    deck: List[int]
    stack: List[int]
    committed: List[int]          # chips in the pot this hand, per player
    street_committed: List[int]
    street: int = 0
    to_act: int = 0
    folded: int = -1
    min_raise: int = 2
    bb: int = 2
    raises: int = 0
    history: List[List[int]] = field(default_factory=lambda: [[], [], [], []])
    acted: List[bool] = field(default_factory=lambda: [False, False])
    # Card buckets are expensive; cache per player and invalidate on each street.
    bucket_cache: List[Optional[str]] = field(default_factory=lambda: [None, None])

    def clone(self) -> "HUState":
        return HUState(
            hole=[list(self.hole[0]), list(self.hole[1])],
            board=list(self.board),
            deck=self.deck,  # shared: chance is fixed for the iteration
            stack=list(self.stack),
            committed=list(self.committed),
            street_committed=list(self.street_committed),
            street=self.street,
            to_act=self.to_act,
            folded=self.folded,
            min_raise=self.min_raise,
            bb=self.bb,
            raises=self.raises,
            history=[list(h) for h in self.history],
            acted=list(self.acted),
            bucket_cache=list(self.bucket_cache),
        )

    @property
    def pot(self) -> int:
        return self.committed[0] + self.committed[1]

    @property
    def current_bet(self) -> int:
        return max(self.street_committed)

    def to_call(self, p: int) -> int:
        return self.current_bet - self.street_committed[p]

    def all_in(self, p: int) -> bool:
        return self.stack[p] <= 0

    def terminal(self) -> bool:
        return self.folded >= 0 or self.street > 3

    def bucket(self, p: int) -> str:
        cached = self.bucket_cache[p]
        if cached is None:
            cached = info_bucket(self.hole[p], self.board)
            self.bucket_cache[p] = cached
        return cached

    def infoset(self, p: int) -> str:
        """Bucketed information set.

        Rather than the literal action sequence (which explodes into millions
        of barely-visited nodes), the betting history is summarised by what
        actually drives the decision: how big the pot is relative to the
        effective stack, how much of the final pot a call costs, how many
        raises have gone in this street, and position.
        """
        effective = min(self.stack[0] + self.committed[0], self.stack[1] + self.committed[1])
        pot_ratio = self.pot / max(1.0, 2.0 * effective)
        pot_lvl = sum(1 for e in _POT_EDGES if pot_ratio > e)
        to_call = self.to_call(p)
        call_lvl = 0
        if to_call > 0:
            frac = to_call / (self.pot + to_call)
            call_lvl = 1 + sum(1 for e in _CALL_EDGES if frac > e)
        return f"{self.bucket(p)}|{pot_lvl}{call_lvl}{min(self.raises, 3)}{p}"


def new_hand(deck_cards: Sequence[int], stack: int = 200, sb: int = 1, bb: int = 2) -> HUState:
    """Deal a heads-up hand.  Player 0 is the button/small blind."""
    cards = list(deck_cards)
    st = HUState(
        hole=[cards[0:2], cards[2:4]],
        board=[],
        deck=cards[4:],
        stack=[stack - sb, stack - bb],
        committed=[sb, bb],
        street_committed=[sb, bb],
        min_raise=bb,
        bb=bb,
        to_act=0,
    )
    return st


def legal_actions(st: HUState) -> List[int]:
    p = st.to_act
    out = [A_CALL]
    to_call = st.to_call(p)
    if to_call > 0:
        out.insert(0, A_FOLD)
    if st.raises < MAX_RAISES_PER_STREET and st.stack[p] > to_call and not st.all_in(1 - p):
        max_to = st.street_committed[p] + st.stack[p]
        seen = set()
        for a in CFR_RAISE_MENU:
            frac = RAISE_FRACTIONS[a]
            target = st.current_bet + max(1, int(round(frac * (st.pot + to_call))))
            target = max(target, st.current_bet + st.min_raise)
            if target < max_to and target not in seen:
                seen.add(target)
                out.append(a)
        out.append(A_ALLIN)
    return out


def apply_action(st: HUState, action: int) -> HUState:
    nxt = st.clone()
    p = nxt.to_act
    to_call = nxt.to_call(p)

    if action == A_FOLD:
        nxt.folded = p
        nxt.history[nxt.street].append(action)
        return nxt

    if action == A_CALL:
        pay = min(to_call, nxt.stack[p])
        nxt.stack[p] -= pay
        nxt.committed[p] += pay
        nxt.street_committed[p] += pay
    else:
        max_to = nxt.street_committed[p] + nxt.stack[p]
        if action == A_ALLIN:
            target = max_to
        else:
            frac = RAISE_FRACTIONS[action]
            target = nxt.current_bet + max(1, int(round(frac * (nxt.pot + to_call))))
            target = min(max(target, nxt.current_bet + nxt.min_raise), max_to)
        raise_size = target - nxt.current_bet
        pay = target - nxt.street_committed[p]
        nxt.stack[p] -= pay
        nxt.committed[p] += pay
        nxt.street_committed[p] += pay
        if raise_size >= nxt.min_raise:
            nxt.min_raise = raise_size
        nxt.raises += 1
        nxt.acted[1 - p] = False

    nxt.acted[p] = True
    nxt.history[nxt.street].append(action)

    if _street_done(nxt):
        _advance(nxt)
    else:
        nxt.to_act = 1 - p
    return nxt


def _street_done(st: HUState) -> bool:
    if st.folded >= 0:
        return True
    if st.all_in(0) and st.all_in(1):
        return True
    for p in (0, 1):
        if st.all_in(p):
            continue
        if not st.acted[p] or st.street_committed[p] < st.current_bet:
            return False
    return True


def _advance(st: HUState) -> None:
    while True:
        if st.street >= 3:
            st.street = 4
            return
        st.street += 1
        need = 3 if st.street == 1 else 1
        st.board.extend(st.deck[:need])
        st.deck = st.deck[need:]
        st.street_committed = [0, 0]
        st.acted = [False, False]
        st.bucket_cache = [None, None]
        st.raises = 0
        st.min_raise = st.bb
        if st.all_in(0) or st.all_in(1):
            continue  # run the board out
        st.to_act = 1  # out of position (big blind) acts first postflop
        return


def payoff(st: HUState, player: int) -> float:
    """Chips won by ``player`` relative to the start of the hand, in big blinds."""
    if st.folded >= 0:
        winner = 1 - st.folded
        value = st.committed[1 - winner] if winner == player else -st.committed[player]
        return value / st.bb
    board = list(st.board)
    deck = list(st.deck)
    while len(board) < 5:
        board.append(deck.pop(0))
    s0 = evaluate(st.hole[0] + board)
    s1 = evaluate(st.hole[1] + board)
    if s0 == s1:
        return 0.0
    winner = 0 if s0 > s1 else 1
    value = st.committed[1 - winner] if winner == player else -st.committed[player]
    return value / st.bb


class Blueprint:
    """Regret and average-strategy tables, plus load/save."""

    def __init__(self) -> None:
        self.regret: Dict[str, np.ndarray] = {}
        self.strategy_sum: Dict[str, np.ndarray] = {}
        self.iterations = 0

    def strategy(self, key: str, legal: Sequence[int]) -> np.ndarray:
        """Regret matching: positive regrets, normalised; uniform if none."""
        regrets = self.regret.get(key)
        probs = np.zeros(NUM_ACTIONS, dtype=np.float64)
        if regrets is None:
            probs[list(legal)] = 1.0 / len(legal)
            return probs
        pos = np.maximum(regrets[list(legal)], 0.0)
        total = pos.sum()
        if total <= 0:
            probs[list(legal)] = 1.0 / len(legal)
        else:
            probs[list(legal)] = pos / total
        return probs

    def average_strategy(self, key: str, legal: Sequence[int]) -> np.ndarray:
        s = self.strategy_sum.get(key)
        probs = np.zeros(NUM_ACTIONS, dtype=np.float64)
        if s is None:
            probs[list(legal)] = 1.0 / len(legal)
            return probs
        vals = s[list(legal)]
        total = vals.sum()
        if total <= 0:
            probs[list(legal)] = 1.0 / len(legal)
        else:
            probs[list(legal)] = vals / total
        return probs

    def __len__(self) -> int:
        return len(self.strategy_sum)

    def merge_(self, regret_delta: Dict[str, np.ndarray],
               strategy_delta: Dict[str, np.ndarray]) -> None:
        """Add another worker's regret/strategy deltas into this table."""
        for key, delta in regret_delta.items():
            cur = self.regret.get(key)
            if cur is None:
                self.regret[key] = delta.astype(np.float64)
            else:
                cur += delta
        for key, delta in strategy_delta.items():
            cur = self.strategy_sum.get(key)
            if cur is None:
                self.strategy_sum[key] = delta.astype(np.float64)
            else:
                cur += delta

    def discount_regrets(self, factor: float) -> None:
        for r in self.regret.values():
            r *= factor

    # -- persistence ---------------------------------------------------------

    def save(self, path: str, min_visits: float = 0.5) -> int:
        keys, rows = [], []
        for key, s in self.strategy_sum.items():
            if s.sum() >= min_visits:
                keys.append(key)
                rows.append(s.astype(np.float32))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            keys=np.array(keys),
            strategy=np.array(rows, dtype=np.float32) if rows else np.zeros((0, NUM_ACTIONS), np.float32),
            iterations=np.array([self.iterations]),
        )
        return len(keys)

    @classmethod
    def load(cls, path: str) -> "Blueprint":
        bp = cls()
        with np.load(path, allow_pickle=False) as data:
            keys = data["keys"]
            strategy = data["strategy"]
            bp.iterations = int(data["iterations"][0])
        for key, row in zip(keys, strategy):
            bp.strategy_sum[str(key)] = row.astype(np.float64)
        return bp


class MCCFRTrainer:
    """External-sampling MCCFR with linear-CFR style discounting."""

    def __init__(self, blueprint: Optional[Blueprint] = None, stack: int = 200,
                 sb: int = 1, bb: int = 2, rng: Optional[random.Random] = None,
                 discount: bool = True, base_weight: float = 1.0):
        self.bp = blueprint if blueprint is not None else Blueprint()
        self.stack, self.sb, self.bb = stack, sb, bb
        self.rng = rng or random.Random()
        self.discount = discount
        # Workers in a parallel run share a constant averaging weight per round,
        # which reproduces linear-CFR averaging without needing a global clock.
        self.base_weight = base_weight

    def _traverse(self, st: HUState, traverser: int, weight: float) -> float:
        if st.terminal():
            return payoff(st, traverser)

        p = st.to_act
        legal = legal_actions(st)
        if len(legal) == 1:
            return self._traverse(apply_action(st, legal[0]), traverser, weight)

        key = st.infoset(p)
        sigma = self.bp.strategy(key, legal)

        if p == traverser:
            utils = np.zeros(NUM_ACTIONS, dtype=np.float64)
            node_util = 0.0
            for a in legal:
                utils[a] = self._traverse(apply_action(st, a), traverser, weight)
                node_util += sigma[a] * utils[a]
            regrets = self.bp.regret.setdefault(key, np.zeros(NUM_ACTIONS))
            for a in legal:
                regrets[a] += utils[a] - node_util
            return node_util

        # Opponent node: sample one action, bank the strategy for the average.
        acc = self.bp.strategy_sum.setdefault(key, np.zeros(NUM_ACTIONS))
        acc += weight * sigma
        probs = sigma[legal]
        probs = probs / probs.sum()
        a = int(self.rng.choices(legal, weights=probs, k=1)[0])
        return self._traverse(apply_action(st, a), traverser, weight)

    def iterate(self, iterations: int, log_every: int = 0) -> Blueprint:
        for i in range(iterations):
            self.bp.iterations += 1
            t = self.bp.iterations
            weight = t if self.discount else self.base_weight
            deck = Deck(self.rng)
            cards = deck.deal(9)
            for traverser in (0, 1):
                st = new_hand(cards, self.stack, self.sb, self.bb)
                self._traverse(st, traverser, weight)
            if self.discount:
                # Linear CFR: fade early, noisy regrets.
                factor = t / (t + 1.0)
                if t % 64 == 0:
                    for r in self.bp.regret.values():
                        r *= factor ** 64
            if log_every and (i + 1) % log_every == 0:
                print(f"  iter {i + 1}/{iterations}  infosets={len(self.bp.strategy_sum)}",
                      flush=True)
        return self.bp


# --- bridging the blueprint back to real games ------------------------------

def observation_infoset(obs) -> str:
    """Build the CFR info-set key for a live :class:`~holdem.engine.Observation`.

    The blueprint was solved heads-up, so a multiway table is mapped onto it by
    treating "in position against the field" as the button seat.
    """
    from .features import _relative_position
    from .abstraction import info_bucket as _bucket

    bucket = _bucket(obs.hole, obs.board)
    # CFR measures the effective stack as chips behind *plus* chips already in
    # for the hand (i.e. the starting stack).  The live game has to match that
    # or the same situation lands in a different pot bucket.
    stacks = [obs.stacks[i] + obs.total_committed[i] for i in range(obs.num_players)
              if obs.in_hand[i] and not obs.folded[i]]
    effective = max(1, min(stacks) if stacks else obs.my_stack)
    pot_ratio = obs.pot / max(1.0, 2.0 * effective)
    pot_lvl = sum(1 for e in _POT_EDGES if pot_ratio > e)
    call_lvl = 0
    if obs.to_call > 0:
        frac = obs.to_call / (obs.pot + obs.to_call)
        call_lvl = 1 + sum(1 for e in _CALL_EDGES if frac > e)
    raises = 0
    for rec in obs.history:
        if rec.street == obs.street and rec.action.name in ("BET", "RAISE"):
            raises += 1
    seat_role = 0 if _relative_position(obs) > 0.5 else 1
    return f"{bucket}|{pot_lvl}{call_lvl}{min(raises, 3)}{seat_role}"

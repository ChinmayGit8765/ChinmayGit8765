"""Replay a finished hand and recover the decision each player faced.

A :class:`~holdem.engine.HandResult` stores who did what, but not the
:class:`~holdem.engine.Observation` in front of them at the time — and that is
exactly what the analyser and the skill tracker need in order to ask "what
would a strong policy have done here?".  Re-running the hand through the same
engine with the cards pinned reconstructs it exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from ..engine import (
    Action,
    ActionRecord,
    ActionType,
    HandResult,
    Observation,
    PlayerState,
    Table,
)


@dataclass
class DecisionPoint:
    """One decision, with the full context that produced it."""

    obs: Observation
    action: Action
    record: ActionRecord
    seat: int
    name: str
    street: int


def replay_hand(result: HandResult, sb: int = 1, bb: int = 2,
                ante: int = 0) -> List[DecisionPoint]:
    """Step a recorded hand back through the engine, one decision at a time."""
    seats = sorted(result.stacks_before)
    players = [
        PlayerState(seat=s, name=result.names.get(s, f"P{s}"),
                    stack=result.stacks_before[s])
        for s in seats
    ]
    table = Table(players, sb=sb, bb=bb, ante=ante, button=result.button,
                  rng=random.Random(0), hand_number=result.hand_number)
    table.start_hand(hole_cards=result.hole, board=result.board or None)

    out: List[DecisionPoint] = []
    for record in result.history:
        if table.hand_over:
            break
        seat = table.current_actor()
        if seat is None or seat != record.seat:
            # The recorded line and the replay have diverged; stop rather than
            # silently producing wrong context.
            break
        obs = table.observation(seat)
        action = _record_to_action(record)
        out.append(DecisionPoint(obs=obs, action=action, record=record,
                                 seat=seat, name=record.name, street=record.street))
        table.apply(action)
    return out


def _record_to_action(record: ActionRecord) -> Action:
    if record.action in (ActionType.BET, ActionType.RAISE):
        return Action(record.action, record.to_amount)
    return Action(record.action)


def decisions_by(result: HandResult, name: str, sb: int = 1, bb: int = 2) -> List[DecisionPoint]:
    return [d for d in replay_hand(result, sb, bb) if d.name == name]

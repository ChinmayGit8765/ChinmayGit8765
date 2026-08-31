"""Turn finished hands — from anywhere — into training samples.

The same function handles hands this project generated and hand histories
imported from a real poker site, because both end up as
:class:`~holdem.engine.HandResult` objects that :func:`replay_hand` can step
through.  That is what makes "train the analyser on real pro hands" a matter of
pointing it at a directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..engine import HandResult
from ..ml.abstraction import NUM_ACTIONS, legal_mask, to_abstract
from ..ml.features import NUM_FEATURES, encode
from .replay import replay_hand


@dataclass
class Sample:
    features: np.ndarray
    mask: np.ndarray
    action: int
    ret: float          # chips won by this player in this hand, in big blinds
    street: int
    name: str


def samples_from_result(result: HandResult, sb: int = 1, bb: int = 2,
                        players: Optional[Sequence[str]] = None,
                        equity_iters: int = 250,
                        known_cards_only: bool = True) -> List[Sample]:
    out: List[Sample] = []
    wanted = set(players) if players else None
    for dp in replay_hand(result, sb=sb, bb=bb):
        if wanted is not None and dp.name not in wanted:
            continue
        if known_cards_only and len(dp.obs.hole) != 2:
            continue
        mask = legal_mask(dp.obs)
        action = to_abstract(dp.obs, dp.action)
        if not mask[action]:  # pragma: no cover - defensive
            continue
        out.append(Sample(
            features=encode(dp.obs, equity_iters=equity_iters),
            mask=mask,
            action=action,
            ret=result.net.get(dp.seat, 0) / max(1.0, bb),
            street=dp.street,
            name=dp.name,
        ))
    return out


def samples_from_results(results: Iterable[HandResult], sb: int = 1, bb: int = 2,
                         players: Optional[Sequence[str]] = None,
                         equity_iters: int = 250) -> List[Sample]:
    out: List[Sample] = []
    for result in results:
        out.extend(samples_from_result(result, sb, bb, players, equity_iters))
    return out


def samples_from_histories(path: str, players: Optional[Sequence[str]] = None,
                           equity_iters: int = 250) -> List[Sample]:
    """Load real hand-history files from ``path`` and turn them into samples."""
    from .handhistory import parse_directory, parse_file, to_hand_result

    hands = parse_directory(path) if os.path.isdir(path) else parse_file(path)
    out: List[Sample] = []
    for parsed in hands:
        result = to_hand_result(parsed)
        out.extend(samples_from_result(result, parsed.small_blind, parsed.big_blind,
                                       players, equity_iters))
    return out


def stack_samples(samples: Sequence[Sample]):
    X = np.stack([s.features for s in samples]).astype(np.float32)
    M = np.stack([s.mask for s in samples])
    A = np.array([s.action for s in samples], dtype=np.int64)
    R = np.array([s.ret for s in samples], dtype=np.float32)
    return X, M, A, R


def save_corpus(path: str, samples: Sequence[Sample]) -> None:
    X, M, A, R = stack_samples(samples)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, X=X, M=M, A=A, R=R,
                        street=np.array([s.street for s in samples], dtype=np.int8))


def load_corpus(path: str):
    with np.load(path, allow_pickle=False) as data:
        return data["X"], data["M"], data["A"], data["R"], data["street"]

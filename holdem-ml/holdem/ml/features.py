"""Turn a poker situation into a fixed-length feature vector.

One encoder is shared by everything that learns: the neural policy, the
opponent model and the pro-game analyser.  Keeping a single named feature list
means a leak the analyser reports ("you fold too much when the pot odds are
good") refers to exactly the number the bot is looking at.

Every feature is scaled into roughly ``[0, 1]`` (or ``[-1, 1]``) so a plain MLP
trains without input normalisation bookkeeping.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..cards import rank_of, suit_of
from ..engine import ActionType, Observation
from ..equity import fast_equity
from .abstraction import draw_class, strength_percentile

FEATURE_NAMES: List[str] = [
    # hand
    "equity_vs_field", "equity_heads_up", "made_hand_percentile",
    "draw_flush", "draw_oesd", "draw_gutshot",
    "hole_high_rank", "hole_low_rank", "hole_suited", "hole_pair", "hole_gap",
    # board
    "board_count", "board_paired", "board_trips", "board_three_suited",
    "board_four_suited", "board_connected", "board_high_rank", "board_low_rank",
    # street
    "street_preflop", "street_flop", "street_turn", "street_river",
    # money
    "pot_log", "pot_vs_stacks", "to_call_log", "pot_odds", "spr",
    "my_stack_bb", "effective_stack_bb", "committed_fraction",
    # table
    "live_opponents", "table_size", "relative_position", "is_button", "is_blind",
    "in_position",
    # action context
    "facing_bet", "raises_this_street", "callers_this_street", "last_was_raise",
    "i_am_pf_aggressor", "checked_to_me", "bet_faced_vs_pot",
    # opponent model
    "opp_vpip", "opp_pfr", "opp_aggression", "opp_fold_to_bet", "opp_wtsd",
    "opp_confidence",
]
NUM_FEATURES = len(FEATURE_NAMES)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _board_texture(board: Sequence[int]) -> Dict[str, float]:
    if not board:
        return {"paired": 0.0, "trips": 0.0, "three_suited": 0.0, "four_suited": 0.0,
                "connected": 0.0, "high": 0.0, "low": 0.0}
    ranks = [rank_of(c) for c in board]
    suits = [suit_of(c) for c in board]
    counts: Dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    suit_counts: Dict[int, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    mask = 0
    for r in ranks:
        mask |= 1 << r
    connected = 0.0
    for high in range(12, 2, -1):
        window = (mask >> max(0, high - 4)) & 0b11111
        if bin(window).count("1") >= 3:
            connected = 1.0
            break
    top_suit = max(suit_counts.values())
    return {
        "paired": 1.0 if max(counts.values()) >= 2 else 0.0,
        "trips": 1.0 if max(counts.values()) >= 3 else 0.0,
        "three_suited": 1.0 if top_suit >= 3 else 0.0,
        "four_suited": 1.0 if top_suit >= 4 else 0.0,
        "connected": connected,
        "high": max(ranks) / 12.0,
        "low": min(ranks) / 12.0,
    }


def street_context(obs: Observation) -> Dict[str, float]:
    """Aggression/limp counts for the current street, read off the history."""
    raises = 0
    callers = 0
    last_was_raise = 0.0
    for rec in obs.history:
        if rec.street != obs.street:
            continue
        if rec.action in (ActionType.BET, ActionType.RAISE):
            raises += 1
            last_was_raise = 1.0
        elif rec.action == ActionType.CALL:
            callers += 1
            last_was_raise = 0.0
        elif rec.action == ActionType.CHECK:
            last_was_raise = 0.0
    return {"raises": float(raises), "callers": float(callers),
            "last_was_raise": last_was_raise}


def preflop_aggressor(obs: Observation) -> Optional[int]:
    seat = None
    for rec in obs.history:
        if rec.street != 0:
            continue
        if rec.action in (ActionType.BET, ActionType.RAISE):
            seat = rec.seat
    return seat


def last_aggressor(obs: Observation) -> Optional[int]:
    seat = None
    for rec in obs.history:
        if rec.street == obs.street and rec.action in (ActionType.BET, ActionType.RAISE):
            seat = rec.seat
    return seat


def _relative_position(obs: Observation) -> float:
    """0.0 = first to act on this street, 1.0 = last."""
    live = [i for i in range(obs.num_players)
            if obs.in_hand[i] and not obs.folded[i] and not obs.all_in[i]]
    if len(live) <= 1:
        return 1.0
    start = (obs.button + 1) % obs.num_players if obs.street > 0 else \
        (obs.button + 3) % obs.num_players
    order = []
    for k in range(obs.num_players):
        s = (start + k) % obs.num_players
        if s in live:
            order.append(s)
    if obs.seat not in order:
        return 1.0
    return order.index(obs.seat) / max(1, len(order) - 1)


def encode(
    obs: Observation,
    equity_field: Optional[float] = None,
    equity_heads_up: Optional[float] = None,
    opponent_stats: Optional[Dict[str, float]] = None,
    equity_iters: int = 400,
) -> np.ndarray:
    """Encode ``obs`` into :data:`NUM_FEATURES` floats.

    ``equity_*`` may be supplied by a caller that caches roll-outs (the bots do
    this once per street); otherwise they are computed here.
    """
    bb = max(1, obs.big_blind)
    opponents = max(1, obs.live_opponents)
    if equity_field is None:
        equity_field = fast_equity(obs.hole, obs.board, opponents, iters=equity_iters)
    if equity_heads_up is None:
        equity_heads_up = (equity_field if opponents == 1
                           else fast_equity(obs.hole, obs.board, 1, iters=equity_iters))

    ranks = sorted((rank_of(c) for c in obs.hole), reverse=True) or [0, 0]
    suited = 1.0 if len(obs.hole) == 2 and suit_of(obs.hole[0]) == suit_of(obs.hole[1]) else 0.0
    pair = 1.0 if len(obs.hole) == 2 and ranks[0] == ranks[1] else 0.0
    gap = (ranks[0] - ranks[1]) / 12.0 if len(ranks) == 2 else 0.0

    tex = _board_texture(obs.board)
    dc = draw_class(obs.hole, obs.board) if obs.board else 0
    pct = strength_percentile(obs.hole, obs.board) if obs.board else 0.0

    stacks_live = [obs.stacks[i] + obs.street_committed[i] for i in range(obs.num_players)
                   if obs.in_hand[i] and not obs.folded[i]]
    effective = min(stacks_live) if stacks_live else obs.my_stack
    pot = max(1, obs.pot)
    spr = min(effective / pot, 20.0) / 20.0

    ctx = street_context(obs)
    pf_agg = preflop_aggressor(obs)
    aggressor = last_aggressor(obs)
    in_position = 1.0
    if aggressor is not None and aggressor != obs.seat:
        in_position = 1.0 if _relative_position(obs) > 0.5 else 0.0

    stats = opponent_stats or {}
    committed_total = obs.total_committed[obs.seat]

    f = np.zeros(NUM_FEATURES, dtype=np.float32)
    i = FEATURE_INDEX
    f[i["equity_vs_field"]] = equity_field
    f[i["equity_heads_up"]] = equity_heads_up
    f[i["made_hand_percentile"]] = pct
    f[i["draw_flush"]] = 1.0 if dc == 3 else 0.0
    f[i["draw_oesd"]] = 1.0 if dc == 2 else 0.0
    f[i["draw_gutshot"]] = 1.0 if dc == 1 else 0.0
    f[i["hole_high_rank"]] = ranks[0] / 12.0
    f[i["hole_low_rank"]] = (ranks[1] if len(ranks) > 1 else ranks[0]) / 12.0
    f[i["hole_suited"]] = suited
    f[i["hole_pair"]] = pair
    f[i["hole_gap"]] = gap
    f[i["board_count"]] = len(obs.board) / 5.0
    f[i["board_paired"]] = tex["paired"]
    f[i["board_trips"]] = tex["trips"]
    f[i["board_three_suited"]] = tex["three_suited"]
    f[i["board_four_suited"]] = tex["four_suited"]
    f[i["board_connected"]] = tex["connected"]
    f[i["board_high_rank"]] = tex["high"]
    f[i["board_low_rank"]] = tex["low"]
    f[i[f"street_{['preflop', 'flop', 'turn', 'river'][min(obs.street, 3)]}"]] = 1.0
    f[i["pot_log"]] = min(np.log1p(pot / bb) / 6.0, 1.5)
    f[i["pot_vs_stacks"]] = pot / max(1.0, pot + sum(stacks_live))
    f[i["to_call_log"]] = min(np.log1p(obs.to_call / bb) / 6.0, 1.5)
    f[i["pot_odds"]] = obs.pot_odds
    f[i["spr"]] = spr
    f[i["my_stack_bb"]] = min(obs.my_stack / (bb * 200.0), 1.5)
    f[i["effective_stack_bb"]] = min(effective / (bb * 200.0), 1.5)
    f[i["committed_fraction"]] = committed_total / max(1.0, committed_total + obs.my_stack)
    f[i["live_opponents"]] = opponents / 8.0
    f[i["table_size"]] = obs.num_players / 9.0
    f[i["relative_position"]] = _relative_position(obs)
    f[i["is_button"]] = 1.0 if obs.seat == obs.button else 0.0
    blinds = {(obs.button + 1) % obs.num_players, (obs.button + 2) % obs.num_players}
    f[i["is_blind"]] = 1.0 if obs.street == 0 and obs.seat in blinds else 0.0
    f[i["in_position"]] = in_position
    f[i["facing_bet"]] = 1.0 if obs.to_call > 0 else 0.0
    f[i["raises_this_street"]] = min(ctx["raises"], 4.0) / 4.0
    f[i["callers_this_street"]] = min(ctx["callers"], 8.0) / 8.0
    f[i["last_was_raise"]] = ctx["last_was_raise"]
    f[i["i_am_pf_aggressor"]] = 1.0 if pf_agg == obs.seat else 0.0
    f[i["checked_to_me"]] = 1.0 if (obs.to_call == 0 and obs.street > 0) else 0.0
    f[i["bet_faced_vs_pot"]] = min(obs.to_call / pot, 3.0) / 3.0
    f[i["opp_vpip"]] = stats.get("vpip", 0.35)
    f[i["opp_pfr"]] = stats.get("pfr", 0.22)
    f[i["opp_aggression"]] = stats.get("aggression", 0.5)
    f[i["opp_fold_to_bet"]] = stats.get("fold_to_bet", 0.45)
    f[i["opp_wtsd"]] = stats.get("wtsd", 0.28)
    f[i["opp_confidence"]] = stats.get("confidence", 0.0)
    return f


def explain(vector: np.ndarray, top: int = 10) -> str:  # pragma: no cover - debug aid
    order = np.argsort(-np.abs(vector))[:top]
    return ", ".join(f"{FEATURE_NAMES[i]}={vector[i]:.2f}" for i in order)


# --- public-information encoder --------------------------------------------
# Used by the opponent model, which must predict what somebody will do without
# seeing their cards.  Deliberately a separate, smaller vector: no equity, no
# hole cards, nothing the observer is not entitled to know.

PUBLIC_FEATURE_NAMES: List[str] = [
    "street_preflop", "street_flop", "street_turn", "street_river",
    "pot_log", "to_call_log", "pot_odds", "bet_faced_vs_pot", "spr",
    "stack_bb", "committed_fraction", "facing_bet", "checked_to_them",
    "raises_this_street", "callers_this_street", "last_was_raise",
    "relative_position", "is_button", "is_blind", "is_pf_aggressor",
    "live_opponents", "table_size",
]
NUM_PUBLIC_FEATURES = len(PUBLIC_FEATURE_NAMES)
PUBLIC_INDEX = {name: i for i, name in enumerate(PUBLIC_FEATURE_NAMES)}


def encode_public(obs: Observation) -> np.ndarray:
    """Encode the situation facing ``obs.seat`` using only public information."""
    bb = max(1, obs.big_blind)
    pot = max(1, obs.pot)
    stacks_live = [obs.stacks[i] + obs.street_committed[i] for i in range(obs.num_players)
                   if obs.in_hand[i] and not obs.folded[i]]
    effective = min(stacks_live) if stacks_live else obs.my_stack
    ctx = street_context(obs)
    pf_agg = preflop_aggressor(obs)
    committed = obs.total_committed[obs.seat]
    blinds = {(obs.button + 1) % obs.num_players, (obs.button + 2) % obs.num_players}

    f = np.zeros(NUM_PUBLIC_FEATURES, dtype=np.float32)
    i = PUBLIC_INDEX
    f[i[f"street_{['preflop', 'flop', 'turn', 'river'][min(obs.street, 3)]}"]] = 1.0
    f[i["pot_log"]] = min(np.log1p(pot / bb) / 6.0, 1.5)
    f[i["to_call_log"]] = min(np.log1p(obs.to_call / bb) / 6.0, 1.5)
    f[i["pot_odds"]] = obs.pot_odds
    f[i["bet_faced_vs_pot"]] = min(obs.to_call / pot, 3.0) / 3.0
    f[i["spr"]] = min(effective / pot, 20.0) / 20.0
    f[i["stack_bb"]] = min(obs.my_stack / (bb * 200.0), 1.5)
    f[i["committed_fraction"]] = committed / max(1.0, committed + obs.my_stack)
    f[i["facing_bet"]] = 1.0 if obs.to_call > 0 else 0.0
    f[i["checked_to_them"]] = 1.0 if (obs.to_call == 0 and obs.street > 0) else 0.0
    f[i["raises_this_street"]] = min(ctx["raises"], 4.0) / 4.0
    f[i["callers_this_street"]] = min(ctx["callers"], 8.0) / 8.0
    f[i["last_was_raise"]] = ctx["last_was_raise"]
    f[i["relative_position"]] = _relative_position(obs)
    f[i["is_button"]] = 1.0 if obs.seat == obs.button else 0.0
    f[i["is_blind"]] = 1.0 if obs.street == 0 and obs.seat in blinds else 0.0
    f[i["is_pf_aggressor"]] = 1.0 if pf_agg == obs.seat else 0.0
    f[i["live_opponents"]] = max(1, obs.live_opponents) / 8.0
    f[i["table_size"]] = obs.num_players / 9.0
    return f

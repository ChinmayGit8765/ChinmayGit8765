"""Multiplayer table over HTTP — several humans plus bots, no dependencies.

The engine runs on its own thread and plays hands continuously.  When a seat
claimed by a person is on the clock, that thread blocks on a queue until the
browser posts an action (or the clock runs out and the seat checks or folds).
Unclaimed seats are played by the bots, so a table is never short-handed.

Only the standard library is used: ``http.server`` plus a single-page client
served from :mod:`holdem.server.client`.
"""

from __future__ import annotations

import json
import queue
import random
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from ..bots.neural import NeuralBot
from ..cards import card_str
from ..engine import Action, ActionRecord, ActionType, HandResult, Observation
from ..equity import fast_equity
from ..evaluator import describe, evaluate
from ..game import BaseAgent, Game
from ..ml.opponent import OpponentTracker
from .client import CLIENT_HTML

ACTION_TIMEOUT = 90.0


@dataclass
class Human:
    player_id: str
    name: str
    seat: int
    last_seen: float = field(default_factory=time.time)
    inbox: "queue.Queue[Action]" = field(default_factory=queue.Queue)


class SeatAgent(BaseAgent):
    """A seat that a person can take over from the bot at any time."""

    def __init__(self, name: str, bot: BaseAgent, table: "TableServer", seat: int):
        super().__init__(name)
        self.bot = bot
        self.table = table
        self.seat_index = seat
        self.human: Optional[Human] = None

    @property
    def display_name(self) -> str:
        return self.human.name if self.human else self.bot.name

    def act(self, obs: Observation) -> Action:
        human = self.human
        if human is None:
            action = self.bot.act(obs)
            self.table.note_action(obs, delay=self.table.bot_delay)
            return action

        self.table.set_pending(self.seat_index, obs)
        deadline = time.time() + ACTION_TIMEOUT
        try:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise queue.Empty
                try:
                    action = human.inbox.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    continue
                if self._legal(action, obs):
                    return action
        except queue.Empty:
            self.table.log(f"{human.name} timed out")
            return self._default(obs)
        finally:
            self.table.clear_pending()

    @staticmethod
    def _legal(action: Action, obs: Observation) -> bool:
        for la in obs.legal:
            if la.type != action.type:
                continue
            if action.type in (ActionType.BET, ActionType.RAISE):
                return la.min_amount <= action.amount <= la.max_amount
            return True
        return False

    @staticmethod
    def _default(obs: Observation) -> Action:
        if obs.to_call == 0:
            return Action(ActionType.CHECK)
        return Action(ActionType.FOLD)

    def on_action(self, record: ActionRecord, obs_public: Observation) -> None:
        hook = getattr(self.bot, "on_action", None)
        if hook:
            hook(record, obs_public)

    def on_hand_end(self, result: HandResult, seat: int) -> None:
        hook = getattr(self.bot, "on_hand_end", None)
        if hook:
            hook(result, seat)


class TableServer:
    """Owns the game thread and the shared, lock-protected view of the table."""

    def __init__(self, seats: int = 6, bots: int = 3, difficulty: str = "regular",
                 stack: int = 200, sb: int = 1, bb: int = 2,
                 bot_delay: float = 0.6, seed: Optional[int] = None):
        self.lock = threading.RLock()
        self.rng = random.Random(seed)
        self.tracker = OpponentTracker()
        self.bot_delay = bot_delay
        self.sb, self.bb = sb, bb
        names = ["Nova", "Rook", "Vega", "Juno", "Atlas", "Kite", "Onyx", "Sable"]
        self.agents: List[SeatAgent] = []
        for i in range(seats):
            bot = NeuralBot(names[i % len(names)], difficulty=difficulty, rng=self.rng,
                            tracker=self.tracker)
            self.agents.append(SeatAgent(bot.name, bot, self, i))
        self.game = Game(self.agents, starting_stack=stack, sb=sb, bb=bb, rng=self.rng)
        self.humans: Dict[str, Human] = {}
        self.pending_seat: Optional[int] = None
        self.pending_obs: Optional[Observation] = None
        self.snapshot: Dict[str, Any] = {}
        self.log_lines: List[str] = []
        self.hand_number = 0
        self.last_result: Optional[HandResult] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._refresh(None)

    # -- table state ---------------------------------------------------------

    def log(self, line: str) -> None:
        with self.lock:
            self.log_lines.append(line)
            del self.log_lines[:-60]

    def note_action(self, obs: Observation, delay: float = 0.0) -> None:
        if delay:
            time.sleep(delay)

    def set_pending(self, seat: int, obs: Observation) -> None:
        with self.lock:
            self.pending_seat = seat
            self.pending_obs = obs
            self._refresh(obs)

    def clear_pending(self) -> None:
        with self.lock:
            self.pending_seat = None
            self.pending_obs = None

    def _refresh(self, obs: Optional[Observation]) -> None:
        players = []
        for i, agent in enumerate(self.agents):
            state = self.game.players[i]
            players.append({
                "seat": i,
                "name": agent.display_name,
                "human": agent.human is not None,
                "stack": state.stack,
                "committed": state.street_committed,
                "folded": state.folded,
                "allIn": state.all_in,
                "inHand": state.in_hand,
            })
        self.snapshot = {
            "players": players,
            "button": self.game.button,
            "handNumber": self.hand_number,
            "blinds": [self.sb, self.bb],
            "board": [card_str(c) for c in (obs.board if obs else [])],
            "pot": obs.pot if obs else 0,
            "street": obs.street if obs else 0,
            "toAct": self.pending_seat,
            "log": list(self.log_lines[-18:]),
        }

    def view_for(self, player_id: Optional[str]) -> Dict[str, Any]:
        with self.lock:
            view = dict(self.snapshot)
            view["you"] = None
            human = self.humans.get(player_id) if player_id else None
            if human:
                human.last_seen = time.time()
                agent = self.agents[human.seat]
                state = self.game.players[human.seat]
                you: Dict[str, Any] = {
                    "seat": human.seat,
                    "name": human.name,
                    "stack": state.stack,
                    "hole": [card_str(c) for c in state.hole],
                    "yourTurn": self.pending_seat == human.seat,
                }
                if self.pending_seat == human.seat and self.pending_obs is not None:
                    obs = self.pending_obs
                    you["toCall"] = obs.to_call
                    you["potOdds"] = round(obs.pot_odds, 3)
                    you["legal"] = [
                        {"type": la.type.name.lower(), "min": la.min_amount,
                         "max": la.max_amount} for la in obs.legal
                    ]
                    you["equity"] = round(
                        fast_equity(obs.hole, obs.board, max(1, obs.live_opponents),
                                    iters=600), 3)
                    if len(obs.board) >= 3:
                        you["madeHand"] = describe(evaluate(list(obs.hole) + list(obs.board)))
                view["you"] = you
            return view

    # -- seats ---------------------------------------------------------------

    def join(self, name: str) -> Dict[str, Any]:
        with self.lock:
            taken = {h.seat for h in self.humans.values()}
            free = [i for i in range(len(self.agents)) if i not in taken]
            if not free:
                return {"error": "table is full"}
            seat = free[0]
            player_id = secrets.token_hex(8)
            human = Human(player_id=player_id, name=name[:16] or f"Player{seat}", seat=seat)
            self.humans[player_id] = human
            self.agents[seat].human = human
            self.game.players[seat].name = human.name
            self.log(f"{human.name} sits down in seat {seat + 1}")
            self._refresh(self.pending_obs)
            return {"playerId": player_id, "seat": seat, "name": human.name}

    def leave(self, player_id: str) -> Dict[str, Any]:
        with self.lock:
            human = self.humans.pop(player_id, None)
            if not human:
                return {"error": "unknown player"}
            agent = self.agents[human.seat]
            agent.human = None
            self.game.players[human.seat].name = agent.bot.name
            self.log(f"{human.name} leaves the table")
            # Publish immediately: otherwise the seat still reads as human
            # until the current hand finishes.
            self._refresh(self.pending_obs)
            return {"ok": True}

    def submit(self, player_id: str, kind: str, amount: int) -> Dict[str, Any]:
        with self.lock:
            human = self.humans.get(player_id)
            if not human:
                return {"error": "unknown player"}
            if self.pending_seat != human.seat:
                return {"error": "not your turn"}
        try:
            action_type = ActionType[kind.upper()]
        except KeyError:
            return {"error": f"unknown action {kind!r}"}
        human.inbox.put(Action(action_type, int(amount)))
        return {"ok": True}

    # -- the game thread -----------------------------------------------------

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="holdem-table", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False

    def _loop(self) -> None:
        while self.running:
            try:
                self.hand_number += 1
                result = self.game.play_hand(keep_history=False)
                self.last_result = result
                self._log_result(result)
                with self.lock:
                    self._refresh(None)
                time.sleep(1.2 if any(a.human for a in self.agents) else 0.05)
            except Exception as exc:  # pragma: no cover - keep the table alive
                self.log(f"hand aborted: {exc}")
                time.sleep(1.0)

    def _log_result(self, result: HandResult) -> None:
        for record in result.history:
            self.log(record.describe())
        if result.board:
            self.log("board " + " ".join(card_str(c) for c in result.board))
        for seat, hole in result.revealed.items():
            self.log(f"{result.names[seat]} shows "
                     f"{' '.join(card_str(c) for c in hole)}")
        for pot in result.pots:
            who = ", ".join(result.names[s] for s in pot.winners)
            self.log(f"pot {pot.amount} to {who} ({pot.description})")
        self.log("—" * 20)


def make_handler(table: TableServer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "holdem-ml/1.0"

        def log_message(self, fmt, *args):  # pragma: no cover - quieten the server
            pass

        def _send(self, payload: Any, status: int = 200,
                  content_type: str = "application/json") -> None:
            body = (json.dumps(payload) if content_type == "application/json"
                    else payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self):  # noqa: N802 - http.server API
            route = urlparse(self.path)
            if route.path in ("/", "/index.html"):
                self._send(CLIENT_HTML, content_type="text/html")
            elif route.path == "/api/state":
                params = parse_qs(route.query)
                player_id = (params.get("playerId") or [None])[0]
                self._send(table.view_for(player_id))
            else:
                self._send({"error": "not found"}, status=404)

        def do_POST(self):  # noqa: N802 - http.server API
            route = urlparse(self.path)
            body = self._body()
            if route.path == "/api/join":
                self._send(table.join(str(body.get("name", "Player"))))
            elif route.path == "/api/leave":
                self._send(table.leave(str(body.get("playerId", ""))))
            elif route.path == "/api/action":
                self._send(table.submit(str(body.get("playerId", "")),
                                        str(body.get("type", "")),
                                        int(body.get("amount", 0) or 0)))
            else:
                self._send({"error": "not found"}, status=404)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000, bots: int = 3,
          seats: int = 6, difficulty: str = "regular", stack: int = 200,
          sb: int = 1, bb: int = 2) -> None:
    table = TableServer(seats=seats, bots=bots, difficulty=difficulty,
                        stack=stack, sb=sb, bb=bb)
    table.start()
    httpd = ThreadingHTTPServer((host, port), make_handler(table))
    print(f"holdem-ml table running at http://{host}:{port}  "
          f"({seats} seats, bots on '{difficulty}')")
    print("open that address in a browser — several people can join the same table")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        table.stop()
        httpd.server_close()

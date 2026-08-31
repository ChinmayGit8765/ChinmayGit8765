import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from holdem.server.app import TableServer, make_handler
from holdem.server.client import CLIENT_HTML

PORT = 8791
BASE = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module")
def server():
    table = TableServer(seats=4, difficulty="casual", bot_delay=0.0, seed=7)
    table.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(table))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while time.time() < deadline and table.hand_number < 2:
        time.sleep(0.1)
    yield table
    table.stop()
    httpd.shutdown()
    httpd.server_close()


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=10).read())


def post(path, body):
    request = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request, timeout=10).read())


def test_client_page_is_served(server):
    html = urllib.request.urlopen(BASE + "/", timeout=10).read().decode()
    assert html == CLIENT_HTML
    assert "holdem" in html and 'id="seats"' in html


def test_bots_play_on_their_own(server):
    first = get("/api/state")
    assert first["handNumber"] >= 1
    assert len(first["players"]) == 4
    assert all(not p["human"] for p in first["players"])
    deadline = time.time() + 20
    while time.time() < deadline and get("/api/state")["handNumber"] <= first["handNumber"]:
        time.sleep(0.2)
    assert get("/api/state")["handNumber"] > first["handNumber"], "the table keeps dealing"


def test_state_hides_other_players_cards(server):
    view = get("/api/state")
    assert view["you"] is None
    assert "hole" not in json.dumps(view["players"])


def test_join_act_and_leave(server):
    joined = post("/api/join", {"name": "Tester"})
    assert "playerId" in joined, joined
    player_id = joined["playerId"]
    seat = joined["seat"]
    try:
        view = get(f"/api/state?playerId={player_id}")
        assert view["players"][seat]["human"] is True
        assert view["players"][seat]["name"] == "Tester"

        deadline = time.time() + 40
        acted = False
        while time.time() < deadline:
            view = get(f"/api/state?playerId={player_id}")
            you = view.get("you") or {}
            if you.get("yourTurn"):
                assert len(you["hole"]) == 2, "a seated player sees their own cards"
                assert 0.0 <= you["equity"] <= 1.0
                kinds = {la["type"] for la in you["legal"]}
                assert kinds & {"check", "call"}
                choice = next(la for la in you["legal"]
                              if la["type"] in ("check", "call"))
                assert post("/api/action",
                            {"playerId": player_id, "type": choice["type"],
                             "amount": choice["min"]}) == {"ok": True}
                acted = True
                break
            time.sleep(0.2)
        assert acted, "never got the chance to act"
    finally:
        assert post("/api/leave", {"playerId": player_id}) == {"ok": True}
    assert get("/api/state")["players"][seat]["human"] is False


def test_acting_out_of_turn_is_rejected(server):
    joined = post("/api/join", {"name": "Impatient"})
    player_id = joined["playerId"]
    try:
        for _ in range(5):
            view = get(f"/api/state?playerId={player_id}")
            if not (view.get("you") or {}).get("yourTurn"):
                assert "error" in post("/api/action",
                                       {"playerId": player_id, "type": "check", "amount": 0})
                return
            time.sleep(0.3)
    finally:
        post("/api/leave", {"playerId": player_id})


def test_unknown_player_and_route(server):
    assert "error" in post("/api/action", {"playerId": "nope", "type": "check"})
    assert "error" in post("/api/leave", {"playerId": "nope"})
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(BASE + "/nope", timeout=5)


def test_table_fills_up(server):
    ids = []
    try:
        while True:
            res = post("/api/join", {"name": f"P{len(ids)}"})
            if "error" in res:
                assert "full" in res["error"]
                break
            ids.append(res["playerId"])
            assert len(ids) <= 4
        assert len(ids) >= 1
    finally:
        for pid in ids:
            post("/api/leave", {"playerId": pid})

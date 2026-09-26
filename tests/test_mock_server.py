"""Mock server behaviour: endpoints, envelope, rec rebroadcast, --no-recs, causality."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.replay.mock_server import create_app

KINDS = {"tick", "detection", "risk", "rec", "official"}


def test_http_endpoints() -> None:
    with TestClient(create_app(autoplay=False)) as c:
        assert c.get("/track").json()["ref_line"]
        assert c.get("/races").json() == [c.get("/track").json()["race"]]
        assert c.get("/official").json()[0]["flag"]
        st = c.post("/replay", json={"speed": 10, "seek_t": 4380.0}).json()
        assert st["speed"] == 10 and st["t"] == 4380.0
        assert c.get("/").status_code == 200


def test_dashboard_files_are_never_served_stale() -> None:
    with TestClient(create_app(autoplay=False)) as c:
        for path in ("/", "/app.js"):
            r = c.get(path)
            assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"


def test_stream_envelopes_and_rec_rebroadcast() -> None:
    with TestClient(create_app()) as c:
        c.post("/replay", json={"speed": 50, "seek_t": 4385.0})
        with c.websocket_connect("/stream") as ws:
            env = ws.receive_json()
            assert env["kind"] in KINDS and isinstance(env["data"], dict)
            rec = {"id": "rec-test", "t": 4386.0, "msector": 9, "flag": "YELLOW", "confidence": 0.5,
                   "reason": "test", "message": "YELLOW IN TRACK SECTOR 9", "source_detections": []}
            ws.send_json({"kind": "rec", "data": rec})
            for _ in range(2000):
                env = ws.receive_json()
                if env["kind"] == "rec" and env["data"]["id"] == "rec-test":
                    break
            else:
                raise AssertionError("rec was not rebroadcast")


def test_no_recs_and_causal_order() -> None:
    app = create_app(no_recs=True, autoplay=False)
    r = app.state.replay
    assert all(env["kind"] != "rec" for _, _, env in r.events)
    r.seek(4300.0)
    r.sim_t = 4310.0
    sent = r.due()
    assert sent and all(env["data"]["t"] <= 4310.0 for env in sent)
    assert r.events[r.cursor][0] > 4310.0

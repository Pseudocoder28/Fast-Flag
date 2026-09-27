"""Replay engine and real server: strict causality, time order, contract shapes.

Uses a small synthetic race so it runs without data/ (which is gitignored)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from test_contracts import check_official, check_tick

from src.replay.engine import CausalityError, Engine, RaceData
from src.replay.server import create_app

T0 = 1000.0


def synthetic_race(seconds: float = 40.0) -> RaceData:
    t = np.round(np.arange(T0, T0 + seconds, 0.25), 2)
    rows = []
    for i, drv in enumerate(["1", "16", "44"]):
        rows.append(pd.DataFrame({
            "t": t, "drv": drv, "x": t + 10.0 * i, "y": 0.0, "dist": (t * 50 + 300 * i) % 5000,
            "lat_off": 0.1, "speed": 250.0, "throttle": 100.0, "brake": 0.0, "gear": 7.0, "rpm": 11000.0,
            "msector": 3, "gap_ahead_m": 50.0, "in_pit": False, "lap": 5, "track_status": "1"}))
    df = pd.concat(rows, ignore_index=True)
    df.loc[(df["drv"] == "44") & (df["t"] > T0 + 20), ["x", "y", "speed", "dist"]] = np.nan  # car 44 drops out
    official = [{"t": T0 + 10.5, "category": "Flag", "message": "YELLOW IN TRACK SECTOR 3", "flag": "YELLOW",
                 "scope": "Sector", "msector": 3, "drivers": []}]
    track = {"race": "TEST", "ref_line": [[0.0, 0.0]] * 20, "msectors": [], "corners": []}
    return RaceData.from_frame("TEST", df, track, official)


class Spy:
    """Records what it is shown and emits one detection per tick, stamped at the tick."""

    def __init__(self) -> None:
        self.seen: list[tuple[float, float, float]] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def on_tick(self, t: float, frame: pd.DataFrame, tick: dict) -> list[dict]:
        self.seen.append((t, float(frame["t"].max()), tick["t"]))
        return [{"kind": "detection", "data": {"id": f"det-{t}", "t": t}}]


def test_processors_only_see_the_current_tick() -> None:
    spy = Spy()
    eng = Engine(synthetic_race(), [spy])
    envs = eng.advance(T0 + 20)
    assert spy.seen and all(frame_max == t == tick_t for t, frame_max, tick_t in spy.seen)
    ts = [e["data"]["t"] for e in envs]
    assert ts == sorted(ts) and max(ts) <= T0 + 20
    assert eng.advance(T0 + 20) == []          # nothing is sent twice


def test_official_events_arrive_at_their_time_only() -> None:
    eng = Engine(synthetic_race())
    assert not any(e["kind"] == "official" for e in eng.advance(T0 + 10.25))
    off = [e for e in eng.advance(T0 + 11) if e["kind"] == "official"]
    assert len(off) == 1 and off[0]["data"]["t"] == T0 + 10.5


def test_processor_output_from_the_future_is_rejected() -> None:
    class Cheat(Spy):
        def on_tick(self, t: float, frame: pd.DataFrame, tick: dict) -> list[dict]:
            return [{"kind": "risk", "data": {"t": t + 1.0}}]

    eng = Engine(synthetic_race())
    eng.processors = [Cheat()]
    with pytest.raises(CausalityError):
        eng.advance(T0 + 1)


def test_seek_resets_and_warms_up_on_history_only() -> None:
    spy = Spy()
    eng = Engine(synthetic_race(), [spy])
    eng.advance(T0 + 30)
    spy.seen.clear()
    eng.seek(T0 + 15)
    assert spy.resets == 2
    assert spy.seen and max(t for t, _, _ in spy.seen) < T0 + 15    # warm-up used only the past
    envs = eng.advance(T0 + 16)
    assert envs and min(e["data"]["t"] for e in envs) >= T0 + 15


def test_ticks_match_contract_and_leave_out_dead_cars() -> None:
    eng = Engine(synthetic_race())
    for env in eng.advance(T0 + 30):
        if env["kind"] == "tick":
            check_tick(env["data"], "tick")
            if env["data"]["t"] > T0 + 21:
                assert {c["drv"] for c in env["data"]["cars"]} == {"1", "16"}
        else:
            check_official(env["data"], "official")


def test_server_endpoints_and_stream() -> None:
    with TestClient(create_app(synthetic_race())) as c:
        assert c.get("/track").json()["race"] == "TEST"
        assert c.get("/official").json()[0]["flag"] == "YELLOW"
        st = c.post("/replay", json={"speed": 20, "seek_t": T0 + 5}).json()
        assert st["speed"] == 20 and st["t"] == T0 + 5
        with c.websocket_connect("/stream") as ws:
            env = ws.receive_json()
            assert env["kind"] in {"tick", "official"} and env["data"]["t"] >= T0 + 5
            ws.send_json({"kind": "rec", "data": {"id": "rec-x", "t": T0 + 6}})
            for _ in range(500):
                if ws.receive_json()["kind"] == "rec":
                    break
            else:
                raise AssertionError("rec was not rebroadcast")


def test_server_seek_while_paused_sends_the_new_position() -> None:
    with TestClient(create_app(synthetic_race(), autoplay=False)) as c:
        with c.websocket_connect("/stream") as ws:
            c.post("/replay", json={"speed": 0, "seek_t": T0 + 12})
            env = ws.receive_json()
            assert env["kind"] == "tick" and T0 + 12 <= env["data"]["t"] < T0 + 13


def test_server_refuses_holdout_by_default() -> None:
    with TestClient(create_app(synthetic_race(), autoplay=False)) as c:
        assert c.post("/replay", json={"race": "2026_Azerbaijan"}).status_code == 403


def test_every_processor_is_timed_per_stage() -> None:
    from src.replay.latency import LatencyTracker, stage_name
    risk_like = type("RiskModel", (), {"__module__": "src.predict.risk"})()
    assert stage_name(risk_like) == "predict"          # new stages are named and timed with no edits
    spy = Spy()
    eng = Engine(synthetic_race(), [spy])
    steps = list(eng.steps(T0 + 5))
    assert steps and all({"tick", "frame", "Spy"} <= set(s.stage_s) for s in steps)
    lines: list[str] = []
    tr = LatencyTracker(report_every_s=0.0, printer=lines.append)
    tr.record({"tick": 0.001, "detect": 0.3}, total_s=0.3)       # 300 ms: over the 250 ms budget
    tr.record({"tick": 0.001, "detect": 0.01}, total_s=0.01)
    tr.maybe_report()
    assert tr.summary()["over_budget"] == 1 and tr.summary()["stages"]["detect"]["n"] == 2
    assert lines and "over 250 ms: 1" in lines[0]


def test_a_page_that_opens_late_gets_what_it_missed_then_live() -> None:
    with TestClient(create_app(synthetic_race(), autoplay=False)) as c:
        with c.websocket_connect("/stream") as rc:
            c.post("/replay", json={"speed": 0, "seek_t": T0 + 12})
            assert rc.receive_json()["kind"] == "tick"
            rc.send_json({"kind": "rec", "data": {"id": "rec-a", "t": T0 + 12}})
            assert rc.receive_json()["data"]["id"] == "rec-a"
            with c.websocket_connect("/stream?catchup=1") as page:
                first, second = page.receive_json(), page.receive_json()
                assert first["data"]["id"] == "rec-a" and second["kind"] == "tick"   # what it missed, the tick last
            with c.websocket_connect("/stream") as live:                              # race control, the bridge: live only
                rc.send_json({"kind": "rec", "data": {"id": "rec-b", "t": T0 + 12}})
                assert live.receive_json()["data"]["id"] == "rec-b"
            c.post("/replay", json={"seek_t": T0 + 25})                              # a seek: nothing from before it
            with c.websocket_connect("/stream?catchup=1") as page:
                env = page.receive_json()
                assert env["kind"] == "tick" and env["data"]["t"] >= T0 + 25


class OneStop:
    """Emits one STOPPED detection for car 44 at T0 + 5, as the real detectors would."""

    def reset(self) -> None:
        pass

    def on_tick(self, t: float, frame: pd.DataFrame, tick: dict) -> list[dict]:
        if t != T0 + 5:
            return []
        return [{"kind": "detection", "data": {"id": "det-stop", "t": t, "drivers": ["44"], "msector": 3,
                                               "type": "STOPPED", "severity": 0.9, "evidence": "speed 0 km/h"}}]


def precomputed_app(tmp_path, monkeypatch):
    import src.replay.timeline as timeline
    monkeypatch.setattr(timeline, "CACHE", tmp_path)
    return create_app(synthetic_race(), make_processors=lambda race: [OneStop()], autoplay=False, timeline=True)


def test_precomputed_race_sends_our_recs_itself(tmp_path, monkeypatch) -> None:
    app = precomputed_app(tmp_path, monkeypatch)
    replay = app.state.replay
    recs = [e for tick in replay.timeline for e in tick if e["kind"] == "rec"]
    assert recs and recs[0]["data"]["msector"] == 3 and recs[0]["data"]["t"] >= T0 + 5
    assert (tmp_path / "TEST.json.gz").exists()                             # cached for the next start
    with TestClient(app) as c, c.websocket_connect("/stream") as rc:
        rc.send_json({"kind": "rec", "data": {"id": "rec-extra", "t": T0}})   # a second race control ...
        c.post("/replay", json={"speed": 0, "seek_t": T0 + 1})
        assert rc.receive_json()["kind"] == "tick"                             # ... is not rebroadcast


def test_precomputed_seek_gives_pages_the_state_of_a_continuous_run(tmp_path, monkeypatch) -> None:
    app = precomputed_app(tmp_path, monkeypatch)
    with TestClient(app) as c, c.websocket_connect("/stream?catchup=1") as page:
        c.post("/replay", json={"speed": 0, "seek_t": T0 + 20})
        tick = page.receive_json()
        assert tick["kind"] == "tick" and tick["data"]["t"] >= T0 + 20       # the page resets on this jump
        burst = [page.receive_json() for _ in range(3)]
        kinds = [e["kind"] for e in burst]
        assert kinds == ["detection", "rec", "official"]                       # everything before, in play order
        assert all(e["data"]["t"] < T0 + 20 for e in burst)
    with TestClient(app) as c, c.websocket_connect("/stream?catchup=1") as late:
        seen = [late.receive_json() for _ in range(4)]                         # a page opened now: same history
        assert [e["kind"] for e in seen] == ["detection", "rec", "official", "tick"]


def test_cars_out_of_the_race_stay_out_through_a_stoppage() -> None:
    from src.replay.cars import Cars
    t = np.round(np.arange(T0, T0 + 200, 0.25), 2)
    frames = []
    for drv, x in (("1", t * 50), ("44", np.where(t < T0 + 10, t * 50, (T0 + 10) * 50)), ("16", t * 50)):
        frames.append(pd.DataFrame({"t": t, "drv": drv, "x": x, "y": 0.0, "in_pit": False, "msector": 3}))
    rows = pd.concat(frames)
    rows = rows[~((rows["drv"] == "16") & (rows["t"] > T0 + 20))]             # car 16 goes silent
    stops = t[(t >= T0 + 100) & (t < T0 + 150)]                                # a red flag: the field stopped
    cars = Cars(rows, stops, T0)
    at = lambda s: {o["drv"]: o for o in cars.at(T0 + s)["out"]}              # noqa: E731
    assert at(50) == {}                                                        # 44 still for 40 s only
    assert at(75)["44"]["why"] == "stopped" and at(75)["44"]["since"] == T0 + 10
    assert at(85)["16"]["why"] == "no data"
    assert "44" in at(120)                                                     # still out during the red flag


def test_a_seek_never_sends_official_messages_from_before_the_replay_start(tmp_path, monkeypatch) -> None:
    import src.replay.timeline as timeline
    monkeypatch.setattr(timeline, "CACHE", tmp_path)
    race = synthetic_race()
    early = {"t": T0 - 50, "category": "Flag", "message": "YELLOW IN TRACK SECTOR 1", "flag": "YELLOW",
             "scope": "Sector", "msector": 1, "drivers": []}
    race = RaceData.from_frame("TEST", race.frame, race.track, [early] + race.official)
    app = create_app(race, make_processors=lambda r: [OneStop()], autoplay=False, timeline=True)
    with TestClient(app) as c, c.websocket_connect("/stream?catchup=1") as page:
        c.post("/replay", json={"speed": 0, "seek_t": T0 + 20})
        assert page.receive_json()["kind"] == "tick"
        burst = [page.receive_json() for _ in range(3)]
        assert all(e["data"]["t"] >= T0 for e in burst)          # as a continuous run from the start: never before it

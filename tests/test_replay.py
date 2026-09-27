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

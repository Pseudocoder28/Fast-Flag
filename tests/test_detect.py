"""Detectors on synthetic races: what fires, what stays quiet, and causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
from test_contracts import check_detection

from src.detect.detectors import Config, DetectorSuite, NaiveThreshold
from src.replay.engine import Engine, RaceData

T0, REF = 1000.0, 250.0
CARS = ["1", "16", "44", "63"]


def race_frame(seconds: float = 60.0) -> pd.DataFrame:
    t = np.round(np.arange(T0, T0 + seconds, 0.25), 2)
    rows = [pd.DataFrame({"t": t, "drv": d, "speed": REF, "x": t * 70 + 100 * i, "y": 0.0,
                          "dist": (t * 70 + 100 * i) % 5000, "msector": 5, "in_pit": False, "brake": 0.0,
                          "lat_off": 0.2, "heading_err": 1.0, "ref_speed": REF, "throttle": 100.0,
                          "gear": 8.0, "rpm": 11000.0, "gap_ahead_m": 80.0, "lap": 10, "track_status": "1",
                          "suspended": False, "field_slow": 0.0})
            for i, d in enumerate(CARS)]
    return pd.concat(rows, ignore_index=True)


def finish(df: pd.DataFrame) -> RaceData:
    """Recompute the ingest columns the detectors read, the same way ingest does."""
    df = df.sort_values(["drv", "t"]).copy()
    g = df.groupby("drv", sort=False)
    df["dspeed_1s"] = df["speed"] - g["speed"].shift(4)
    df["dspeed_min_2s"] = g["dspeed_1s"].rolling(8, min_periods=2).min().reset_index(level=0, drop=True)
    df["speed_ratio"] = df["speed"] / df["ref_speed"]
    on = df["speed"].notna() & ~df["in_pit"]
    med = df[on].groupby("t")["speed_ratio"].median()
    df["field_ratio"] = df["t"].map(med).fillna(1.0)
    df["rel_ratio"] = df["speed_ratio"] / df["field_ratio"].clip(0.2, 1.0)
    df["lap_ratio"] = (df["speed"] / REF).clip(upper=1.1)
    return RaceData.from_frame("TEST", df, {"race": "TEST", "msectors": [{"id": i} for i in range(1, 11)]})


def crash(df: pd.DataFrame, drv: str, at: float, msector: int = 5) -> pd.DataFrame:
    m = (df["drv"] == drv) & (df["t"] >= at)
    df.loc[m, "speed"] = np.where(df.loc[m, "t"] < at + 0.75, 60.0, 0.0)
    df.loc[m, "msector"] = msector
    return df


def detections(race: RaceData, proc=None) -> list[dict]:
    eng = Engine(race, [proc or DetectorSuite(Config(), n_sectors=10)])
    return [e["data"] for e in eng.advance(eng.t_end) if e["kind"] == "detection"]


def test_quiet_race_has_no_detections() -> None:
    assert detections(finish(race_frame())) == []


def test_crash_gives_one_impact_and_one_stopped() -> None:
    dets = detections(finish(crash(race_frame(), "44", T0 + 20)))
    kinds = [d["type"] for d in dets if d["drivers"] == ["44"]]
    assert kinds.count("IMPACT") == 1 and kinds.count("STOPPED") == 1
    assert all(T0 + 20 <= d["t"] <= T0 + 23 for d in dets)
    for d in dets:
        check_detection(d, d["id"])


def test_grid_and_suspension_stay_quiet() -> None:
    df = race_frame()
    df.loc[df["t"] < T0 + 30, ["speed", "field_slow"]] = [0.0, 1.0]       # everyone on the grid
    assert detections(finish(df)) == []
    df = crash(race_frame(), "44", T0 + 20)
    df["suspended"] = True
    assert detections(finish(df)) == []


def test_dropout_when_data_stops_at_speed() -> None:
    df = race_frame()
    m = (df["drv"] == "63") & (df["t"] >= T0 + 30)
    df.loc[m, ["speed", "x", "y", "dist"]] = np.nan
    dets = detections(finish(df))
    assert [d["type"] for d in dets] == ["DROPOUT"] and dets[0]["drivers"] == ["63"]


def test_two_cars_in_the_same_place_give_multi() -> None:
    df = crash(crash(race_frame(), "44", T0 + 20, msector=5), "16", T0 + 21, msector=6)
    multi = [d for d in detections(finish(df)) if d["type"] == "MULTI"]
    assert len(multi) == 1 and set(multi[0]["drivers"]) == {"16", "44"}


def test_baseline_fires_on_slow_car() -> None:
    dets = detections(finish(crash(race_frame(), "44", T0 + 20)), NaiveThreshold())
    assert [d["drivers"] for d in dets] == [["44"]]

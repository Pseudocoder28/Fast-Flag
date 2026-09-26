"""Evaluation helpers: the fixed onset rule and crash-site exposure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.case_study import passes
from src.eval.onset import add_own_ratio, onsets

L, V = 1000.0, 180.0             # lap length (m), cruise speed (km/h): a 20 s lap, 80 ticks


def race(seconds: float = 120.0, cars: int = 4) -> pd.DataFrame:
    t = np.round(np.arange(0, seconds, 0.25), 2)
    rows = []
    for i in range(cars):
        travelled = t * V / 3.6 + i * 100
        rows.append(pd.DataFrame({"t": t, "drv": str(i + 1), "dist": travelled % L, "x": travelled % L,
                                  "drv_lap": np.floor(travelled / L) + 1, "speed": V, "track_status": "1",
                                  "in_pit": False, "suspended": False, "msector": (travelled % L // 100 + 1).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def test_collapse_gives_one_onset_at_its_first_tick() -> None:
    df = race()
    df.loc[(df["drv"] == "1") & (df["t"] >= 86) & (df["t"] < 90), "speed"] = 40.0
    o = onsets(add_own_ratio(df, L), L)
    assert o["drv"].tolist() == ["1"] and o["t"].tolist() == [86.0]


def test_procession_gives_no_onset() -> None:
    df = race()
    df.loc[(df["t"] >= 86) & (df["t"] < 95), "speed"] = 40.0          # the whole field slows together
    assert onsets(add_own_ratio(df, L), L).empty


def test_stopped_car_gets_no_new_onset_after_an_sc() -> None:
    df = race()
    df.loc[(df["drv"] == "1") & (df["t"] >= 86), "speed"] = 0.0         # stops and stays stopped
    df.loc[(df["t"] >= 95) & (df["t"] < 100), "track_status"] = "4"     # SC: the gate closes, then reopens
    o = onsets(add_own_ratio(df, L), L)
    assert o["t"].tolist() == [86.0]


def test_passes_finds_the_car_crossing_the_crash_site() -> None:
    df = add_own_ratio(race(), L)
    got = passes(df, exclude={"1"}, crash_dist=500.0, t0=60.0, t1=80.0, length=L, names={"2": "TWO"})
    two = [p for p in got if p["car"] == "2"]
    # car 2 starts 100 m ahead at 50 m/s: it reaches 500 m of its lap at t = 8 + 20 k s
    assert [p["t"] for p in two] == [68.0] and two[0]["driver"] == "TWO"
    assert two[0]["pct_of_own_normal"] == 100.0

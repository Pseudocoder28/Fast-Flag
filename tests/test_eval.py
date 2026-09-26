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


def test_numbers_md_builds_from_the_committed_charts() -> None:
    from src.eval.numbers import build
    text = build()
    assert "replay of historical FastF1 data" in text
    for section in ("## Detection, headline", "## Latency from crash onset", "## Risk model", "## Case studies",
                    "## Escalation scorecard", "## Holdout"):
        assert section in text


def test_track_wide_state_follows_escalations_track_clear_and_resets() -> None:
    from src.eval.escalation import track_wide_at, track_wide_changes
    recs = [{"t": 10.0, "flag": "SC", "msector": 3, "message": "SAFETY CAR DEPLOYED"},
            {"t": 40.0, "flag": "CLEAR", "msector": 3, "message": "CLEAR IN TRACK SECTOR 3"},   # sector only: SC stays
            {"t": 80.0, "flag": "CLEAR", "msector": 3, "message": "TRACK CLEAR"},
            {"t": 100.0, "flag": "VSC", "msector": 7, "message": "VIRTUAL SAFETY CAR DEPLOYED"}]
    changes = track_wide_changes(recs, resets=[120.0])
    assert track_wide_at(changes, 5.0) == ("CLEAR", None, None)
    assert track_wide_at(changes, 50.0) == ("SC", 10.0, 3)
    assert track_wide_at(changes, 90.0)[0] == "CLEAR"
    assert track_wide_at(changes, 110.0) == ("VSC", 100.0, 7)
    assert track_wide_at(changes, 130.0)[0] == "CLEAR"        # an engine reset wipes the flag


def test_impact_rule_is_scored_against_always_sc() -> None:
    from src.eval.escalation import OFFICIAL_COLUMNS, impact_rule
    rows = [("SC", "IMPACT STOPPED"), ("SC", "STOPPED"), ("VSC", "STOPPED"), ("VSC", "")]
    off = pd.DataFrame([{**dict.fromkeys(OFFICIAL_COLUMNS), "official_flag": f, "onset_car": "1",
                         "onset_car_alerts": a} for f, a in rows])
    r = impact_rule(off)
    assert r["right_with_impact_rule"] == 3 and r["right_with_current_rule"] == 2

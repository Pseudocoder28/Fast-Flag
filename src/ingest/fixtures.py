"""Build the Section 7.6 fixtures from one training race around one real incident.

Race: 2023 Australian GP. Incident: car 23 (Albon) crashes at Turn 6 on lap 7,
marshal sector 9 (official YELLOW at t=4390, SC at t=4402, RED at t=4560).

- ticks and official events are real FastF1 data.
- detections, risk and recs are hand-built from the real timing, only so B can
  build against realistic shapes before the real detectors exist.

Run: python -m src.ingest.fixtures
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.holdout import assert_not_holdout, race_id
from src.ingest.merge import merge_session
from src.ingest.official import official_events
from src.ingest.reference import build_track_ref

YEAR, EVENT, LOCATION = 2023, "Australia", "Melbourne"
T_START, T_END = 4240.0, 4570.0          # ~150 s before the crash to just after the red flag
INCIDENT_DRV = "23"
OUT = Path("fixtures")


def num(v, nd: int = 1) -> float:
    return round(float(v), nd) if np.isfinite(v) else -1.0


def car_row(r) -> dict:
    return {"drv": r.drv, "x": num(r.x), "y": num(r.y), "dist": num(r.dist),
            "lat_off": num(r.lat_off, 2), "speed": num(r.speed), "throttle": num(r.throttle),
            "brake": bool(r.brake > 0), "gear": int(r.gear), "rpm": int(r.rpm),
            "msector": int(r.msector), "gap_ahead_m": num(r.gap_ahead_m),
            "in_pit": bool(r.in_pit)}


def build_ticks(df: pd.DataFrame) -> list[dict]:
    """One tick per grid time. Cars with no fresh data are left out of the tick."""
    ticks = []
    live = df.dropna(subset=["x", "y", "speed", "gear", "rpm", "dist"])
    for t, g in live.groupby("t", sort=True):
        ticks.append({"t": round(float(t), 2), "lap": int(g["lap"].iloc[0]),
                      "track_status": str(g["track_status"].iloc[0]),
                      "cars": [car_row(r) for r in g.itertuples()]})
    return ticks


def incident_times(df: pd.DataFrame) -> tuple[float, float, pd.Series]:
    """First sustained stop (<20 km/h for 2 s, on track), then the big speed loss
    (>80 km/h within 1 s) in the 5 s before it. Normal corner braking also loses
    80 km/h, so the impact is anchored to the stop."""
    car = df[df.drv == INCIDENT_DRV].set_index("t").sort_index()
    slow = ((car["speed"] < 20) & ~car["in_pit"]).astype(int).rolling(8).sum() == 8
    t_stop = float(slow[slow].index[0])
    loss = car["speed"].shift(4) - car["speed"]
    near = loss[(loss.index >= t_stop - 5.0) & (loss.index <= t_stop) & (loss > 80)]
    t_impact = float(near.index[0])
    return t_impact, t_stop, car


def build_detections(df: pd.DataFrame) -> list[dict]:
    t_imp, t_stop, car = incident_times(df)
    at_imp, at_stop = car.loc[t_imp], car.loc[t_stop]
    before = car.loc[t_imp - 1.0]
    ms = int(at_imp["msector"])
    return [
        {"id": "det-000001", "t": t_imp, "drivers": [INCIDENT_DRV], "msector": ms, "type": "IMPACT",
         "severity": 0.9, "evidence": f"speed {before.speed:.0f} -> {at_imp.speed:.0f} km/h in 1.0 s"},
        {"id": "det-000002", "t": t_imp + 0.25, "drivers": [INCIDENT_DRV], "msector": ms,
         "type": "ANOMALY", "severity": 0.8, "evidence": "anomaly score 0.81 (speed deviation, decel)"},
        {"id": "det-000003", "t": t_stop, "drivers": [INCIDENT_DRV], "msector": int(at_stop["msector"]),
         "type": "STOPPED", "severity": 0.85,
         "evidence": f"speed {at_stop.speed:.0f} km/h vs ref {at_stop.ref_speed:.0f} km/h"},
    ]


def build_recs(dets: list[dict]) -> list[dict]:
    imp, _, stop = dets
    ms = imp["msector"]
    return [
        {"id": "rec-000001", "t": imp["t"] + 0.25, "msector": ms, "flag": "YELLOW", "confidence": 0.7,
         "reason": f"car {INCIDENT_DRV} impact signature", "message": f"YELLOW IN TRACK SECTOR {ms}",
         "source_detections": [imp["id"]]},
        {"id": "rec-000002", "t": stop["t"] + 0.25, "msector": ms, "flag": "DOUBLE_YELLOW",
         "confidence": 0.85, "reason": f"car {INCIDENT_DRV} stopped on racing line after impact",
         "message": f"DOUBLE YELLOW IN TRACK SECTOR {ms}", "source_detections": [imp["id"], stop["id"]]},
        {"id": "rec-000003", "t": stop["t"] + 3.0, "msector": ms, "flag": "SC", "confidence": 0.75,
         "reason": f"high severity impact, car {INCIDENT_DRV} stopped, recovery needed",
         "message": "SAFETY CAR DEPLOYED", "source_detections": [imp["id"], stop["id"]]},
    ]


def build_risk(df: pd.DataFrame, t_impact: float, ms: int) -> list[dict]:
    """Every car once per second. Low noisy baseline, the crashed car high after impact,
    cars approaching the incident sector raised after impact. No fake precursor."""
    rng = np.random.default_rng(7)
    sec = df[(df.t % 1.0 == 0) & df.dist.notna()]
    out = []
    for r in sec.itertuples():
        r10 = float(rng.uniform(0.01, 0.08))
        feats = ["tyre_life", "battle_density"]
        if r.t >= t_impact and r.drv == INCIDENT_DRV:
            r10, feats = 0.97, ["speed_dev", "decel"]
        elif r.t >= t_impact and r.msector in (ms - 2, ms - 1, ms):
            r10, feats = float(rng.uniform(0.3, 0.55)), ["closing_rate", "speed_dev"]
        out.append({"t": round(float(r.t), 2), "drv": r.drv, "risk_10s": round(r10, 3),
                    "risk_30s": round(min(1.0, r10 * 1.6 + 0.02), 3), "top_features": feats})
    return out


def write_jsonl(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["t"]):
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def main() -> None:
    import fastf1
    assert_not_holdout(YEAR, LOCATION)
    fastf1.Cache.enable_cache(str(Path("data") / "fastf1_cache"))
    fastf1.set_log_level(logging.ERROR)
    s = fastf1.get_session(YEAR, EVENT, "R")
    s.load(weather=False)
    rid = race_id(YEAR, s.event["EventName"])
    ref = build_track_ref(s)
    df = merge_session(s, ref, T_START, T_END)
    OUT.mkdir(exist_ok=True)
    dets = build_detections(df)
    write_jsonl("ticks_sample.jsonl", build_ticks(df))
    write_jsonl("detections_sample.jsonl", dets)
    write_jsonl("recs_sample.jsonl", build_recs(dets))
    write_jsonl("risk_sample.jsonl", build_risk(df, dets[0]["t"], dets[0]["msector"]))
    write_jsonl("official_sample.jsonl", [e for e in official_events(s) if T_START <= e["t"] <= T_END])
    (OUT / "track_sample.json").write_text(json.dumps(ref.to_track_json(rid)), encoding="utf-8")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:<26} {p.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()

"""Build the Section 7.6 fixtures from one training race around one real incident.

Race: 2023 Australian GP. Incident: car 23 (Albon) crashes at Turn 6 on lap 7,
marshal sector 9 (official YELLOW at t=4390, SC at t=4402, RED at t=4560).

- ticks and official events are real FastF1 data.
- detections and recs are hand-built from the real timing, only so B can build
  against realistic shapes before the real detectors exist.
- risk comes from the real risk model (see real_risk). It replaced hand-built values
  that all sat above the dashboard's high-risk line, so every car looked high risk on
  the mock.

Run: python -m src.ingest.fixtures                (everything, from the FastF1 cache)
     python -m src.ingest.fixtures --risk-only    (only risk_sample.jsonl, from data/features and data/models)
"""

from __future__ import annotations

import argparse
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
RID = "2023_Australian"
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


def real_risk(t0: float = T_START, t1: float = T_END) -> list[dict]:
    """One row per racing car per second, in the real server's shape. The values come
    from the leave-one-race-out risk model, which never trained on this race
    (data/features/<race>_risk_pred.parquet, from python -m src.predict.risk). The
    production model did train on it and shows car 23 high 30 s before its crash, which
    is memory, not prediction, so it must not appear on the mock. top_features: the
    production model's explanation for the same row, as the real server streams it.
    Rows the leave-one-race-out set has no forecast for (a retired car, a car after its
    crash) are left out."""
    from src.predict.pipeline import risk_processors
    from src.replay.engine import Engine, RaceData
    race = RaceData.load(RID)
    procs = risk_processors(race)
    if not procs:
        raise SystemExit("no risk models in data/models: train them first (python -m src.predict.risk train)")
    eng = Engine(race, procs)
    eng.seek(t0)
    live = [e["data"] for e in eng.advance(t1) if e["kind"] == "risk"]
    loro = pd.read_parquet(Path("data") / "features" / f"{RID}_risk_pred.parquet", columns=["t", "drv", "p10", "p30"])
    honest = {(round(float(t), 2), str(d)): (p10, p30) for t, d, p10, p30 in loro.itertuples(index=False)}
    out = []
    for r in live:
        if (r["t"], r["drv"]) not in honest:
            continue
        p10, p30 = honest[(r["t"], r["drv"])]
        out.append({**r, "risk_10s": round(float(p10), 4), "risk_30s": round(float(max(p30, p10)), 4)})
    return out


def write_jsonl(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["t"]):
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Section 7.6 fixtures")
    ap.add_argument("--risk-only", action="store_true",
                   help="only rebuild risk_sample.jsonl, from data/features and data/models")
    a = ap.parse_args()
    assert_not_holdout(YEAR, LOCATION)
    OUT.mkdir(exist_ok=True)
    if a.risk_only:
        write_jsonl("risk_sample.jsonl", real_risk())
        print(f"risk_sample.jsonl          {(OUT / 'risk_sample.jsonl').stat().st_size / 1e6:6.2f} MB")
        return
    import fastf1
    fastf1.Cache.enable_cache(str(Path("data") / "fastf1_cache"))
    fastf1.set_log_level(logging.ERROR)
    s = fastf1.get_session(YEAR, EVENT, "R")
    s.load(weather=False)
    rid = race_id(YEAR, s.event["EventName"])
    ref = build_track_ref(s)
    df = merge_session(s, ref, T_START, T_END)
    dets = build_detections(df)
    write_jsonl("ticks_sample.jsonl", build_ticks(df))
    write_jsonl("detections_sample.jsonl", dets)
    write_jsonl("recs_sample.jsonl", build_recs(dets))
    write_jsonl("risk_sample.jsonl", real_risk())
    write_jsonl("official_sample.jsonl", [e for e in official_events(s) if T_START <= e["t"] <= T_END])
    (OUT / "track_sample.json").write_text(json.dumps(ref.to_track_json(rid)), encoding="utf-8")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:<26} {p.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()

"""A6 risk dataset: labels and features per training race (Section 6.4).

Labels (training races only, never the holdout):
- Run the tuned detectors over the race. A car's incident time is its earliest
  IMPACT, STOPPED or SPIN detection that matches an official incident (the same
  matching as src.eval.incidents: sector rule, [t_official - 60 s, + 10 s]).
- y10 / y20 / y30 = 1 for that car's ticks in [t_incident - H, t_incident).
- The car's ticks from its incident on are dropped (Section 6.4).
- Ambiguous ticks are dropped: cars in a matching sector in the 30 s before an
  official incident that no detection attributed to a car. We cannot tell which
  car it was, so they are neither positive nor safely negative.
- Only racing ticks (src.predict.features.racing_rows).

Features: src.predict.features, with the ANOMALY score from a model trained on the
OTHER training races (leave-one-race-out), so no race is scored by a model that saw it.

Output: data/features/<race>_risk.parquet with every kept tick: features (float32),
labels, t, drv, and train_keep (all positives + a 3% sample of negatives).

Run: python -m src.predict.dataset                 (all training races)
     python -m src.predict.dataset 2023_Australian
"""

from __future__ import annotations

import argparse
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.detect.anomaly import attach_scores, train as train_anomaly
from src.detect.detectors import DetectorSuite
from src.detect.pipeline import load_config
from src.eval.incidents import alert_matches, build_incidents, race_files, suspended_times
from src.ingest.holdout import assert_not_holdout
from src.ingest.sectors import sector_matches
from src.predict.features import feature_matrix, racing_rows
from src.replay.engine import Engine, RaceData, available_races

FEATURES_DIR = Path("data/features")
HORIZONS = (10, 20, 30)
CRASH_TYPES = {"IMPACT", "STOPPED", "SPIN"}
AMBIGUOUS_S = 30.0
NEG_SAMPLE = 0.03


def detections(race: RaceData) -> list[dict]:
    n = len(race.track.get("msectors", [])) or None
    eng = Engine(race, [DetectorSuite(load_config(), n_sectors=n)])
    return [e["data"] for e in eng.advance(eng.t_end) if e["kind"] == "detection"]


def incident_times(race: RaceData, dets: list[dict]) -> tuple[dict[str, float], list, int]:
    """{car: first matched crash-type detection time}, unattributed incidents, n sectors."""
    official, meta = race_files(race.race)
    n = meta["n_msectors"]
    incidents, _ = build_incidents(official, meta, suspended_times(race.frame), n)
    first: dict[str, float] = {}
    unattributed = []
    for inc in incidents:
        hits = [d for d in dets if d["type"] in CRASH_TYPES and alert_matches(inc, d["t"], d["msector"], n)]
        if not hits:
            unattributed.append(inc)
        for d in hits:
            drv = d["drivers"][0]
            first[drv] = min(first.get(drv, np.inf), d["t"])
    return first, unattributed, n


def label(df: pd.DataFrame, first: dict[str, float], unattributed: list, n: int) -> pd.DataFrame:
    t_inc = df["drv"].map(first).astype(float)          # NaN for cars with no incident
    keep = t_inc.isna() | (df["t"] < t_inc)             # drop the car's ticks from its incident on
    ambiguous = np.zeros(len(df), dtype=bool)
    for inc in unattributed:
        w = (df["t"] >= inc.t - AMBIGUOUS_S) & (df["t"] <= inc.t)
        if inc.sectors:
            w &= df["msector"].apply(lambda m, secs=tuple(inc.sectors): any(sector_matches(int(m), s, n) for s in secs))
        ambiguous |= w.to_numpy()
    out = df.loc[keep.to_numpy() & ~ambiguous].copy()
    t_inc = out["drv"].map(first).astype(float)
    for h in HORIZONS:
        out[f"y{h}"] = ((out["t"] >= t_inc - h) & (out["t"] < t_inc)).astype("int8")
    return out


def build(rid: str) -> dict:
    assert_not_holdout(rid=rid)
    race = RaceData.load(rid)
    others = [r for r in available_races() if r != rid]
    attach_scores(race.frame, train_anomaly(others))       # ANOMALY trained without this race
    dets = detections(race)
    first, unattributed, n = incident_times(race, dets)
    df = race.frame[racing_rows(race.frame)]
    df = label(df, first, unattributed, n)
    x = feature_matrix(df)
    rng = np.random.default_rng(zlib.crc32(rid.encode()))
    pos = df[f"y{max(HORIZONS)}"].to_numpy() == 1
    x["train_keep"] = pos | (rng.random(len(df)) < NEG_SAMPLE)
    for c in ["t", "drv", "msector"] + [f"y{h}" for h in HORIZONS]:
        x[c] = df[c].to_numpy()
    x.to_parquet(FEATURES_DIR / f"{rid}_risk.parquet", index=False)
    return {"race": rid, "rows": len(x), "cars_with_incident": len(first), "unattributed": len(unattributed),
            **{f"pos{h}": int(x[f"y{h}"].sum()) for h in HORIZONS}}


def main() -> None:
    p = argparse.ArgumentParser(description="Build risk datasets for training races")
    p.add_argument("races", nargs="*")
    a = p.parse_args()
    races = a.races or available_races()
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        rows = list(ex.map(build, races))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"total: {df['rows'].sum()} ticks, {df['cars_with_incident'].sum()} car incidents, "
          f"positives y10 {df['pos10'].sum()}, y30 {df['pos30'].sum()}")


if __name__ == "__main__":
    main()

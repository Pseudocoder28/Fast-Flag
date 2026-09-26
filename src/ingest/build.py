"""A1: build per-race features and sidecar files from the FastF1 cache.

Per race: reference line, speed profile and marshal sectors from the same
weekend's qualifying (before the race, so no leakage), the race merged onto the
250 ms grid from lights out to the chequered flag, engineered features, and
everything the real server needs, so nothing downstream reloads raw FastF1 data.

Outputs for training races, in data/features/:
  <race>.parquet          one row per (t, drv)
  <race>_ref.json         full-resolution reference (TrackRef.load)
  <race>_track.json       GET /track shape
  <race>_official.json    official events (Section 7.5 shape)
  <race>_meta.json        race window, lap length, sanity checks
  data/plots/<race>_track.png

Run: python -m src.ingest.build 2023_Australian 2026_Dutch
     python -m src.ingest.build --top 20
     python -m src.ingest.build --holdout 2026_Azerbaijan   (A7 only, writes to data/holdout/)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import fastf1
import pandas as pd

from src.ingest.features import NEUTRAL_STATUS, add_features
from src.ingest.holdout import assert_not_holdout, is_holdout_id, race_id
from src.ingest.merge import merge_session, race_window
from src.ingest.official import official_events
from src.ingest.plot_track import PLOTS, plot_track
from src.ingest.reference import build_track_ref, msectors_in_order
from src.ingest.scan import CACHE_DIR, OUT_CSV as RANKING_CSV

FEATURES = Path("data/features")
HOLDOUT_DIR = Path("data/holdout")
END_MARGIN_S = 10.0
SLOW_RATIO = 0.6          # sanity check: a car below 60% of reference speed counts as slow


def resolve(rid: str) -> tuple[int, int, str]:
    """(year, round, location) for a race id like 2023_Australian."""
    year = int(rid.split("_")[0])
    sched = fastf1.get_event_schedule(year, include_testing=False)
    for _, ev in sched.iterrows():
        if race_id(year, ev["EventName"]) == rid:
            return year, int(ev["RoundNumber"]), str(ev["Location"])
    raise KeyError(f"unknown race id {rid}")


def sector_offset(car: int, flagged: int, n: int) -> int:
    """Circular offset from the flagged marshal sector to the car's sector."""
    return (car - flagged + n // 2) % n - n // 2


def flag_matches(car: int, flagged: int, n: int) -> bool:
    """Race control also flags the sectors before an incident (e.g. car stopped in
    sector 18, double yellows in 16 and 17), so the car may be up to 2 sectors
    downstream of the flagged sector, or 1 upstream."""
    return -1 <= sector_offset(car, flagged, n) <= 2


def flag_alignment(df: pd.DataFrame, official: list[dict], n_sectors: int) -> tuple[int, int]:
    """Ingest sanity check only (never fed to detect or predict): how many official
    sector yellows during normal running had a slow car in a matching marshal
    sector in the 20 s before the message. Skips neutralised or suspended running
    and standing starts, where cars are slow by design."""
    race = df.drop_duplicates("t").set_index("t")[["track_status", "suspended", "field_slow"]]
    slow = df[(df["speed_ratio"] < SLOW_RATIO) & ~df["in_pit"] & (df["msector"] > 0)]
    hits = total = 0
    for e in official:
        if e["flag"] not in ("YELLOW", "DOUBLE_YELLOW") or e["msector"] is None:
            continue
        near = race.loc[(race.index >= e["t"] - 1) & (race.index <= e["t"])]
        if not len(near):
            continue
        now = near.iloc[-1]
        if now["track_status"] in NEUTRAL_STATUS or now["suspended"] or now["field_slow"] >= 0.5:
            continue
        total += 1
        w = slow[(slow["t"] >= e["t"] - 20) & (slow["t"] <= e["t"] + 2)]
        hits += any(flag_matches(int(m), e["msector"], n_sectors) for m in w["msector"].unique())
    return hits, total


def load(year: int, rnd: int, kind: str):
    s = fastf1.get_session(year, rnd, kind)
    s.load(weather=(kind == "R"), messages=(kind == "R"))
    return s


def build_race(rid: str, holdout: bool = False) -> dict:
    year, rnd, location = resolve(rid)
    if holdout:
        if not is_holdout_id(rid):
            raise ValueError(f"{rid} is not a holdout race")
        out_dir = HOLDOUT_DIR
    else:
        assert_not_holdout(year, location, rid=rid)
        out_dir = FEATURES
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref = build_track_ref(load(year, rnd, "Q"))
    race = load(year, rnd, "R")
    t_start, t_end = race_window(race)
    df = add_features(merge_session(race, ref, t_start, t_end + END_MARGIN_S), race)
    df.insert(0, "race", rid)
    df.to_parquet(out_dir / f"{rid}.parquet", index=False)
    official = official_events(race)
    hits, total = flag_alignment(df, [e for e in official if t_start <= e["t"] <= t_end], len(ref.msectors))
    meta = {"race": rid, "year": year, "round": rnd, "location": location,
            "t_start": round(t_start, 2), "t_end": round(t_end, 2), "lap_length_m": round(ref.length, 1),
            "n_msectors": len(ref.msectors), "msectors_in_order": msectors_in_order(ref.msectors),
            "reference_source": "qualifying", "rows": len(df), "cars": int(df["drv"].nunique()),
            "flag_alignment": f"{hits}/{total}"}
    ref.save(out_dir / f"{rid}_ref.json")
    (out_dir / f"{rid}_track.json").write_text(json.dumps(ref.to_track_json(rid)))
    (out_dir / f"{rid}_official.json").write_text(json.dumps(official))
    (out_dir / f"{rid}_meta.json").write_text(json.dumps(meta, indent=2))
    plot_track(ref, f"{rid} (reference from qualifying, {ref.length:.0f} m)", PLOTS / f"{rid}_track.png")
    meta["seconds"] = round(time.time() - t0)
    return meta


def top_races(n: int) -> list[str]:
    df = pd.read_csv(RANKING_CSV, keep_default_na=False)
    return df[df["error"] == ""].sort_values("score", ascending=False)["race"].head(n).tolist()


def main() -> None:
    p = argparse.ArgumentParser(description="Build per-race features (A1)")
    p.add_argument("races", nargs="*", help="race ids, e.g. 2023_Australian")
    p.add_argument("--top", type=int, help="build the top N races of the ranking")
    p.add_argument("--holdout", help="build one holdout race into data/holdout/ (A7 only)")
    a = p.parse_args()
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level(logging.ERROR)
    if a.holdout:
        jobs, holdout = [a.holdout], True
    else:
        jobs, holdout = (a.races or []) + (top_races(a.top) if a.top else []), False
    for rid in dict.fromkeys(jobs):
        try:
            m = build_race(rid, holdout=holdout)
            print(f"{rid:<22} {m['rows']:>8} rows  {m['cars']} cars  lap {m['lap_length_m']:>6.0f} m  "
                  f"sectors {m['n_msectors']} in order={m['msectors_in_order']}  "
                  f"flag alignment {m['flag_alignment']:>5}  {m['seconds']} s", flush=True)
        except Exception as e:  # keep going, report
            print(f"{rid:<22} FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

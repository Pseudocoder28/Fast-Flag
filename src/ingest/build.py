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
     python -m src.ingest.build --case 2021_Azerbaijan    (case study race: data/case_studies/, never training)
     python -m src.ingest.build --holdout 2026_Azerbaijan   (A7 only, writes to data/holdout/)
     python -m src.ingest.build 2026_Dutch --official-only  (rewrite only the official events)
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
from src.ingest.sectors import sector_matches

FEATURES = Path("data/features")
HOLDOUT_DIR = Path("data/holdout")
CASE_DIR = Path("data/case_studies")
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
        hits += any(sector_matches(int(m), e["msector"], n_sectors) for m in w["msector"].unique())
    return hits, total


def load(year: int, rnd: int, kind: str):
    s = fastf1.get_session(year, rnd, kind)
    s.load(weather=(kind == "R"), messages=(kind == "R"))
    return s


def output_dir(rid: str, year: int, location: str, holdout: bool, case: bool) -> Path:
    if holdout:
        if not is_holdout_id(rid):
            raise ValueError(f"{rid} is not a holdout race")
        return HOLDOUT_DIR
    assert_not_holdout(year, location, rid=rid)
    return CASE_DIR if case else FEATURES


def build_race(rid: str, holdout: bool = False, case: bool = False) -> dict:
    year, rnd, location = resolve(rid)
    out_dir = output_dir(rid, year, location, holdout, case)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref = build_track_ref(load(year, rnd, "Q"))
    race = load(year, rnd, "R")
    t_start, t_end = race_window(race)
    df = add_features(merge_session(race, ref, t_start, t_end + END_MARGIN_S), race, ref)
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
    (out_dir / f"{rid}_track.json").write_text(json.dumps(ref.to_track_json(rid)), encoding="utf-8")
    (out_dir / f"{rid}_official.json").write_text(json.dumps(official), encoding="utf-8")
    (out_dir / f"{rid}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot_track(ref, f"{rid} (reference from qualifying, {ref.length:.0f} m)", PLOTS / f"{rid}_track.png")
    meta["seconds"] = round(time.time() - t0)
    return meta


def rebuild_official(rid: str, holdout: bool = False, case: bool = False) -> int:
    """Rewrite only <race>_official.json (after a fix to src.ingest.official), leaving
    the features and every other sidecar file as they are. Returns the event count."""
    year, rnd, location = resolve(rid)
    path = output_dir(rid, year, location, holdout, case) / f"{rid}_official.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: build the race first")
    official = official_events(load(year, rnd, "R"))
    path.write_text(json.dumps(official), encoding="utf-8")
    return len(official)


def top_races(n: int) -> list[str]:
    df = pd.read_csv(RANKING_CSV, keep_default_na=False, encoding="utf-8")
    return df[df["error"] == ""].sort_values("score", ascending=False)["race"].head(n).tolist()


def main() -> None:
    p = argparse.ArgumentParser(description="Build per-race features (A1)")
    p.add_argument("races", nargs="*", help="race ids, e.g. 2023_Australian")
    p.add_argument("--top", type=int, help="build the top N races of the ranking")
    p.add_argument("--holdout", help="build one holdout race into data/holdout/ (A7 only)")
    p.add_argument("--case", help="build one case-study race into data/case_studies/ (not a training race)")
    p.add_argument("--official-only", action="store_true",
                   help="only rewrite <race>_official.json of races that are already built")
    a = p.parse_args()
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level(logging.ERROR)
    case = bool(a.case)
    if a.holdout:
        jobs, holdout = [a.holdout], True
    elif a.case:
        jobs, holdout = [a.case], False
    else:
        jobs, holdout = (a.races or []) + (top_races(a.top) if a.top else []), False
    for rid in dict.fromkeys(jobs):
        if a.official_only:
            print(f"{rid:<22} {rebuild_official(rid, holdout=holdout, case=case)} official events rewritten",
                  flush=True)
            continue
        try:
            m = build_race(rid, holdout=holdout, case=case)
            print(f"{rid:<22} {m['rows']:>8} rows  {m['cars']} cars  lap {m['lap_length_m']:>6.0f} m  "
                  f"sectors {m['n_msectors']} in order={m['msectors_in_order']}  "
                  f"flag alignment {m['flag_alignment']:>5}  {m['seconds']} s", flush=True)
        except Exception as e:  # keep going, report
            print(f"{rid:<22} FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

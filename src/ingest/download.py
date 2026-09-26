"""Full download of the top incident-heavy races into the FastF1 cache (A1, step 1).

Downloads the race (laps, telemetry, weather, messages) and the same weekend's
qualifying (laps, telemetry). Qualifying happens before the race, so reference
lines and speed profiles built from it cannot leak future race data.

Run: python -m src.ingest.download            (top 20 of data/race_ranking.csv)
     python -m src.ingest.download --top 25
Never touches the holdout. Safe to re-run: cached sessions load from disk.
"""

from __future__ import annotations

import argparse
import logging
import time

import fastf1
import pandas as pd

from src.ingest.holdout import assert_not_holdout
from src.ingest.scan import CACHE_DIR, OUT_CSV as RANKING_CSV


def top_races(n: int) -> pd.DataFrame:
    df = pd.read_csv(RANKING_CSV, keep_default_na=False)
    return df[df["error"] == ""].sort_values("score", ascending=False).head(n)


def load_session(year: int, rnd: int, kind: str) -> int:
    s = fastf1.get_session(year, rnd, kind)
    if kind == "R":
        s.load()
    else:
        s.load(weather=False, messages=False)
    return len(s.laps)


def download(row) -> str:
    assert_not_holdout(int(row.year), row.location, rid=row.race)
    t0 = time.time()
    n_race = load_session(int(row.year), int(row.round), "R")
    n_quali = load_session(int(row.year), int(row.round), "Q")
    return f"{row.race:<24} race {n_race:>5} laps, quali {n_quali:>4} laps, {time.time() - t0:5.0f} s"


def main() -> None:
    p = argparse.ArgumentParser(description="Full download of the top ranked races")
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args()
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level(logging.WARNING)
    for row in top_races(a.top).itertuples():
        try:
            print(download(row), flush=True)
        except Exception as e:  # keep going, report at the end
            print(f"{row.race:<24} FAILED {type(e).__name__}: {e}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

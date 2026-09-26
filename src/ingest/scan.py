"""Cheap scan: load race control messages only for every race 2023 to 2026,
count incident events and rank races. Holdout races are skipped.

Run: python -m src.ingest.scan
Output: data/race_ranking.csv (rewritten after every race, so partial
results are usable while it runs). Re-running skips races already scanned.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import pandas as pd

from src.ingest.holdout import is_holdout, race_id

CACHE_DIR = Path("data/fastf1_cache")
OUT_CSV = Path("data/race_ranking.csv")
YEARS = [2023, 2024, 2025, 2026]
COLUMNS = ["race", "year", "round", "event", "location", "yellow",
           "double_yellow", "sc", "vsc", "red", "score", "error"]


def count_events(msgs: pd.DataFrame) -> dict[str, int]:
    """Count incident events in a race control message table."""
    flag = msgs.get("Flag", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    cat = msgs.get("Category", pd.Series(dtype=str)).fillna("").astype(str)
    text = msgs.get("Message", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    deployed = (cat == "SafetyCar") & text.str.contains("DEPLOYED") & ~text.str.contains("ENDING")
    return {
        "yellow": int((flag == "YELLOW").sum()),
        "double_yellow": int((flag == "DOUBLE YELLOW").sum()),
        "sc": int((deployed & ~text.str.contains("VIRTUAL")).sum()),
        "vsc": int((deployed & text.str.contains("VIRTUAL")).sum()),
        "red": int((flag == "RED").sum()),
    }


def score(c: dict[str, int]) -> float:
    """Weight rarer, bigger events higher."""
    return c["yellow"] + 2 * c["double_yellow"] + 5 * c["vsc"] + 8 * c["sc"] + 10 * c["red"]


def past_races(year: int) -> pd.DataFrame:
    sched = fastf1.get_event_schedule(year, include_testing=False)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return sched[sched["EventDate"] < today]


def scan_race(year: int, row: pd.Series) -> dict:
    rid = race_id(year, row["EventName"])
    rec = {"race": rid, "year": year, "round": int(row["RoundNumber"]),
           "event": row["EventName"], "location": row["Location"], "error": ""}
    try:
        s = fastf1.get_session(year, int(row["RoundNumber"]), "R")
        s.load(laps=False, telemetry=False, weather=False, messages=True)
        c = count_events(s.race_control_messages)
        rec.update(c, score=score(c))
    except Exception as e:  # keep scanning, record the failure
        rec.update({k: 0 for k in ["yellow", "double_yellow", "sc", "vsc", "red"]},
                   score=0.0, error=f"{type(e).__name__}: {e}"[:200])
    return rec


def load_existing() -> pd.DataFrame:
    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV, keep_default_na=False)
        return df[df["error"] == ""]
    return pd.DataFrame(columns=COLUMNS)


def save(rows: list[dict]) -> None:
    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("score", ascending=False)
    df.to_csv(OUT_CSV, index=False)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level(logging.WARNING)
    existing = load_existing()
    rows = existing.to_dict("records")
    done = set(existing["race"])
    for year in YEARS:
        for _, row in past_races(year).iterrows():
            if is_holdout(year, row["Location"]):
                print(f"skip holdout {year} {row['Location']}")
                continue
            if race_id(year, row["EventName"]) in done:
                continue
            rec = scan_race(year, row)
            rows.append(rec)
            save(rows)
            print(f"{rec['race']:<28} score={rec['score']:>5} {rec['error']}", flush=True)
    save(rows)
    print(f"done, {len(rows)} races -> {OUT_CSV}")


if __name__ == "__main__":
    main()

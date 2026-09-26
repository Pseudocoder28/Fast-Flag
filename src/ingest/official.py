"""Official race control events (Section 6.7) with SessionTime in seconds.

Kept: Flag in {YELLOW, DOUBLE YELLOW, RED, CLEAR (sector or track)} and
SC/VSC deployment messages. Ignored: BLUE, track limits, DRS, investigations.
Used only by eval and by the dashboard timeline. Never fed to detect/predict.

Run: python -m src.ingest.official 2023 Australia
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

FLAG_MAP = {"YELLOW": "YELLOW", "DOUBLE YELLOW": "DOUBLE_YELLOW", "RED": "RED", "CLEAR": "CLEAR"}


def to_session_time(session, times: pd.Series) -> pd.Series:
    """Race control messages carry UTC wall time; convert to SessionTime seconds."""
    return (pd.to_datetime(times) - session.t0_date).dt.total_seconds()


def classify(row: pd.Series) -> str | None:
    msg = str(row.get("Message") or "").upper()
    if row.get("Category") == "SafetyCar":
        if "DEPLOYED" in msg and "ENDING" not in msg:
            # 2023 to 2025: "VIRTUAL SAFETY CAR DEPLOYED"; 2026: "VSC DEPLOYED"
            return "VSC" if "VIRTUAL" in msg or re.search(r"\bVSC\b", msg) else "SC"
        return None
    flag = str(row.get("Flag") or "").upper()
    return FLAG_MAP.get(flag)


def drivers_in(msg: str) -> list[str]:
    return re.findall(r"CARS? (\d+)", msg) + re.findall(r"AND (\d+) \(", msg)


def official_events(session) -> list[dict]:
    msgs = session.race_control_messages
    out = []
    for t, (_, row) in zip(to_session_time(session, msgs["Time"]), msgs.iterrows()):
        flag = classify(row)
        if flag is None:
            continue
        sector = row.get("Sector")
        out.append({
            "t": round(float(t), 2),
            "category": str(row.get("Category")),
            "message": str(row.get("Message")),
            "flag": flag,
            "scope": str(row.get("Scope") or "Track"),
            "msector": int(sector) if pd.notna(sector) else None,
            "drivers": [str(row["RacingNumber"])] if pd.notna(row.get("RacingNumber")) else drivers_in(str(row.get("Message"))),
        })
    return sorted(out, key=lambda e: e["t"])


def main() -> None:
    import fastf1
    year, event = int(sys.argv[1]), sys.argv[2]
    fastf1.Cache.enable_cache(str(Path("data") / "fastf1_cache"))
    s = fastf1.get_session(year, event, "R")
    s.load(weather=False)
    for e in official_events(s):
        print(f"{e['t']:9.1f}  {e['flag']:<14} sector={e['msector']}  {e['message']}")


if __name__ == "__main__":
    main()

"""Merge car_data + pos_data per driver onto a common 250 ms SessionTime grid.

Strictly causal: the value at grid time t is the latest sample with
SessionTime <= t (as-of join). We never interpolate towards a future sample
and never extrapolate: a car whose latest sample is older than MAX_STALE_S
gets NaN (that gap is what the DROPOUT detector looks for).

Run: python -m src.ingest.merge 2023 Australia   (prints a summary of the first 5 minutes)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.ingest.official import to_session_time
from src.ingest.reference import POS_SCALE, TrackRef, secs, valid_pos

GRID_S = 0.25
MAX_STALE_S = 1.0
CAR_COLS = {"Speed": "speed", "Throttle": "throttle", "Brake": "brake", "nGear": "gear", "RPM": "rpm"}


def asof(grid: np.ndarray, t: np.ndarray, values: np.ndarray,
         max_stale: float = MAX_STALE_S) -> np.ndarray:
    """Latest value with sample time <= grid time, NaN if none or older than max_stale."""
    order = np.argsort(t, kind="stable")
    t, values = t[order], values[order].astype(float)
    idx = np.searchsorted(t, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    stale = np.ones(len(grid), dtype=bool)
    stale[ok] = (grid[ok] - t[idx[ok]]) > max_stale
    out[stale] = np.nan
    return out


def driver_frame(session, drv: str, grid: np.ndarray) -> pd.DataFrame:
    car = session.car_data[drv]
    pos = valid_pos(session.pos_data[drv])
    ct, pt = secs(car["SessionTime"]), secs(pos["SessionTime"])
    df = pd.DataFrame({"t": grid, "drv": drv})
    for src, dst in CAR_COLS.items():
        df[dst] = asof(grid, ct, car[src].to_numpy(float))
    df["x"] = asof(grid, pt, pos["X"].to_numpy(float)) * POS_SCALE
    df["y"] = asof(grid, pt, pos["Y"].to_numpy(float)) * POS_SCALE
    return df


def pit_intervals(laps: pd.DataFrame, drv: str) -> list[tuple[float, float]]:
    """(pit entry, pit exit) SessionTime pairs for one driver."""
    dl = laps[laps["DriverNumber"] == drv].sort_values("LapNumber")
    ins = dl["PitInTime"].dropna().dt.total_seconds().tolist()
    outs = dl["PitOutTime"].dropna().dt.total_seconds().tolist()
    pairs = []
    for t_in in ins:
        later = [o for o in outs if o > t_in]
        pairs.append((t_in, later[0] if later else np.inf))
    return pairs


def in_pit_mask(t: np.ndarray, pairs: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(len(t), dtype=bool)
    for a, b in pairs:
        m |= (t >= a) & (t < b)
    return m


def lap_numbers(laps: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Race lap at time t = leader's lap (highest lap started by t)."""
    starts = laps.dropna(subset=["LapStartTime"])
    st = starts["LapStartTime"].dt.total_seconds().to_numpy(float)
    ln = starts["LapNumber"].to_numpy(int)
    order = np.argsort(st)
    st, ln = st[order], np.maximum.accumulate(ln[order])
    idx = np.searchsorted(st, grid, side="right") - 1
    return np.where(idx >= 0, ln[np.clip(idx, 0, None)], 1)


def track_status_at(session, grid: np.ndarray) -> np.ndarray:
    ts = session.track_status
    t = ts["Time"].dt.total_seconds().to_numpy(float)
    idx = np.searchsorted(t, grid, side="right") - 1
    codes = ts["Status"].astype(str).to_numpy()
    return np.where(idx >= 0, codes[np.clip(idx, 0, None)], "1")


def race_window(session) -> tuple[float, float]:
    """(lights out, chequered flag) in SessionTime. Lap 1 starts at lights out."""
    lap1 = session.laps.loc[session.laps["LapNumber"] == 1, "LapStartTime"].dropna()
    t_start = float(lap1.dt.total_seconds().min())
    msgs = session.race_control_messages
    cheq = msgs[msgs["Message"].astype(str).str.upper().str.contains("CHEQUERED FLAG")]
    if len(cheq):
        t_end = float(to_session_time(session, cheq["Time"]).iloc[0])
    else:
        t_end = float(session.laps["Time"].dt.total_seconds().max())
    return t_start, t_end


def add_gaps(df: pd.DataFrame, length: float) -> pd.DataFrame:
    """Metres to the next car physically ahead / behind on track (pit cars excluded).

    Adds gap_ahead_m, gap_behind_m and ahead_drv. NaN when fewer than 2 cars are on track.
    """
    for col in ("gap_ahead_m", "gap_behind_m"):
        df[col] = np.nan
    df["ahead_drv"] = None
    on = df["dist"].notna() & ~df["in_pit"]
    d = df.loc[on, ["t", "drv", "dist"]].sort_values(["t", "dist"])
    g = d.groupby("t", sort=False)
    size = g["dist"].transform("size")
    nxt = g["dist"].shift(-1).fillna(g["dist"].transform("first"))
    nxt_drv = g["drv"].shift(-1).fillna(g["drv"].transform("first"))
    prv = g["dist"].shift(1).fillna(g["dist"].transform("last"))
    many = size >= 2
    idx = d.index[many]
    df.loc[idx, "gap_ahead_m"] = ((nxt - d["dist"]) % length)[many].to_numpy()
    df.loc[idx, "gap_behind_m"] = ((d["dist"] - prv) % length)[many].to_numpy()
    df.loc[idx, "ahead_drv"] = nxt_drv[many].to_numpy()
    return df


def merge_session(session, ref: TrackRef, t_start: float | None = None,
                  t_end: float | None = None) -> pd.DataFrame:
    """Long frame, one row per (t, drv), plus race-level lap and track_status."""
    drivers = [d for d in session.drivers if d in session.car_data and d in session.pos_data]
    lo = min(secs(session.pos_data[d]["SessionTime"]).min() for d in drivers)
    hi = max(secs(session.pos_data[d]["SessionTime"]).max() for d in drivers)
    lo = max(lo, t_start) if t_start is not None else lo
    hi = min(hi, t_end) if t_end is not None else hi
    grid = np.round(np.arange(np.ceil(lo / GRID_S) * GRID_S, hi, GRID_S), 2)
    frames = []
    for drv in drivers:
        f = driver_frame(session, drv, grid)
        f["in_pit"] = in_pit_mask(grid, pit_intervals(session.laps, drv))
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["dist"], df["lat_off"] = ref.project(df["x"].to_numpy(), df["y"].to_numpy())
    # timing-line pit flags miss the entry / exit roads, pit lane starts and some stops
    df["in_pit"] = df["in_pit"] | ref.near_pit(df["x"].to_numpy(), df["y"].to_numpy(), df["lat_off"].to_numpy())
    df["msector"] = ref.msector_of(df["dist"].to_numpy())
    df["ref_speed"] = ref.ref_speed(df["dist"].to_numpy())
    race = pd.DataFrame({"t": grid, "lap": lap_numbers(session.laps, grid),
                         "track_status": track_status_at(session, grid)})
    df = df.merge(race, on="t", how="left").sort_values(["t", "drv"]).reset_index(drop=True)
    return add_gaps(df, ref.length)


def main() -> None:
    import fastf1

    from src.ingest.reference import build_track_ref
    year, event = int(sys.argv[1]), sys.argv[2]
    fastf1.Cache.enable_cache("data/fastf1_cache")
    q = fastf1.get_session(year, event, "Q")
    q.load(weather=False, messages=False)
    s = fastf1.get_session(year, event, "R")
    s.load(weather=False)
    t0, _ = race_window(s)
    df = merge_session(s, build_track_ref(q, with_profile=False), t0, t0 + 300)
    print(df.describe().T[["count", "mean", "min", "max"]])


if __name__ == "__main__":
    main()

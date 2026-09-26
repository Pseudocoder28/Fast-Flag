"""Merge car_data + pos_data per driver onto a common 250 ms SessionTime grid.

Strictly causal: the value at grid time t is the latest sample with
SessionTime <= t (as-of join). We never interpolate towards a future sample
and never extrapolate: a car whose latest sample is older than MAX_STALE_S
gets NaN (that gap is what the DROPOUT detector looks for).

Run: python -m src.ingest.merge 2023 Australia   (prints a summary)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.ingest.reference import POS_SCALE, TrackRef

GRID_S = 0.25
MAX_STALE_S = 1.0
CAR_COLS = {"Speed": "speed", "Throttle": "throttle", "Brake": "brake", "nGear": "gear", "RPM": "rpm"}


def secs(td: pd.Series) -> np.ndarray:
    return td.dt.total_seconds().to_numpy(float)


def asof(grid: np.ndarray, t: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Latest value with sample time <= grid time, NaN if none or too old."""
    order = np.argsort(t, kind="stable")
    t, values = t[order], values[order].astype(float)
    idx = np.searchsorted(t, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    stale = np.ones(len(grid), dtype=bool)
    stale[ok] = (grid[ok] - t[idx[ok]]) > MAX_STALE_S
    out[stale] = np.nan
    return out


def driver_frame(session, drv: str, grid: np.ndarray) -> pd.DataFrame:
    car = session.car_data[drv]
    pos = session.pos_data[drv]
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


def add_gap_ahead(df: pd.DataFrame, length: float) -> pd.DataFrame:
    """Metres to the next car physically ahead on track (pit cars excluded)."""
    gap = np.full(len(df), np.nan)
    on = df["dist"].notna() & ~df["in_pit"]
    for _, g in df[on].groupby("t"):
        if len(g) < 2:
            continue
        d = g["dist"].to_numpy()
        order = np.argsort(d)
        ahead = np.roll(order, -1)
        diff = (d[ahead] - d[order]) % length
        gap[g.index.to_numpy()[order]] = diff
    df["gap_ahead_m"] = gap
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
    df["msector"] = ref.msector_of(df["dist"].to_numpy())
    df["ref_speed"] = ref.ref_speed(df["dist"].to_numpy())
    race = pd.DataFrame({"t": grid, "lap": lap_numbers(session.laps, grid),
                         "track_status": track_status_at(session, grid)})
    df = df.merge(race, on="t", how="left").sort_values(["t", "drv"]).reset_index(drop=True)
    return add_gap_ahead(df, ref.length)


def main() -> None:
    import fastf1

    from src.ingest.reference import build_track_ref
    year, event = int(sys.argv[1]), sys.argv[2]
    fastf1.Cache.enable_cache("data/fastf1_cache")
    s = fastf1.get_session(year, event, "R")
    s.load(weather=False)
    ref = build_track_ref(s, with_profile=False)
    df = merge_session(s, ref, t_end=secs(s.laps["LapStartTime"].dropna()).min() + 300)
    print(df.describe().T[["count", "mean", "min", "max"]])


if __name__ == "__main__":
    main()

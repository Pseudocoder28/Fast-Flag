"""Crash onset: a fixed, simple rule, independent of the detectors, used to anchor
latencies (src.eval.latency_by_type) and case studies (src.eval.case_study).

Own reference: for each car and lap, the median speed per 10 m of track over that
car's previous 3 clean laps (lap 2 or later, green for the whole lap, never in the
pit lane, race never suspended, data over at least 90% of the lap). Earlier laps
only, so it is causal. NaN until the car has at least one clean lap.

Onset: the first tick of a run of at least 1 s (4 ticks) where the car's speed is
below 50% of its own reference at that point, right after a tick where it was at
50% or more (a moving car collapsing, not a car that was already stopped), while the track is green or under a
local yellow (not SC, VSC or red), the car is on track and not in the pit lane,
and the field is racing: the median car on track is at 80% or more of its own
reference (after a red flag or a standing start the field runs slow laps where
many cars are below 50% at once, which is a procedure, not a crash).

The rule reuses no detector threshold. The field-racing gate replaced a first
"fewer than half the field below 50%" gate after that one let post-red-flag
processions through; this was decided before any latency was computed.

Run: python -m src.eval.onset 2023_Australian    (lists the onsets of a race)
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

BIN_M = 10.0
PREV_CLEAN_LAPS = 3
MIN_COVERAGE = 0.9
COLLAPSE = 0.5
SUSTAIN_TICKS = 4
FIELD_RACING = 0.8
RULE = ("speed below 50% of the car's own median speed at that point (10 m bins) on its previous 3 clean "
        "laps, for at least 1 s, right after a tick at 50% or more, while green or under a local yellow, on track, not in the pit lane, and "
        "while the field is racing (median car on track at 80% or more of its own reference)")


MAX_GAP_M = 100.0        # a gap between samples longer than this counts as missing data


def clean_lap_profiles(car: pd.DataFrame, nbins: int, length: float) -> dict[int, np.ndarray]:
    """{lap number: speed at each 10 m bin centre} for this car's clean laps. At 4 Hz a
    car covers 15 to 20 m per sample, so each lap is interpolated along the distance;
    coverage is the share of the lap without a sampling gap over MAX_GAP_M."""
    out = {}
    centres = (np.arange(nbins) + 0.5) * BIN_M
    for lap, g in car.groupby("drv_lap"):
        if lap < 2 or not (g["track_status"] == "1").all() or g["in_pit"].any() or g["suspended"].any():
            continue
        ok = g["speed"].notna() & g["dist"].notna()
        d, v = g.loc[ok, "dist"].to_numpy(), g.loc[ok, "speed"].to_numpy()
        order = np.argsort(d)
        d, v = d[order], v[order]
        if len(d) < 20:
            continue
        gaps = np.diff(d)
        covered = gaps[gaps <= MAX_GAP_M].sum()
        if covered / length >= MIN_COVERAGE:
            out[int(lap)] = np.interp(centres, d, v)
    return out


def own_reference(frame: pd.DataFrame, length: float) -> pd.Series:
    """Per row: the car's median speed at that point over its previous 3 clean laps."""
    nbins = int(np.ceil(length / BIN_M))
    ref = pd.Series(np.nan, index=frame.index)
    for _, car in frame.groupby("drv", sort=False):
        profiles = clean_lap_profiles(car, nbins, length)
        laps = sorted(profiles)
        for lap, g in car.groupby("drv_lap"):
            prev = [profiles[k] for k in laps if k < lap][-PREV_CLEAN_LAPS:]
            if not prev:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN bins stay NaN
                med = np.nanmedian(np.vstack(prev), axis=0)
            bins = (g["dist"].fillna(0) // BIN_M).astype(int).clip(0, nbins - 1).to_numpy()
            vals = med[bins]
            vals[g["dist"].isna().to_numpy()] = np.nan
            ref.loc[g.index] = vals
    return ref


def add_own_ratio(frame: pd.DataFrame, length: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["own_ref"] = own_reference(frame, length)
    frame["own_ratio"] = frame["speed"] / frame["own_ref"]
    return frame


def onsets(frame: pd.DataFrame, length: float) -> pd.DataFrame:
    """Every onset in the race: drv, t, dist, msector, speed, own_ref."""
    f = frame if "own_ratio" in frame else add_own_ratio(frame, length)
    on_track = (f["speed"].notna() & f["x"].notna() & ~f["in_pit"].astype(bool) & ~f["suspended"].astype(bool)
                & f["track_status"].isin(["1", "2"]))
    collapsed = on_track & (f["own_ratio"] < COLLAPSE)
    field_median = f["own_ratio"].where(on_track).groupby(f["t"]).median()
    field_ok = f["t"].map(field_median) >= FIELD_RACING
    hit = (collapsed & field_ok).to_numpy()
    moving = (f["own_ratio"] >= COLLAPSE).to_numpy()          # NaN (no reference or no data) is not moving
    rows = []
    for drv, idx in f.groupby("drv", sort=False).indices.items():
        idx = idx[np.argsort(f["t"].to_numpy()[idx])]
        h, mv = hit[idx], moving[idx]
        start = None
        for k in range(len(idx)):
            if h[k]:
                start = k if start is None else start
                if k - start + 1 == SUSTAIN_TICKS and start > 0 and mv[start - 1]:
                    r = f.iloc[idx[start]]
                    rows.append({"drv": str(drv), "t": float(r["t"]), "dist": float(r["dist"]),
                                 "msector": int(r["msector"]), "speed": float(r["speed"]),
                                 "own_ref": float(r["own_ref"])})
            else:
                start = None
    return pd.DataFrame(rows, columns=["drv", "t", "dist", "msector", "speed", "own_ref"]).sort_values("t")


def main() -> None:
    from src.replay.engine import RaceData
    race = RaceData.load(sys.argv[1] if len(sys.argv) > 1 else "2023_Australian")
    length = float(race.meta["lap_length_m"])
    o = onsets(race.frame, length)
    print(f"{race.race}: {len(o)} onsets ({RULE})")
    print(o.round(1).to_string(index=False))


if __name__ == "__main__":
    main()

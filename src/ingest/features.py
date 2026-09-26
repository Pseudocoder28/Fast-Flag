"""Engineered per-car features on the merged 250 ms grid (Section 6.4 inputs).

Every feature at time t uses only samples at or before t: rolling windows look
backwards, and lap, tyre and weather values are as-of joins on times already
passed (a lap's compound and tyre life are known when the lap starts).

Run: python -m src.ingest.features 2023_Australian   (summary of a built race)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

WIN_1S = 4               # samples in 1 s at 250 ms
WIN_2S = 8
WIN_5S = 20
MIN_SPEED_MS = 1.0       # avoid dividing by ~0 when converting gaps to seconds
NEUTRAL_STATUS = {"4", "5", "6", "7"}  # SC, red, VSC, VSC ending


def add_car_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Speed vs reference, deceleration, line and pedal behaviour (backward windows)."""
    df = df.sort_values(["drv", "t"])
    g = df.groupby("drv", sort=False)
    ref = df["ref_speed"].clip(lower=30.0)
    df["speed_dev"] = df["speed"] - df["ref_speed"]
    df["speed_ratio"] = df["speed"] / ref
    df["dspeed_1s"] = df["speed"] - g["speed"].shift(WIN_1S)
    df["lat_off_abs"] = df["lat_off"].abs()
    df["lat_off_std_2s"] = g["lat_off"].rolling(WIN_2S, min_periods=4).std().reset_index(level=0, drop=True)
    df["throttle_std_2s"] = g["throttle"].rolling(WIN_2S, min_periods=4).std().reset_index(level=0, drop=True)
    df["brake_throttle"] = ((df["brake"] > 0) & (df["throttle"] > 10)).astype(float)
    same_ahead = df["ahead_drv"].notna() & (df["ahead_drv"] == g["ahead_drv"].shift(WIN_1S))
    closing = g["gap_ahead_m"].shift(WIN_1S) - df["gap_ahead_m"]
    # m/s, positive = catching the car ahead; over 100 m/s is the gap wrapping at the line, not physics
    df["closing_rate"] = closing.where(same_ahead & (closing.abs() <= 100))
    df = add_windows(df, g)
    return df.sort_values(["t", "drv"])


def rolling(g, col: str, how: str, n: int = WIN_2S) -> pd.Series:
    r = g[col].rolling(n, min_periods=2)
    return getattr(r, how)().reset_index(level=0, drop=True)


def add_windows(df: pd.DataFrame, g) -> pd.DataFrame:
    """2 s backward windows used by the detectors and the ANOMALY model (Section 6.3)."""
    df["ratio_min_2s"] = rolling(g, "speed_ratio", "min")
    df["dspeed_min_2s"] = rolling(g, "dspeed_1s", "min")
    df["lat_abs_max_2s"] = rolling(g, "lat_off_abs", "max")
    df["brake_throttle_2s"] = rolling(g, "brake_throttle", "mean")
    return df


def ref_point(ref, dist: np.ndarray) -> np.ndarray:
    """Reference line point at each lap distance (NaN distance -> NaN point)."""
    n = len(ref.ref_xy)
    idx = np.clip(np.searchsorted(ref.ref_dist, np.nan_to_num(dist)) - 1, 0, n - 1)
    out = ref.ref_xy[idx].astype(float)
    out[~np.isfinite(dist)] = np.nan
    return out


def add_heading(df: pd.DataFrame, ref) -> pd.DataFrame:
    """Angle in degrees between the car's path over the last 1 s and the reference
    line's path between the same two track positions (0 = following the track,
    180 = going backwards). Comparing chord with chord means a hairpin bends both
    paths the same way. This is the direction of travel, not where the car points
    (there is no yaw channel): it shows a car leaving the track or reversing.
    NaN when the car moved less than 4 m (no reliable direction)."""
    df = df.sort_values(["drv", "t"])
    g = df.groupby("drv", sort=False)
    dx = (df["x"] - g["x"].shift(WIN_1S)).to_numpy()
    dy = (df["y"] - g["y"].shift(WIN_1S)).to_numpy()
    r1 = ref_point(ref, df["dist"].to_numpy())
    r0 = ref_point(ref, g["dist"].shift(WIN_1S).to_numpy())
    rx, ry = r1[:, 0] - r0[:, 0], r1[:, 1] - r0[:, 1]
    moved, rlen = np.hypot(dx, dy), np.hypot(rx, ry)
    cos = (dx * rx + dy * ry) / (moved * rlen + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    df["heading_err"] = np.where((moved >= 4.0) & (rlen >= 4.0), ang, np.nan)
    return df.sort_values(["t", "drv"])


def add_prev_lap(df: pd.DataFrame, length: float) -> pd.DataFrame:
    """The car's speed at the same point one lap earlier, and today's speed as a
    share of it (lap_ratio, capped at 1.1 so only being slower stands out).
    Race distance only counts forward progress, so the lookup always lands on
    samples from about one lap before: causal by construction."""
    df = df.sort_values(["drv", "t"])
    prev = np.full(len(df), np.nan)
    pos = 0
    for _, g in df.groupby("drv", sort=False):
        d = g["dist"].to_numpy()
        step = (np.diff(d, prepend=d[0]) + length / 2) % length - length / 2
        step = np.where(np.isfinite(step) & (step > 0) & (step < 150), step, 0.0)
        rd = np.cumsum(step)
        v = g["speed"].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(d)
        target = rd - length
        if ok.sum() > 1:
            p = np.interp(target, rd[ok], v[ok])
            p[target < rd[ok][0]] = np.nan
            prev[pos:pos + len(g)] = p
        pos += len(g)
    df["prev_lap_speed"] = prev
    df["lap_ratio"] = (df["speed"] / df["prev_lap_speed"].clip(lower=20)).clip(upper=1.1)
    df["lap_ratio_min_2s"] = df.groupby("drv", sort=False)["lap_ratio"].rolling(
        WIN_2S, min_periods=2).min().reset_index(level=0, drop=True)
    return df.sort_values(["t", "drv"])


def add_battle(df: pd.DataFrame) -> pd.DataFrame:
    v = (df["speed"] / 3.6).clip(lower=MIN_SPEED_MS)
    df["gap_ahead_s"] = df["gap_ahead_m"] / v
    df["gap_behind_s"] = df["gap_behind_m"] / v
    df["cars_within_1s"] = (df["gap_ahead_s"] < 1).astype(int) + (df["gap_behind_s"] < 1).astype(int)
    return df


def add_lap_info(df: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Driver's own lap, compound, tyre life, stint and laps since pit (as-of lap start)."""
    lp = laps.dropna(subset=["LapStartTime"]).copy()
    lp["t_lap"] = lp["LapStartTime"].dt.total_seconds()
    lp["drv"] = lp["DriverNumber"].astype(str)
    lp["stint_first_lap"] = lp.groupby(["drv", "Stint"])["LapNumber"].transform("min")
    lp = lp[["t_lap", "drv", "LapNumber", "Compound", "TyreLife", "Stint", "stint_first_lap"]]
    lp = lp.sort_values("t_lap")
    out = pd.merge_asof(df.sort_values("t"), lp, left_on="t", right_on="t_lap", by="drv",
                        direction="backward")
    out = out.rename(columns={"LapNumber": "drv_lap", "Compound": "compound",
                              "TyreLife": "tyre_life", "Stint": "stint"})
    out["laps_since_pit"] = out["drv_lap"] - out["stint_first_lap"]
    out["compound"] = out["compound"].fillna("UNKNOWN").astype(str)
    return out.drop(columns=["t_lap", "stint_first_lap"])


def add_lap_pace(df: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """last_lap_delta: the driver's last completed lap time vs the median of the 3
    laps before it (0.05 = 5% slower). Known from the moment that lap ends."""
    lp = laps.dropna(subset=["LapTime", "Time"]).copy()
    lp["drv"] = lp["DriverNumber"].astype(str)
    lp["t_end"] = lp["Time"].dt.total_seconds()
    lp["lap_s"] = lp["LapTime"].dt.total_seconds()
    lp = lp.sort_values(["drv", "t_end"])
    med3 = lp.groupby("drv")["lap_s"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    lp["last_lap_delta"] = lp["lap_s"] / med3 - 1
    lp = lp[["t_end", "drv", "last_lap_delta"]].sort_values("t_end")
    out = pd.merge_asof(df.sort_values("t"), lp, left_on="t", right_on="t_end", by="drv", direction="backward")
    return out.drop(columns=["t_end"])


def add_long_windows(df: pd.DataFrame) -> pd.DataFrame:
    """5 s backward windows for the risk model (Section 6.4: rolling windows of 2 to 5 s)."""
    df = df.sort_values(["drv", "t"])
    g = df.groupby("drv", sort=False)
    for col, how, name in [("rel_ratio", "min", "rel_ratio_min_5s"), ("lap_ratio", "mean", "lap_ratio_mean_5s"),
                           ("lap_ratio", "min", "lap_ratio_min_5s"), ("lat_off", "std", "lat_off_std_5s"),
                           ("throttle", "std", "throttle_std_5s"), ("brake_throttle", "mean", "brake_throttle_5s"),
                           ("dspeed_1s", "min", "dspeed_min_5s"), ("closing_rate", "max", "closing_rate_max_5s"),
                           ("heading_err", "max", "heading_max_5s")]:
        if col in df:
            df[name] = rolling(g, col, how, WIN_5S)
    return df.sort_values(["t", "drv"])


def add_weather(df: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    if weather is None or not len(weather):
        df["air_temp"] = df["track_temp"] = df["rainfall"] = np.nan
        return df
    w = pd.DataFrame({"t_w": weather["Time"].dt.total_seconds(), "air_temp": weather["AirTemp"],
                      "track_temp": weather["TrackTemp"],
                      "rainfall": weather["Rainfall"].astype(float)}).sort_values("t_w")
    out = pd.merge_asof(df.sort_values("t"), w, left_on="t", right_on="t_w", direction="backward")
    return out.drop(columns=["t_w"])


def add_field_state(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tick field state from the current tick only.

    field_in_pit: share of cars in the pit lane. field_slow: share of on-track cars
    below 30 km/h (standing starts, restarts). suspended: red flag, or most of the
    field in the pit lane (track status can read green during a red flag suspension).
    field_ratio: median speed ratio of on-track cars. rel_ratio: the car's speed
    ratio divided by it, so SC / VSC periods, processions and wet races look
    normal and only a car that is slow compared with the field stands out.
    """
    on = df["speed"].notna()
    track = on & ~df["in_pit"] & df["x"].notna()
    g = df[on].groupby("t")
    state = pd.DataFrame({
        "field_in_pit": g["in_pit"].mean(),
        "field_slow": df[on & ~df["in_pit"]].assign(slow=lambda d: d["speed"] < 30).groupby("t")["slow"].mean(),
        "field_ratio": df[track].groupby("t")["speed_ratio"].median(),
    }).reset_index()
    out = df.merge(state, on="t", how="left")
    out["field_slow"] = out["field_slow"].fillna(0.0)
    out["field_ratio"] = out["field_ratio"].fillna(1.0)
    out["suspended"] = (out["track_status"] == "5") | (out["field_in_pit"] >= 0.5)
    out["rel_ratio"] = out["speed_ratio"] / out["field_ratio"].clip(0.2, 1.0)
    out = out.sort_values(["drv", "t"])
    out["rel_ratio_min_2s"] = out.groupby("drv", sort=False)["rel_ratio"].rolling(
        WIN_2S, min_periods=2).min().reset_index(level=0, drop=True)
    return out


def add_race_context(df: pd.DataFrame) -> pd.DataFrame:
    """Lap 1 flag and seconds since the last neutralisation (SC, VSC, red) ended."""
    race = df[["t", "track_status"]].drop_duplicates("t").sort_values("t")
    neutral = race["track_status"].isin(NEUTRAL_STATUS).to_numpy()
    ended = np.zeros(len(race), dtype=bool)
    ended[1:] = neutral[:-1] & ~neutral[1:]
    last_end = pd.Series(np.where(ended, race["t"], np.nan)).ffill().to_numpy()
    race["t_since_restart"] = np.where(np.isnan(last_end), np.nan, race["t"].to_numpy() - last_end)
    out = df.merge(race[["t", "t_since_restart"]], on="t", how="left")
    out["is_lap1"] = (out["drv_lap"] == 1).astype(int)
    return out


def add_features(df: pd.DataFrame, session, ref=None) -> pd.DataFrame:
    df = add_car_dynamics(df)
    if ref is not None:
        df = add_heading(df, ref)
        df = add_prev_lap(df, ref.length)
    df = add_battle(df)
    df = add_lap_info(df, session.laps)
    df = add_weather(df, getattr(session, "weather_data", None))
    df = add_lap_pace(df, session.laps)
    df = add_race_context(df)
    df = add_field_state(df)
    df = add_long_windows(df)
    return df.sort_values(["t", "drv"]).reset_index(drop=True)


def main() -> None:
    path = f"data/features/{sys.argv[1]}.parquet"
    df = pd.read_parquet(path)
    print(f"{path}: {len(df)} rows, {df['drv'].nunique()} cars, t {df['t'].min():.0f} to {df['t'].max():.0f}")
    print(df.describe().T[["count", "mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()

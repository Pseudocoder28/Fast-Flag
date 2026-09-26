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
    df["closing_rate"] = closing.where(same_ahead)            # m/s, positive = catching the car ahead
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
    """
    on = df["speed"].notna()
    g = df[on].groupby("t")
    state = pd.DataFrame({
        "field_in_pit": g["in_pit"].mean(),
        "field_slow": df[on & ~df["in_pit"]].assign(slow=lambda d: d["speed"] < 30).groupby("t")["slow"].mean(),
    }).reset_index()
    out = df.merge(state, on="t", how="left")
    out["field_slow"] = out["field_slow"].fillna(0.0)
    out["suspended"] = (out["track_status"] == "5") | (out["field_in_pit"] >= 0.5)
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


def add_features(df: pd.DataFrame, session) -> pd.DataFrame:
    df = add_car_dynamics(df)
    df = add_battle(df)
    df = add_lap_info(df, session.laps)
    df = add_weather(df, getattr(session, "weather_data", None))
    df = add_race_context(df)
    df = add_field_state(df)
    return df.sort_values(["t", "drv"]).reset_index(drop=True)


def main() -> None:
    path = f"data/features/{sys.argv[1]}.parquet"
    df = pd.read_parquet(path)
    print(f"{path}: {len(df)} rows, {df['drv'].nunique()} cars, t {df['t'].min():.0f} to {df['t'].max():.0f}")
    print(df.describe().T[["count", "mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()

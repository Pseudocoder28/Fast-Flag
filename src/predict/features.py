"""Risk model features (Section 6.4). One function builds the matrix for training,
evaluation and live replay, so the model always sees the same numbers.

Every column is backward-looking (computed in src.ingest.features from samples at
or before t), plus the ANOMALY score, which only uses the car's last 2 s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = [
    # pace vs the field, vs its own last lap, vs recent laps
    "rel_ratio", "rel_ratio_min_2s", "rel_ratio_min_5s", "lap_ratio", "lap_ratio_min_2s", "lap_ratio_mean_5s",
    "lap_ratio_min_5s", "last_lap_delta", "speed", "ref_speed",
    # deceleration, line and pedals
    "dspeed_1s", "dspeed_min_2s", "dspeed_min_5s", "lat_off_abs", "lat_abs_max_2s", "lat_off_std_2s",
    "lat_off_std_5s", "heading_err", "heading_max_5s", "throttle", "throttle_std_2s", "throttle_std_5s",
    "brake_throttle_2s", "brake_throttle_5s",
    # battles
    "gap_ahead_s", "gap_behind_s", "closing_rate", "closing_rate_max_5s", "cars_within_1s",
    # tyres, race phase, weather, field
    "tyre_life", "laps_since_pit", "drv_lap", "is_lap1", "t_since_restart", "track_temp", "air_temp",
    "rainfall", "field_ratio",
    # learned: the ANOMALY model's score
    "anomaly_score",
]
COMPOUNDS = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
CATEGORICAL = ["compound_code", "status_code"]
FEATURES = NUMERIC + CATEGORICAL


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """float32 feature frame in FEATURES order. Missing columns (e.g. no ANOMALY
    model loaded) become NaN, which LightGBM handles."""
    out = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        out[c] = df[c].astype("float32") if c in df else np.float32(np.nan)
    out["compound_code"] = df["compound"].map(COMPOUNDS).fillna(-1).astype("float32") if "compound" in df else -1.0
    out["status_code"] = pd.to_numeric(df["track_status"], errors="coerce").fillna(1).astype("float32")
    return out[FEATURES]


def racing_rows(df: pd.DataFrame) -> pd.Series:
    """Ticks where a risk forecast makes sense: live car on track, race running, not on the grid."""
    return (df["speed"].notna() & df["x"].notna() & ~df["in_pit"].astype(bool) & ~df["suspended"].astype(bool)
            & (df["field_slow"] < 0.5))

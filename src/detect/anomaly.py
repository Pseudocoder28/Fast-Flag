"""ANOMALY model (Section 6.3): an IsolationForest on normal racing windows.

The learned core of the product (CLAUDE.md rule 6: never cut). Trained only on
training races, never the holdout, and only on rows of normal racing: green
track status, not in the pit lane, not suspended, field racing (not on the
grid), car moving above 60 km/h. The features are 2 s backward windows computed
in ingest, so training and replay use exactly the same numbers and nothing looks
ahead. Speed is relative to the field, so wet races and SC periods do not look
unusual on their own.

Score: higher = more unusual (negated IsolationForest score_samples). The alert
threshold is a high quantile of scores on normal training rows.

Run: python -m src.detect.anomaly train              (all training races -> data/models/anomaly.joblib)
     python -m src.detect.anomaly score 2023_Australian
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.ingest.holdout import assert_not_holdout

FEATURES = ["rel_ratio", "rel_ratio_min_2s", "lap_ratio", "lap_ratio_min_2s", "dspeed_1s", "dspeed_min_2s",
            "lat_abs_max_2s", "lat_off_std_2s", "throttle_std_2s", "brake_throttle_2s", "heading_err"]
CONTEXT = ["track_status", "in_pit", "suspended", "field_slow", "speed", "x"]
FEATURE_DIR = Path("data/features")
MODEL_PATH = Path("data/models/anomaly.joblib")
SAMPLE_PER_RACE = 40_000
THRESHOLD_Q = 0.99995         # default alert threshold: quantile of scores on normal training rows
QUANTILES = (0.999, 0.9995, 0.9999, 0.99995, 0.99999)
SEED = 0


@dataclass
class AnomalyModel:
    forest: IsolationForest
    fill: dict[str, float]          # training medians for missing values
    threshold: float
    races: list[str]
    quantiles: dict[float, float]   # score quantiles on normal training rows, for choosing a threshold

    def threshold_at(self, q: float) -> float:
        return self.quantiles[q]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return -self.forest.score_samples(matrix(df, self.fill))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "AnomalyModel":
        return joblib.load(path)


def matrix(df: pd.DataFrame, fill: dict[str, float]) -> np.ndarray:
    x = df[FEATURES].astype(float)
    x = x.assign(heading_err=x["heading_err"].fillna(0.0))    # NaN = car barely moved: no direction
    return x.fillna(fill).to_numpy()


def normal_rows(df: pd.DataFrame) -> pd.Series:
    core = [f for f in FEATURES if f != "heading_err"]
    return ((df["track_status"] == "1") & ~df["in_pit"] & ~df["suspended"] & (df["field_slow"] < 0.1)
            & (df["speed"] > 60) & df["x"].notna() & df[core].notna().all(axis=1))


def read(rid: str) -> pd.DataFrame:
    return pd.read_parquet(FEATURE_DIR / f"{rid}.parquet", columns=FEATURES + CONTEXT)


def train(races: list[str], seed: int = SEED) -> AnomalyModel:
    parts = []
    for rid in races:
        assert_not_holdout(rid=rid)
        df = read(rid)
        df = df[normal_rows(df)]
        parts.append(df.sample(min(SAMPLE_PER_RACE, len(df)), random_state=seed))
    data = pd.concat(parts, ignore_index=True)
    fill = data[FEATURES].median().to_dict()
    x = matrix(data, fill)
    forest = IsolationForest(n_estimators=200, max_samples=512, random_state=seed, n_jobs=-1).fit(x)
    scores = -forest.score_samples(x)
    quantiles = {q: float(np.quantile(scores, q)) for q in QUANTILES}
    return AnomalyModel(forest, fill, quantiles[THRESHOLD_Q], sorted(races), quantiles)


def attach_scores(frame: pd.DataFrame, model: AnomalyModel) -> pd.DataFrame:
    """Per-row score. Each row's features only use that car's past 2 s, so scoring the
    whole race up front is still causal: row t never sees anything after t."""
    frame["anomaly_score"] = model.score(frame)
    return frame


def main() -> None:
    from src.replay.engine import available_races
    p = argparse.ArgumentParser(description="Train or apply the ANOMALY model")
    p.add_argument("action", choices=["train", "score"])
    p.add_argument("race", nargs="?")
    a = p.parse_args()
    if a.action == "train":
        races = available_races()
        model = train(races)
        model.save()
        print(f"trained on {len(races)} training races, threshold {model.threshold:.3f} -> {MODEL_PATH}")
        return
    model = AnomalyModel.load()
    df = read(a.race)
    s = model.score(df)
    n = normal_rows(df)
    print(f"{a.race}: score p50 {np.median(s):.3f}, p99.95 on normal rows {np.quantile(s[n], 0.9995):.3f}, "
          f"threshold {model.threshold:.3f}, rows above threshold {int((s > model.threshold).sum())}")


if __name__ == "__main__":
    main()

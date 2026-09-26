"""A6 risk model: LightGBM for "this car has an incident within H s" (Section 6.4).

Evaluation is leave-one-race-out on training races only: for each race, models are
trained on the other races' risk datasets (src.predict.dataset) and score every
racing tick of the held-out race. Metrics:
- PR-AUC per horizon, pooled over all held-out ticks (not accuracy).
- Baselines on the same ticks: the ANOMALY score, the naive speed threshold
  (lower speed = higher risk) and the base rate.
- Precursor PR-AUC: the same, with each incident's last 3 s before detection left
  out, so the car already crashing does not count as "predicting" it.
- Early warning at several thresholds: share of car incidents where risk_30s
  crossed the threshold at least 3 s before the detection, and false high-risk
  episodes per race hour (a car above the threshold, with no incident of its
  own in the next 30 s; above-threshold ticks less than 5 s apart are one episode).
Some incidents have no precursor: this is risk forecasting, not a crystal ball.

Negatives are downsampled to 3% for training, so raw model outputs are inflated;
risk values are corrected back to real probabilities (odds x 0.03).

Run: python -m src.predict.risk cv        (leave-one-race-out, writes docs/charts/risk_eval.md)
     python -m src.predict.risk train     (all training races -> data/models/risk_*.txt)
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.ingest.holdout import assert_not_holdout
from src.predict.dataset import FEATURES_DIR, HORIZONS, NEG_SAMPLE
from src.predict.features import FEATURES
from src.predict.processor import correct as correct_rate
from src.replay.engine import available_races

MODELS = Path("data/models")
CHARTS = Path("docs/charts")
PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
          "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
          "random_state": 0, "verbose": -1}
ALERT_NEG_RATES = (0.005, 0.002, 0.001)   # share of no-incident ticks above the "high risk" threshold
DEFAULT_NEG_RATE = 0.001
EARLY_S = 3.0
EPISODE_GAP_S = 5.0
# Air and track temperature are left out: they mostly identify the race and time of day (a fingerprint),
# and without them precursor PR-AUC is higher at 10 and 20 s (docs/charts/risk_eval.md). RISK_DROP adds more.
DROP = ["track_temp", "air_temp"] + [f for f in os.environ.get("RISK_DROP", "").split(",") if f]
ACTIVE = [f for f in FEATURES if f not in DROP]


def risk_path(rid: str) -> Path:
    return FEATURES_DIR / f"{rid}_risk.parquet"


def correct(p: np.ndarray) -> np.ndarray:
    """Undo negative downsampling (shared with the live processor)."""
    return correct_rate(p, NEG_SAMPLE)


def training_rows(races: list[str]) -> pd.DataFrame:
    parts = []
    for rid in races:
        assert_not_holdout(rid=rid)
        df = pd.read_parquet(risk_path(rid))
        parts.append(df[df["train_keep"]])
    return pd.concat(parts, ignore_index=True)


def fit(train: pd.DataFrame, h: int, n_jobs: int = 2) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**PARAMS, n_jobs=n_jobs).fit(train[ACTIVE], train[f"y{h}"])


def cv_race(rid: str) -> str:
    """Train on the other races, score every racing tick of rid, save predictions."""
    others = [r for r in available_races() if r != rid]
    train = training_rows(others)
    test = pd.read_parquet(risk_path(rid))
    out = test[["t", "drv", "msector", "speed", "anomaly_score"] + [f"y{h}" for h in HORIZONS]].copy()
    for h in HORIZONS:
        out[f"p{h}"] = correct(fit(train, h).predict_proba(test[ACTIVE])[:, 1]).astype("float32")
    out["race"] = rid
    path = FEATURES_DIR / f"{rid}_risk_pred.parquet"
    out.to_parquet(path, index=False)
    return str(path)


def incident_end(pred: pd.DataFrame) -> pd.Series:
    """Per row: the time of that car's incident (NaN for cars with none). Positives
    end just before the incident, so it is the last positive tick + one tick."""
    ends = pred[pred["y30"] == 1].groupby(["race", "drv"])["t"].max() + 0.25
    return pd.Series(pd.MultiIndex.from_frame(pred[["race", "drv"]]).map(ends), index=pred.index)


def early_warning(pred: pd.DataFrame, threshold: float, hours: float) -> dict:
    """Incidents flagged EARLY_S before detection, and false high-risk episodes per race hour."""
    t_inc = incident_end(pred)
    rows = []
    for (race, drv), g in pred[pred["y30"] == 1].groupby(["race", "drv"]):
        t0 = t_inc.loc[g.index[0]]
        crossed = g[(g["t"] <= t0 - EARLY_S) & (g["p30"] >= threshold)]
        rows.append({"early": len(crossed) > 0,
                     "seconds_before": float(t0 - crossed["t"].min()) if len(crossed) else np.nan})
    inc = pd.DataFrame(rows)
    hi = pred[(pred["p30"] >= threshold) & (pred["y30"] == 0)].sort_values(["race", "drv", "t"])
    new_episode = (hi["t"].diff() > EPISODE_GAP_S) | (hi["drv"] != hi["drv"].shift()) | (hi["race"] != hi["race"].shift())
    return {"threshold_p30": threshold, "car_incidents": len(inc),
            "flagged_3s_before": float(inc["early"].mean()),
            "median_s_before_when_flagged": float(inc["seconds_before"].median()),
            "false_episodes_per_hour": float(new_episode.sum() / hours)}


def pr_auc_per_race(pred: pd.DataFrame, y: str, s: np.ndarray) -> float:
    vals = [average_precision_score(g[y], s[g.index]) for _, g in pred.groupby("race") if g[y].any()]
    return float(np.mean(vals))


def evaluate(pred: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    pred = pred.reset_index(drop=True)
    t_inc = incident_end(pred)
    precursor = ~((pred["t"] >= t_inc - EARLY_S) & (pred["t"] < t_inc)).to_numpy()
    rows = []
    for h in HORIZONS:
        y = pred[f"y{h}"].to_numpy()
        scores = {"lightgbm": pred[f"p{h}"].to_numpy(), "anomaly_score": pred["anomaly_score"].fillna(0).to_numpy(),
                  "speed_threshold": -pred["speed"].to_numpy()}
        for name, s in scores.items():
            rows.append({"horizon_s": h, "model": name, "pr_auc": average_precision_score(y, s),
                         "pr_auc_precursor": average_precision_score(y[precursor], s[precursor]),
                         "pr_auc_mean_per_race": pr_auc_per_race(pred, f"y{h}", s), "base_rate": y.mean()})
    neg = pred.loc[pred["y30"] == 0, "p30"]
    hours = len(pred) / 4 / 3600 / pred.groupby("race")["drv"].nunique().mean()   # racing time, per car-field
    early = [early_warning(pred, float(np.quantile(neg, 1 - r)), hours) | {"neg_tick_rate": r}
             for r in ALERT_NEG_RATES]
    return pd.DataFrame(rows), early


def importance(model: lgb.LGBMClassifier) -> pd.DataFrame:
    gain = model.booster_.feature_importance(importance_type="gain")
    return (pd.DataFrame({"feature": ACTIVE, "gain": gain / gain.sum()})
            .sort_values("gain", ascending=False).reset_index(drop=True))


def cmd_cv(retrain: bool = True) -> None:
    races = available_races()
    if retrain:
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as ex:
            list(ex.map(cv_race, races))
    pred = pd.concat([pd.read_parquet(FEATURES_DIR / f"{r}_risk_pred.parquet") for r in races], ignore_index=True)
    table, early = evaluate(pred)
    ew = pd.DataFrame(early)[["neg_tick_rate", "threshold_p30", "flagged_3s_before",
                              "median_s_before_when_flagged", "false_episodes_per_hour"]]
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    print(ew.round(4).to_string(index=False))
    CHARTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(CHARTS / "risk_eval.csv", index=False)
    (CHARTS / "risk_eval.md").write_text(
        "# Risk model, leave-one-race-out (training races only)\n\nReplay of historical FastF1 data. For each "
        "race, LightGBM is trained on the other training races and scores every racing tick of that race. "
        "PR-AUC is pooled over all held-out ticks; pr_auc_precursor leaves out each incident's last "
        f"{EARLY_S:.0f} s before detection. Baselines: ANOMALY score (IsolationForest, also leave-one-race-out) "
        "and the naive speed threshold (lower speed = higher risk).\n\n" + table.round(4).to_markdown(index=False)
        + f"\n\nEarly warning for {early[0]['car_incidents']} car incidents (risk_30s, flagged = crossed the "
          f"threshold at least {EARLY_S:.0f} s before detection; false episodes = a car above the threshold with "
          "no incident of its own in the next 30 s):\n\n" + ew.round(4).to_markdown(index=False)
        + "\n\nSome incidents have no precursor in the data: this is risk forecasting, not a crystal ball.\n")
    MODELS.mkdir(parents=True, exist_ok=True)
    chosen = next(e for e in early if e["neg_tick_rate"] == DEFAULT_NEG_RATE)
    (MODELS / "risk_cv.json").write_text(json.dumps({"chosen": chosen, "all": early}, indent=2))


def cmd_train() -> None:
    races = available_races()
    train = training_rows(races)
    MODELS.mkdir(parents=True, exist_ok=True)
    meta = {"features": ACTIVE, "horizons": list(HORIZONS), "neg_sample": NEG_SAMPLE, "races": races}
    cv = MODELS / "risk_cv.json"
    if cv.exists():
        meta["threshold_p30"] = json.loads(cv.read_text())["chosen"]["threshold_p30"]
    for h in HORIZONS:
        model = fit(train, h, n_jobs=-1)
        model.booster_.save_model(str(MODELS / f"risk_{h}.txt"))
        if h == max(HORIZONS):
            imp = importance(model)
            imp.to_csv(CHARTS / "risk_importance.csv", index=False)
            print(imp.head(15).round(4).to_string(index=False))
    (MODELS / "risk_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"trained on {len(races)} training races ({len(train)} rows) -> {MODELS}/risk_*.txt")


def main() -> None:
    p = argparse.ArgumentParser(description="Risk model: leave-one-race-out CV or final training")
    p.add_argument("action", choices=["cv", "eval", "train"], help="eval = re-score saved cv predictions")
    a = p.parse_args()
    if a.action == "train":
        cmd_train()
    else:
        cmd_cv(retrain=a.action == "cv")


if __name__ == "__main__":
    from src.predict.risk import main as run    # so anything pickled records src.predict.risk, not __main__
    run()

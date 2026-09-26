"""Live risk forecasts for the replay engine (contract Section 7.3).

One `risk` envelope per racing car per second of race time: risk_10s, risk_30s
(real probabilities, corrected for the training downsampling) and the two
features pushing that car's 30 s risk up the most (LightGBM contributions).
Stateless: it only reads the current tick's rows, whose features only look back.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.predict.features import feature_matrix, racing_rows

TOP_K = 2


def correct(p: np.ndarray, neg_sample: float) -> np.ndarray:
    """Undo negative downsampling: true odds = model odds x sampling rate."""
    odds = p / np.clip(1 - p, 1e-9, None) * neg_sample
    return odds / (1 + odds)


class RiskProcessor:
    def __init__(self, boosters: dict[int, lgb.Booster], features: list[str], neg_sample: float,
                 every_s: float = 1.0) -> None:
        self.b10, self.b30 = boosters[10], boosters[30]
        self.features, self.neg_sample, self.every_s = features, neg_sample, every_s

    def reset(self) -> None:
        pass                                     # no state: every forecast uses the current tick only

    def due(self, t: float) -> bool:
        k = t / self.every_s
        return abs(k - round(k)) < 1e-6

    def on_tick(self, t: float, frame: pd.DataFrame, tick: dict) -> list[dict]:
        if not self.due(t):
            return []
        rows = frame[racing_rows(frame)]
        if not len(rows):
            return []
        x = feature_matrix(rows)[self.features]
        r10 = correct(self.b10.predict(x), self.neg_sample)
        r30 = np.maximum(correct(self.b30.predict(x), self.neg_sample), r10)   # within 30 s >= within 10 s
        contrib = self.b30.predict(x, pred_contrib=True)[:, :-1]              # last column is the bias
        out = []
        for i, drv in enumerate(rows["drv"].astype(str)):
            order = np.argsort(-contrib[i])[:TOP_K]
            top = [self.features[j] for j in order if contrib[i, j] > 0]
            out.append({"kind": "risk", "data": {"t": round(float(t), 2), "drv": drv,
                                                 "risk_10s": round(float(r10[i]), 4),
                                                 "risk_30s": round(float(r30[i]), 4), "top_features": top}})
        return out

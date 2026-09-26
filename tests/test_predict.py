"""Risk model: labels, downsampling correction, live processor output and causality."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from test_contracts import check_risk

from src.eval.incidents import Incident
from src.predict.dataset import label
from src.predict.features import FEATURES
from src.predict.processor import RiskProcessor, correct
from src.replay.engine import Engine, RaceData

T0 = 1000.0


def frame(seconds: float = 40.0) -> pd.DataFrame:
    t = np.round(np.arange(T0, T0 + seconds, 0.25), 2)
    rows = [pd.DataFrame({"t": t, "drv": d, "msector": 5 + 7 * i, "speed": 200.0, "x": t, "y": 0.0, "dist": t,
                          "in_pit": False, "suspended": False, "field_slow": 0.0, "track_status": "1",
                          "compound": "SOFT", "lap": 3, "gear": 7.0, "rpm": 11000.0, "lat_off": 0.1,
                          "throttle": 100.0, "brake": 0.0, "gap_ahead_m": 50.0, "tyre_life": 5.0})
            for i, d in enumerate(["1", "44"])]
    return pd.concat(rows, ignore_index=True)


def test_labels_mark_the_seconds_before_and_drop_after() -> None:
    out = label(frame(), {"44": T0 + 20}, [], n=20)
    car = out[out["drv"] == "44"]
    assert car["t"].max() < T0 + 20                                    # nothing from the incident on
    pos = car.loc[car["y10"] == 1, "t"]
    assert pos.min() == T0 + 10 and pos.max() == T0 + 19.75
    assert out.loc[out["drv"] == "1", ["y10", "y30"]].to_numpy().sum() == 0


def test_unattributed_incident_drops_nearby_cars() -> None:
    inc = Incident(t=T0 + 30, sectors={5})
    out = label(frame(), {}, [inc], n=20)
    dropped = out[(out["drv"] == "1") & (out["t"] >= T0) & (out["t"] <= T0 + 30)]
    assert dropped.empty                                               # car 1 sits in sector 5
    assert len(out[out["drv"] == "44"]) == len(frame()) // 2           # car 44 (sector 12, far away) kept


def test_downsampling_correction() -> None:
    assert np.isclose(correct(np.array([0.5]), 0.03)[0], 0.03 / 1.03)
    p = np.linspace(0.01, 0.99, 50)
    assert np.all(np.diff(correct(p, 0.03)) > 0)                      # ranking unchanged


def tiny_boosters(features: list[str]) -> dict[int, lgb.Booster]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(400, len(features))), columns=features)
    y = (x.iloc[:, 0] > 1).astype(int)
    b = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 4}, lgb.Dataset(x, y), num_boost_round=5)
    return {10: b, 30: b}


def test_processor_emits_valid_risk_once_per_second_per_car() -> None:
    feats = [f for f in FEATURES if f not in ("track_temp", "air_temp")]
    proc = RiskProcessor(tiny_boosters(feats), feats, neg_sample=0.03)
    eng = Engine(RaceData.from_frame("TEST", frame()), [proc])
    risks = [e["data"] for e in eng.advance(T0 + 10) if e["kind"] == "risk"]
    assert len(risks) == 2 * 11                                        # t = 1000 .. 1010, two cars
    for r in risks:
        check_risk(r, "risk")
        assert r["t"] == int(r["t"]) and r["risk_30s"] >= r["risk_10s"]

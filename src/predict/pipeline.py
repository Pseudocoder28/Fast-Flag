"""Live risk pipeline for one race (used by the real server and the latency benchmark).

Loads the final risk models (data/models/risk_10.txt, risk_30.txt, risk_meta.json,
written by python -m src.predict.risk train on training races only). Returns no
processors if they are missing, so the server still runs without them. Makes sure
the race has ANOMALY scores (a risk feature) even when detection is switched off.

Run: python -m src.predict.pipeline 2023_Australian   (first forecasts of the race)
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb

from src.detect.anomaly import attach_scores
from src.detect.pipeline import load_anomaly
from src.predict.processor import RiskProcessor
from src.replay.engine import Engine, Processor, RaceData

MODELS = Path("data/models")


@lru_cache(maxsize=1)
def load_models() -> tuple[dict[int, lgb.Booster], dict] | None:
    meta_path = MODELS / "risk_meta.json"
    paths = {h: MODELS / f"risk_{h}.txt" for h in (10, 30)}
    if not meta_path.exists() or not all(p.exists() for p in paths.values()):
        return None
    return {h: lgb.Booster(model_file=str(p)) for h, p in paths.items()}, json.loads(meta_path.read_text(encoding="utf-8"))


def risk_processors(race: RaceData) -> list[Processor]:
    models = load_models()
    if models is None:
        return []
    boosters, meta = models
    model = load_anomaly()
    if model is not None and "anomaly_score" not in race.frame:
        attach_scores(race.frame, model)
    return [RiskProcessor(boosters, meta["features"], meta["neg_sample"])]


def main() -> None:
    race = RaceData.load(sys.argv[1] if len(sys.argv) > 1 else "2023_Australian")
    procs = risk_processors(race)
    if not procs:
        print("no risk models in data/models (python -m src.predict.risk train)")
        return
    eng = Engine(race, procs)
    risks = [e["data"] for e in eng.advance(eng.t_start + 60) if e["kind"] == "risk"]
    top = sorted(risks, key=lambda d: -d["risk_30s"])[:5]
    print(f"{len(risks)} risk envelopes in the first 60 s; highest risk_30s:")
    for d in top:
        print(d)


if __name__ == "__main__":
    main()

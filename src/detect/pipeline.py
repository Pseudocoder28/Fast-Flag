"""Live detection pipeline for one race (used by the real server).

Loads the tuned detector settings (data/models/detector_config.json, written by
src.eval.tune) and the ANOMALY model (data/models/anomaly.joblib, written by
python -m src.detect.anomaly train) when they exist, otherwise uses defaults.
The ANOMALY model is trained on training races only, so on the holdout race it
scores data it has never seen.

Run: python -m src.detect.pipeline 2023_Australian   (prints what would be loaded)
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from src.detect.anomaly import MODEL_PATH, AnomalyModel, attach_scores
from src.detect.detectors import Config, DetectorSuite
from src.replay.engine import Processor, RaceData

CONFIG_PATH = Path("data/models/detector_config.json")


@lru_cache(maxsize=1)
def load_config() -> Config:
    if not CONFIG_PATH.exists():
        return Config()
    saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["config"]
    return replace(Config(), **{k: v for k, v in saved.items() if k in Config.__dataclass_fields__})


@lru_cache(maxsize=1)
def load_anomaly() -> AnomalyModel | None:
    return AnomalyModel.load() if MODEL_PATH.exists() else None


def detection_processors(race: RaceData) -> list[Processor]:
    model = load_anomaly()
    threshold = None
    if model is not None:
        if "anomaly_score" not in race.frame:
            attach_scores(race.frame, model)
        threshold = model.threshold
    n = len(race.track.get("msectors", [])) or None
    return [DetectorSuite(load_config(), anomaly_threshold=threshold, n_sectors=n)]


def main() -> None:
    race = RaceData.load(sys.argv[1] if len(sys.argv) > 1 else "2023_Australian")
    procs = detection_processors(race)
    print(f"config from {'tuned file' if CONFIG_PATH.exists() else 'defaults'}: {load_config()}")
    print(f"anomaly model: {'loaded, threshold %.3f' % load_anomaly().threshold if load_anomaly() else 'none'}")
    print(f"processors: {[type(p).__name__ for p in procs]}")


if __name__ == "__main__":
    main()

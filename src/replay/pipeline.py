"""Every live processor for one race, in order: detection (which also attaches the
ANOMALY scores), then risk. Used by the real server and the latency benchmark, so
the benchmark always times every stage that exists.
"""

from __future__ import annotations

from src.replay.engine import Processor, RaceData


def all_processors(race: RaceData, detect: bool = True, predict: bool = True) -> list[Processor]:
    procs: list[Processor] = []
    if detect:
        from src.detect.pipeline import detection_processors
        procs += detection_processors(race)
    if predict:
        from src.predict.pipeline import risk_processors
        procs += risk_processors(race)
    return procs

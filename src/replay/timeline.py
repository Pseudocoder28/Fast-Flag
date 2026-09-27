"""Precomputed race: our whole pipeline run once over the race, played back by the server.

The detectors, the risk model and our race control (src.racecontrol.engine.RaceControl,
fed the same envelopes in the same order as the race control client) run once over the
whole race, tick by tick in time order, so every envelope only used data up to its own
tick: exactly what the live pipeline sends. The server then plays the ticks and these
envelopes back. A seek lands on the state a continuous run has at that time: the pages
get every detection, rec and official message up to it (src.replay.hub), instead of race
control forgetting the flags still out and the detectors starting cold.

Cached in data/timeline/<race>.json.gz, keyed by the race and every source and model file
that could change the result; a stale cache is rebuilt.

Run: python -m src.replay.timeline 2021_Azerbaijan           (build the cache, about a minute)
     python -m src.replay.timeline 2026_Azerbaijan --holdout
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from src.ingest.holdout import is_holdout_id
from src.racecontrol.engine import RaceControl, race_laps_from_track
from src.replay.engine import Engine, Processor, RaceData

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "timeline"
KEY_PATHS = ("src/detect", "src/predict", "src/racecontrol/engine.py", "src/replay/engine.py",
             "src/replay/pipeline.py", "src/replay/timeline.py", "src/ingest/sectors.py", "data/models")
KEY_SUFFIXES = {".py", ".json", ".joblib", ".txt"}


def shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def cache_key(rid: str, detect: bool, predict: bool) -> str:
    """The code and models by content, the race's own files (a rebuild) by size and mtime."""
    h = hashlib.sha256(f"{rid}|detect={detect}|predict={predict}".encode())
    for rel in KEY_PATHS:
        base = ROOT / rel
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.suffix in KEY_SUFFIXES)
        for p in files:
            h.update(str(p.relative_to(ROOT)).encode())
            h.update(p.read_bytes())
    from src.replay.engine import race_dir
    for p in sorted(race_dir(rid).glob(f"{rid}[._]*")):
        st = p.stat()
        h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def cache_path(rid: str, detect: bool, predict: bool) -> Path:
    tag = "" if detect and predict else f"_detect{int(detect)}_predict{int(predict)}"
    return CACHE / f"{rid}{tag}.json.gz"


def build(race: RaceData, processors: list[Processor]) -> list[list[dict]]:
    """For every tick k, the envelopes our pipeline sends after that tick (detections, risk,
    then the recs race control makes of the tick and of them), in order."""
    eng = Engine(race, processors)
    rc = RaceControl(race_laps=race_laps_from_track(race.track))
    per_tick: list[list[dict]] = []
    for step in eng.steps(eng.t_end):
        out, recs = [], []
        for env in step.envelopes:
            kind = env["kind"]
            if kind == "tick":
                recs += rc.on_tick(env["data"])[0]
            elif kind == "detection":
                out.append(env)
                recs += rc.on_detection(env["data"])
            elif kind == "risk":
                out.append(env)
                recs += rc.on_risk(env["data"])
        per_tick.append(out + [{"kind": "rec", "data": r} for r in recs])
    return per_tick


def load_or_build(race: RaceData, make_processors: Callable[[RaceData], list[Processor]],
                  detect: bool = True, predict: bool = True, log: Callable[[str], None] = print) -> list[list[dict]]:
    key = cache_key(race.race, detect, predict)
    path = cache_path(race.race, detect, predict)
    if path.exists():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("key") == key and len(cached.get("ticks", [])) == len(race.times):
                log(f"timeline: {race.race} from {shown(path)}")
                return cached["ticks"]
        except (OSError, ValueError):
            pass
    log(f"timeline: running our pipeline over all of {race.race} once (about a minute) ...")
    t0 = perf_counter()
    ticks = build(race, make_processors(race))
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump({"key": key, "race": race.race, "ticks": ticks}, f, separators=(",", ":"))
    os.replace(tmp, path)                  # atomic: a half-written cache is never read
    n_recs = sum(e["kind"] == "rec" for tk in ticks for e in tk)
    log(f"timeline: {race.race} built in {perf_counter() - t0:.0f} s, {n_recs} recs, cached in {shown(path)}")
    return ticks


def main() -> None:
    p = argparse.ArgumentParser(description="Build the precomputed timeline cache for one race")
    p.add_argument("race")
    p.add_argument("--holdout", action="store_true", help="allow a holdout race (A7 demo only)")
    p.add_argument("--no-detect", action="store_true")
    p.add_argument("--no-predict", action="store_true")
    a = p.parse_args()
    if is_holdout_id(a.race) and not a.holdout:
        p.error(f"{a.race} is a holdout race: add --holdout")
    from functools import partial

    from src.replay.pipeline import all_processors
    detect, predict = not a.no_detect, not a.no_predict
    load_or_build(RaceData.load(a.race), partial(all_processors, detect=detect, predict=predict), detect, predict)


if __name__ == "__main__":
    main()

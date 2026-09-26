"""Latency benchmark: replay one full training race as fast as possible (no sleeping)
through every pipeline stage that exists, and save the per-stage summary.

Stages are timed per tick by the engine (src.replay.latency): tick, frame, one per
processor (detect today, predict once it is plugged in), send (JSON encoding of the
tick's envelopes, standing in for the WebSocket send) and total (from the moment
the tick is emitted until its envelopes are sent).

The ANOMALY model scores every row once when the race is loaded (each row only
uses that car's past 2 s, so this is still causal). That cost is reported under
setup, plus an estimate of what scoring each tick live would cost.

Run: python -m src.replay.benchmark                  (2023_Australian)
     python -m src.replay.benchmark 2024_Canadian --profile
Writes docs/charts/latency.json. If any stage averages over 50 ms (or with
--profile) it prints the slowest functions from cProfile.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
from pathlib import Path
from time import perf_counter

import numpy as np

from src.ingest.holdout import assert_not_holdout
from src.replay.engine import Engine, RaceData
from src.replay.latency import LatencyTracker, stats

OUT = Path("docs/charts/latency.json")
SLOW_STAGE_MS = 50.0
PROFILE_TICKS = 4000


def build_processors(race: RaceData) -> list:
    from src.detect.pipeline import load_anomaly, load_config
    from src.replay.pipeline import all_processors
    return all_processors(race), load_config(), load_anomaly()


def anomaly_per_tick_ms(race: RaceData, model, n: int = 300) -> dict | None:
    """What scoring the ANOMALY model tick by tick (instead of once at load) would cost."""
    if model is None:
        return None
    ks = np.linspace(0, len(race.times) - 1, n).astype(int)
    samples = []
    for k in ks:
        rows = race.frame.iloc[race.rows(k)]
        t0 = perf_counter()
        model.score(rows)
        samples.append(perf_counter() - t0)
    return stats(samples)


def replay(engine: Engine, tracker: LatencyTracker, limit: int | None = None) -> int:
    n = 0
    for step in engine.steps(engine.t_end):
        t0 = perf_counter()
        for env in step.envelopes:
            json.dumps(env, separators=(",", ":"))
        sent = perf_counter()
        step.stage_s["send"] = sent - t0
        tracker.record(step.stage_s, sent - step.emitted_at)
        tracker.maybe_report()
        n += 1
        if limit and n >= limit:
            break
    engine.finish(engine.t_end)
    return n


def profile(race: RaceData, procs_factory) -> str:
    procs, _, _ = procs_factory(race)
    eng = Engine(race, procs)
    eng.seek(eng.t_start + (eng.t_end - eng.t_start) / 2)
    prof = cProfile.Profile()
    prof.enable()
    replay(eng, LatencyTracker(report_every_s=1e9), limit=PROFILE_TICKS)
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(12)
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser(description="Replay one training race as fast as possible and time each stage")
    p.add_argument("race", nargs="?", default="2023_Australian")
    p.add_argument("--profile", action="store_true", help="always print the cProfile top functions")
    a = p.parse_args()
    assert_not_holdout(rid=a.race)
    t0 = perf_counter()
    race = RaceData.load(a.race)
    t_load = perf_counter() - t0
    t0 = perf_counter()
    procs, cfg, model = build_processors(race)
    t_procs = perf_counter() - t0
    eng = Engine(race, procs)
    tracker = LatencyTracker(label=f"latency {a.race}")
    t0 = perf_counter()
    ticks = replay(eng, tracker)
    wall = perf_counter() - t0
    summary = tracker.summary()
    result = {
        "race": a.race, "note": "replay of historical FastF1 data, as fast as possible (no sleeping)",
        "ticks": ticks, "race_seconds": round(eng.t_end - eng.t_start, 1), "wall_seconds": round(wall, 2),
        "ticks_per_second": round(ticks / wall, 1),
        "times_real_time": round((eng.t_end - eng.t_start) / wall, 1),
        "processors": [type(x).__name__ for x in procs],
        "setup_seconds": {"load_race": round(t_load, 2), "build_processors_incl_anomaly_scoring": round(t_procs, 2)},
        "anomaly": ({"threshold": round(model.threshold, 4), "scored_rows_at_load": len(race.frame),
                     "per_tick_if_scored_live": anomaly_per_tick_ms(race, model)} if model else None),
        **summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(tracker.format(tracker.all, summary["ticks"], summary["over_budget"], wall))
    print(f"{ticks} ticks in {wall:.1f} s = {result['ticks_per_second']} ticks/s "
          f"({result['times_real_time']}x real time), setup {result['setup_seconds']}")
    if result["anomaly"]:
        print(f"ANOMALY scored at load for {len(race.frame)} rows; per tick if scored live: "
              f"{result['anomaly']['per_tick_if_scored_live']}")
    print(f"saved {OUT}")
    slow = {k: v["mean_ms"] for k, v in summary["stages"].items() if k != "total" and v["mean_ms"] > SLOW_STAGE_MS}
    if slow or a.profile:
        print(f"\nstages averaging over {SLOW_STAGE_MS:.0f} ms: {slow or 'none'}; cProfile over {PROFILE_TICKS} ticks:")
        print(profile(race, build_processors))


if __name__ == "__main__":
    main()

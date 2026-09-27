"""Replay engine: plays one built race tick by tick, strictly causal (Section 6.2).

RaceData holds a race built by src.ingest.build. Engine keeps a clock and
`advance(to_t)` returns every envelope with t <= to_t not sent yet, in time
order: official events, ticks, and whatever the processors (detectors, risk
model) emit for each tick. A processor only ever sees the current tick and its
own memory of earlier ticks, so nothing from the future can reach it.

`steps(to_t)` yields the same envelopes one tick at a time (TickStep), with the
time spent in each pipeline stage, so the server and the benchmark can measure
latency per tick (src.replay.latency).

Run: python -m src.replay.engine 2023_Australian   (prints the first few seconds)
"""

from __future__ import annotations

import bisect
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
import pandas as pd

from src.ingest.holdout import is_holdout_id
from src.replay.latency import stage_name

FEATURES = Path("data/features")
HOLDOUT_DIR = Path("data/holdout")
CASE_DIR = Path("data/case_studies")   # replayable, but never listed as training races
WARMUP_S = 30.0          # after a seek, processors silently replay this much history


class CausalityError(RuntimeError):
    """A processor produced data stamped later than the tick it was given."""


class Processor(Protocol):
    def reset(self) -> None: ...

    def on_tick(self, t: float, frame: pd.DataFrame, tick: dict) -> list[dict]:
        """Called once per tick with that tick's rows only. Returns envelopes."""
        ...


def race_dir(rid: str) -> Path:
    if is_holdout_id(rid):
        return HOLDOUT_DIR
    return CASE_DIR if (CASE_DIR / f"{rid}_meta.json").exists() else FEATURES


def available_races(include_holdout: bool = False) -> list[str]:
    dirs = [FEATURES] + ([HOLDOUT_DIR] if include_holdout else [])
    return sorted(p.name[: -len("_meta.json")] for d in dirs if d.exists() for p in d.glob("*_meta.json"))


def num(v: float, nd: int = 1) -> float:
    return round(float(v), nd) if np.isfinite(v) else -1.0


@dataclass
class RaceData:
    race: str
    frame: pd.DataFrame      # one row per (t, drv), sorted by t then drv
    times: np.ndarray        # tick times
    bounds: np.ndarray       # rows of tick k are frame[bounds[k]:bounds[k + 1]]
    track: dict
    official: list[dict]
    meta: dict

    @classmethod
    def from_frame(cls, race: str, frame: pd.DataFrame, track: dict | None = None,
                   official: list[dict] | None = None, meta: dict | None = None) -> "RaceData":
        frame = frame.sort_values(["t", "drv"]).reset_index(drop=True)
        times, starts = np.unique(frame["t"].to_numpy(), return_index=True)
        return cls(race, frame, times, np.append(starts, len(frame)), track or {"race": race},
                   sorted(official or [], key=lambda e: e["t"]), meta or {"race": race})

    @classmethod
    def load(cls, rid: str) -> "RaceData":
        d = race_dir(rid)
        read = lambda suffix: json.loads((d / f"{rid}{suffix}").read_text(encoding="utf-8"))  # noqa: E731
        return cls.from_frame(rid, pd.read_parquet(d / f"{rid}.parquet"), read("_track.json"),
                              read("_official.json"), read("_meta.json"))

    def __post_init__(self) -> None:
        f = self.frame
        self._cols = {c: f[c].to_numpy() for c in
                      ("drv", "x", "y", "dist", "lat_off", "speed", "throttle", "brake", "gear", "rpm",
                       "msector", "gap_ahead_m", "in_pit", "lap", "track_status")}
        c = self._cols
        self._live = np.isfinite(c["x"].astype(float)) & np.isfinite(c["y"].astype(float)) \
            & np.isfinite(c["speed"].astype(float)) & np.isfinite(c["dist"].astype(float)) \
            & np.isfinite(c["gear"].astype(float)) & np.isfinite(c["rpm"].astype(float))
        # the entry list: every car in the race, with live data or not (GET /status)
        self.cars = sorted({str(d) for d in c["drv"]}, key=lambda d: (not d.isdigit(), int(d) if d.isdigit() else 0, d))

    def rows(self, k: int) -> slice:
        return slice(int(self.bounds[k]), int(self.bounds[k + 1]))

    def tick(self, k: int) -> dict:
        """Tick contract (Section 7.1). Cars with no fresh data are left out."""
        c, sl = self._cols, self.rows(k)
        cars = []
        for i in range(sl.start, sl.stop):
            if not self._live[i]:
                continue
            cars.append({"drv": str(c["drv"][i]), "x": num(c["x"][i]), "y": num(c["y"][i]),
                         "dist": num(c["dist"][i]), "lat_off": num(c["lat_off"][i], 2),
                         "speed": num(c["speed"][i]), "throttle": num(c["throttle"][i]),
                         "brake": bool(c["brake"][i] > 0), "gear": int(c["gear"][i]),
                         "rpm": int(c["rpm"][i]), "msector": int(c["msector"][i]),
                         "gap_ahead_m": num(c["gap_ahead_m"][i]), "in_pit": bool(c["in_pit"][i])})
        return {"t": round(float(self.times[k]), 2), "lap": int(c["lap"][sl.start]),
                "track_status": str(c["track_status"][sl.start]), "cars": cars}


def envelope_t(env: dict) -> float:
    return float(env["data"]["t"])


@dataclass
class TickStep:
    t: float
    envelopes: list[dict]        # official events due by this tick, the tick, then processor output
    stage_s: dict[str, float]    # seconds spent in each pipeline stage for this tick
    emitted_at: float            # perf_counter() when the tick was emitted into the pipeline


class Engine:
    def __init__(self, race: RaceData, processors: list[Processor] | None = None) -> None:
        self.race = race
        self.processors = processors or []
        self.official_t = [e["t"] for e in race.official]
        self.k = 0                                   # next tick to play
        self.j = 0                                   # next official event to play
        self.clock = float(race.times[0])
        self.seek(self.clock)

    @property
    def t_start(self) -> float:
        return float(self.race.times[0])

    @property
    def t_end(self) -> float:
        return float(self.race.times[-1])

    @property
    def finished(self) -> bool:
        return self.k >= len(self.race.times)

    def _process(self, k: int, tick: dict, stage_s: dict[str, float] | None = None) -> list[dict]:
        """Run every processor on tick k. Time spent is added to stage_s: 'frame' for
        slicing the rows, then one entry per processor (src.replay.latency.stage_name)."""
        stage_s = {} if stage_s is None else stage_s
        t = float(self.race.times[k])
        out = []
        if not self.processors:
            return out
        t0 = perf_counter()
        frame = self.race.frame.iloc[self.race.rows(k)]
        stage_s["frame"] = stage_s.get("frame", 0.0) + perf_counter() - t0
        for p in self.processors:
            t0 = perf_counter()
            envs = p.on_tick(t, frame, tick)
            name = stage_name(p)
            stage_s[name] = stage_s.get(name, 0.0) + perf_counter() - t0
            for env in envs:
                if envelope_t(env) > t:
                    raise CausalityError(f"{type(p).__name__} emitted t={envelope_t(env)} at tick t={t}")
                out.append(env)
        return out

    def seek(self, t: float) -> None:
        """Jump to t. Processors are reset and warmed up on the WARMUP_S before t
        (history only, nothing is emitted), so stateful detectors work after a seek."""
        t = min(max(t, self.t_start), self.t_end)
        k_new = bisect.bisect_left(self.race.times, t)
        for p in self.processors:
            p.reset()
        for k in range(bisect.bisect_left(self.race.times, t - WARMUP_S), k_new):
            self._process(k, self.race.tick(k))
        self.k, self.j, self.clock = k_new, bisect.bisect_left(self.official_t, t), t

    def steps(self, to_t: float) -> Iterator[TickStep]:
        """One TickStep per tick with t <= to_t not played yet, in time order.
        Call finish(to_t) after consuming it (advance() does both)."""
        times = self.race.times
        while self.k < len(times) and times[self.k] <= to_t:
            t = float(times[self.k])
            envs = self._official_until(t)
            t0 = perf_counter()
            tick = self.race.tick(self.k)
            emitted = perf_counter()
            stage_s = {"tick": emitted - t0}
            envs.append({"kind": "tick", "data": tick})
            envs.extend(self._process(self.k, tick, stage_s))
            self.k += 1
            yield TickStep(t, envs, stage_s, emitted)

    def finish(self, to_t: float) -> list[dict]:
        """Official events after the last tick but <= to_t, and move the clock to to_t."""
        out = self._official_until(to_t)
        self.clock = max(self.clock, min(to_t, self.t_end))
        return out

    def advance(self, to_t: float) -> list[dict]:
        """Every envelope with t <= to_t not sent yet, in time order."""
        out = [env for step in self.steps(to_t) for env in step.envelopes]
        return out + self.finish(to_t)

    def _official_until(self, t: float) -> list[dict]:
        out = []
        while self.j < len(self.official_t) and self.official_t[self.j] <= t:
            out.append({"kind": "official", "data": self.race.official[self.j]})
            self.j += 1
        return out


def main() -> None:
    race = RaceData.load(sys.argv[1] if len(sys.argv) > 1 else "2023_Australian")
    eng = Engine(race)
    envs = eng.advance(eng.t_start + 2.0)
    print(f"{race.race}: {len(race.times)} ticks, t {eng.t_start:.1f} to {eng.t_end:.1f}")
    for env in envs[:3]:
        d = env["data"]
        print(env["kind"], d["t"], f"{len(d.get('cars', []))} cars" if env["kind"] == "tick" else d)


if __name__ == "__main__":
    main()

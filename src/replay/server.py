"""Real server: plays built races over the Section 7.5 contract, same as mock_server.

Run: python -m src.replay.server                          (2023_Australian, with detectors)
     python -m src.replay.server --race 2021_Azerbaijan --speed 0   (the demo, paused)
     python -m src.replay.server --no-detect --no-predict   (ticks and official events only)
     python -m src.replay.server --race 2026_Azerbaijan --holdout   (A7 demo only)
     python -m src.replay.server --live                    (run the pipeline tick by tick)

By default the race is precomputed (src.replay.timeline): our detectors, risk model and
race control run once over the whole race at startup (cached in data/timeline/), strictly
tick by tick, and the server plays the result back. Seeking then shows exactly what a
continuous run shows at that time, and our recs come from the server itself: a separately
started python -m src.racecontrol is not needed, and its recs are ignored. --live runs the
processors as the replay plays instead (recs then come from python -m src.racecontrol),
and prints the per-stage latency every 30 s of wall time (src.replay.latency).

Endpoints (identical to the mock server, plus GET /races):
  ws  /stream        envelopes {"kind": "tick|detection|risk|rec|official", "data": {...}}
                     clients may send {"kind": "rec", ...}: it is rebroadcast to everyone
  ws  /stream?catchup=1   the same, after everything since the last seek (src.replay.hub),
                     so a page opened mid-race shows the flags already out
  POST /replay       {"speed": 10, "seek_t": 4350.0, "race": "2024_Canadian"} (speed 0 pauses)
  GET /track         track map for the loaded race
  GET /official      all official events for the loaded race (dashboard only)
  GET /status        race, replay time, speed, start and end time, the cars in the race
  GET /cars          cars out of the race at the replay time (src.replay.cars)
  GET /races         races available to load
  /                  the dashboard/ folder
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import sys
from time import perf_counter
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.ingest.holdout import is_holdout_id
from src.replay.cars import Cars
from src.replay.engine import Engine, Processor, RaceData, available_races
from src.replay.hub import RESET_JUMP_S, Hub
from src.replay.latency import LatencyTracker
from src.replay.static import NoCacheStaticFiles

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "dashboard"
STEP_S = 0.05            # wall-clock step of the playback loop
DEFAULT_RACE = "2023_Australian"


def no_processors(race: RaceData) -> list[Processor]:
    return []


class LiveReplay:
    def __init__(self, race: RaceData, make_processors: Callable[[RaceData], list[Processor]],
                 allow_holdout: bool = False, timeline: bool = False,
                 flags: tuple[bool, bool] = (True, True)) -> None:
        self.make_processors = make_processors
        self.allow_holdout = allow_holdout
        self.use_timeline = timeline
        self.flags = flags                # (detect, predict), part of the timeline cache key
        self.hub = Hub()
        self.send_lock = asyncio.Lock()           # one tick and its history at a time, never interleaved
        self.speed = 1.0
        self.latency = LatencyTracker()
        self.warned = False
        self.load(race)

    def prepare(self, race: RaceData) -> tuple:
        """Everything a race needs before it can play (slow: the timeline may be built)."""
        if self.use_timeline:
            from src.replay.timeline import load_or_build
            timeline = load_or_build(race, self.make_processors, *self.flags,
                                     log=lambda m: print(m, file=sys.stderr, flush=True))
            engine = Engine(race, [])
        else:
            timeline, engine = None, Engine(race, self.make_processors(race))
        return race, timeline, engine, Cars.from_race(race)

    def install(self, prepared: tuple) -> None:
        """Swap in a prepared race, all at once on the event loop: the playback loop never sees
        one race's ticks with another's timeline."""
        self.race, self.timeline, self.engine, self.cars = prepared
        self.epoch = getattr(self, "epoch", 0) + 1   # a seek or race switch: the playback loop drops its step
        self.tick_t = float(self.race.times[0]) if len(self.race.times) else 0.0
        self.pending: list[dict] | None = None      # history to push after the next tick (a seek)
        self.hub.reset()

    def load(self, race: RaceData) -> None:
        self.install(self.prepare(race))

    def seek(self, t: float) -> None:
        self.epoch += 1
        self.engine.seek(t)
        times = self.race.times
        new_t = float(times[min(self.engine.k, len(times) - 1)]) if len(times) else t
        # the pages' rule, tick time against tick time: the server pushes the history exactly
        # when the pages wipe their state
        if abs(new_t - self.tick_t) > RESET_JUMP_S:
            self.hub.reset()              # the pages and race control wipe their state too
            if self.timeline is not None:
                self.pending = self.history()
        elif self.pending is not None:
            # back near the pages' time before an earlier seek's tick went out: the pages do
            # not reset, so push nothing; the catch-up for a page opened later is rebuilt
            self.pending = None
            if self.timeline is not None:
                self.hub.restart_history(self.history())

    def history(self) -> list[dict]:
        """Everything the pages had by now in a continuous run: official messages, detections
        and recs before the next tick, in the order they were played, then the latest risk
        per car."""
        k, j = self.engine.k, self.engine.j
        off, times = self.race.official, self.race.times
        o = bisect.bisect_left(self.engine.official_t, self.engine.t_start)   # the engine never plays earlier ones
        envs, risk = [], {}
        for i in range(k):
            while o < j and off[o]["t"] <= times[i]:
                envs.append({"kind": "official", "data": off[o]})
                o += 1
            for env in self.timeline[i]:
                if env["kind"] == "risk":
                    risk[str(env["data"].get("drv"))] = env
                else:
                    envs.append(env)
        envs += [{"kind": "official", "data": e} for e in off[o:j]]
        return envs + list(risk.values())

    def extras(self, t: float) -> list[dict]:
        """The precomputed envelopes of the tick at t (none when live)."""
        if self.timeline is None:
            return []
        return self.timeline[bisect.bisect_left(self.race.times, t)]

    async def play(self, envs: list[dict]) -> None:
        """Broadcast one tick's envelopes. After a seek in a precomputed race the tick goes
        first (every page resets on the jump), then the catch-up pages get the history,
        then the rest."""
        async with self.send_lock:
            pending, self.pending = self.pending, None     # taken before any await: a later seek sets its own
            if pending is not None:
                for env in envs:
                    if env["kind"] == "tick":
                        self.tick_t = float(env["data"]["t"])
                        await self.hub.broadcast(env)
                await self.hub.push_catchup(self.hub.restart_history(pending))
                envs = [env for env in envs if env["kind"] != "tick"]
            for env in envs:
                if env["kind"] == "tick":
                    self.tick_t = float(env["data"]["t"])
                await self.hub.broadcast(env)

    async def client_rec(self, data: dict) -> None:
        """A rec from a race control client: rebroadcast when live. In a precomputed race
        our recs come from the timeline, so a second race control would double them."""
        if self.timeline is None:
            await self.hub.broadcast({"kind": "rec", "data": data})
        elif not self.warned:
            self.warned = True
            print("precomputed race: ignoring recs from a separate race control client "
                  "(not needed; use --live to take them)", file=sys.stderr, flush=True)

    def switch(self, rid: str) -> tuple:
        """Load and prepare another race (run in a worker thread), then install() it."""
        if is_holdout_id(rid) and not self.allow_holdout:
            raise HTTPException(403, f"{rid} is a holdout race: start the server with --holdout")
        if rid not in available_races(include_holdout=self.allow_holdout):
            raise HTTPException(404, f"race {rid} is not built (python -m src.ingest.build {rid})")
        return self.prepare(RaceData.load(rid))

    async def run(self) -> None:
        while True:
            await asyncio.sleep(STEP_S)
            if self.speed <= 0 or self.engine.finished:
                self.latency.maybe_report()
                continue
            to_t = self.engine.clock + STEP_S * self.speed
            epoch = self.epoch
            for step in self.engine.steps(to_t):
                t0 = perf_counter()
                await self.play(step.envelopes + self.extras(step.t))
                if self.epoch != epoch:
                    break                 # a seek came in while sending: never play on to the old to_t
                sent = perf_counter()
                step.stage_s["send"] = sent - t0
                if self.timeline is None:
                    self.latency.record(step.stage_s, sent - step.emitted_at)
            if self.epoch != epoch:
                continue
            for env in self.engine.finish(to_t):
                await self.hub.broadcast(env)
            self.latency.maybe_report()

    async def preview(self) -> None:
        """While paused, send the tick at the current position, so a seek shows at once
        (the replay does not advance, so nothing else would be sent). No processor runs
        on it: detections resume from here when playback does."""
        if len(self.race.times):
            k = min(self.engine.k, len(self.race.times) - 1)
            await self.play([{"kind": "tick", "data": self.race.tick(k)}])

    def status(self) -> dict:
        e = self.engine
        return {"race": self.race.race, "t": round(e.clock, 2), "speed": self.speed,
                "t_start": e.t_start, "t_end": e.t_end, "finished": e.finished,
                "clients": len(self.hub.clients), "cars": self.race.cars,
                "mode": "live" if self.timeline is None else "precomputed"}


def create_app(race: RaceData | None = None, make_processors: Callable[[RaceData], list[Processor]] = no_processors,
               allow_holdout: bool = False, autoplay: bool = True, timeline: bool = False,
               flags: tuple[bool, bool] = (True, True)) -> FastAPI:
    replay = LiveReplay(race or RaceData.load(DEFAULT_RACE), make_processors, allow_holdout, timeline, flags)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(replay.run()) if autoplay else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Fast Flag server", lifespan=lifespan)
    app.state.replay = replay

    @app.websocket("/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            await replay.hub.join(ws, catchup=ws.query_params.get("catchup") == "1")
        except Exception:
            replay.hub.leave(ws)          # it went away during its catch-up
            return
        try:
            while True:
                env = json.loads(await ws.receive_text())
                if isinstance(env, dict) and env.get("kind") == "rec" and isinstance(env.get("data"), dict):
                    await replay.client_rec(env["data"])
        except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError):
            pass
        finally:
            replay.hub.leave(ws)

    @app.post("/replay")
    async def control(body: dict) -> dict:
        if body.get("race") and body["race"] != replay.race.race:
            replay.install(await asyncio.to_thread(replay.switch, str(body["race"])))   # loading takes seconds
        if "speed" in body:
            replay.speed = float(body["speed"])
        if body.get("seek_t") is not None:
            replay.seek(float(body["seek_t"]))
        if replay.speed <= 0 and (body.get("seek_t") is not None or body.get("race")):
            await replay.preview()
        return replay.status()

    @app.get("/status")
    async def status() -> dict:
        return replay.status()

    @app.get("/races")
    async def races() -> list[str]:
        return available_races(include_holdout=replay.allow_holdout)

    @app.get("/track")
    async def track() -> dict:
        return replay.race.track

    @app.get("/official")
    async def official() -> list[dict]:
        return replay.race.official

    @app.get("/cars")
    async def cars() -> dict:
        return replay.cars.at(replay.tick_t)

    if (DASHBOARD / "index.html").exists():
        app.mount("/", NoCacheStaticFiles(directory=DASHBOARD, html=True), name="dashboard")
    else:
        @app.get("/", response_class=HTMLResponse)
        async def placeholder() -> str:
            return "<h1>Fast Flag server</h1><p>dashboard/index.html not built yet.</p>"

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--race", default=DEFAULT_RACE)
    p.add_argument("--holdout", action="store_true", help="allow loading holdout races (A7 demo only)")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--no-detect", action="store_true", help="no detection envelopes")
    p.add_argument("--no-predict", action="store_true", help="no risk envelopes")
    p.add_argument("--live", action="store_true", help="run the processors as the replay plays (no precomputed race)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    if is_holdout_id(a.race) and not a.holdout:
        p.error(f"{a.race} is a holdout race: add --holdout")
    from functools import partial

    from src.replay.pipeline import all_processors
    flags = (not a.no_detect, not a.no_predict)
    make = partial(all_processors, detect=flags[0], predict=flags[1])
    app = create_app(RaceData.load(a.race), make_processors=make, allow_holdout=a.holdout, timeline=not a.live,
                     flags=flags)
    app.state.replay.speed = a.speed
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()

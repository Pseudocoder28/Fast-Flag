"""Real server: plays built races over the Section 7.5 contract, same as mock_server.

Run: python -m src.replay.server                          (2023_Australian, with detectors)
     python -m src.replay.server --race 2024_Canadian
     python -m src.replay.server --no-detect                (ticks and official events only)
     python -m src.replay.server --race 2026_Azerbaijan --holdout   (A7 demo only)

Endpoints (identical to the mock server, plus GET /races):
  ws  /stream        envelopes {"kind": "tick|detection|risk|rec|official", "data": {...}}
                     clients may send {"kind": "rec", ...}: it is rebroadcast to everyone
  POST /replay       {"speed": 10, "seek_t": 4350.0, "race": "2024_Canadian"} (speed 0 pauses)
  GET /track         track map for the loaded race
  GET /official      all official events for the loaded race (dashboard only)
  GET /status        race, replay time, speed, start and end time
  GET /races         races available to load
  /                  the dashboard/ folder
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.ingest.holdout import is_holdout_id
from src.replay.engine import Engine, Processor, RaceData, available_races

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "dashboard"
STEP_S = 0.05            # wall-clock step of the playback loop
DEFAULT_RACE = "2023_Australian"


def no_processors(race: RaceData) -> list[Processor]:
    return []


class Hub:
    """Connected WebSocket clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def broadcast(self, env: dict) -> None:
        msg = json.dumps(env, separators=(",", ":"))
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


class LiveReplay:
    def __init__(self, race: RaceData, make_processors: Callable[[RaceData], list[Processor]],
                 allow_holdout: bool = False) -> None:
        self.make_processors = make_processors
        self.allow_holdout = allow_holdout
        self.hub = Hub()
        self.speed = 1.0
        self.load(race)

    def load(self, race: RaceData) -> None:
        self.race = race
        self.engine = Engine(race, self.make_processors(race))

    def switch(self, rid: str) -> None:
        if is_holdout_id(rid) and not self.allow_holdout:
            raise HTTPException(403, f"{rid} is a holdout race: start the server with --holdout")
        if rid not in available_races(include_holdout=self.allow_holdout):
            raise HTTPException(404, f"race {rid} is not built (python -m src.ingest.build {rid})")
        self.load(RaceData.load(rid))

    async def run(self) -> None:
        while True:
            await asyncio.sleep(STEP_S)
            if self.speed <= 0 or self.engine.finished:
                continue
            for env in self.engine.advance(self.engine.clock + STEP_S * self.speed):
                await self.hub.broadcast(env)

    def status(self) -> dict:
        e = self.engine
        return {"race": self.race.race, "t": round(e.clock, 2), "speed": self.speed,
                "t_start": e.t_start, "t_end": e.t_end, "finished": e.finished,
                "clients": len(self.hub.clients)}


def create_app(race: RaceData | None = None, make_processors: Callable[[RaceData], list[Processor]] = no_processors,
               allow_holdout: bool = False, autoplay: bool = True) -> FastAPI:
    replay = LiveReplay(race or RaceData.load(DEFAULT_RACE), make_processors, allow_holdout)

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
        replay.hub.clients.add(ws)
        try:
            while True:
                env = json.loads(await ws.receive_text())
                if isinstance(env, dict) and env.get("kind") == "rec" and isinstance(env.get("data"), dict):
                    await replay.hub.broadcast({"kind": "rec", "data": env["data"]})
        except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError):
            pass
        finally:
            replay.hub.clients.discard(ws)

    @app.post("/replay")
    async def control(body: dict) -> dict:
        if body.get("race") and body["race"] != replay.race.race:
            await asyncio.to_thread(replay.switch, str(body["race"]))   # loading + scoring takes seconds
        if "speed" in body:
            replay.speed = float(body["speed"])
        if body.get("seek_t") is not None:
            replay.engine.seek(float(body["seek_t"]))
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

    if (DASHBOARD / "index.html").exists():
        app.mount("/", StaticFiles(directory=DASHBOARD, html=True), name="dashboard")
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
    p.add_argument("--no-detect", action="store_true", help="stream ticks and official events only")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    if is_holdout_id(a.race) and not a.holdout:
        p.error(f"{a.race} is a holdout race: add --holdout")
    make = no_processors
    if not a.no_detect:
        from src.detect.pipeline import detection_processors
        make = detection_processors
    app = create_app(RaceData.load(a.race), make_processors=make, allow_holdout=a.holdout)
    app.state.replay.speed = a.speed
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()

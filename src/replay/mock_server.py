"""Mock server: plays the fixtures over the real contract (Section 7.5 and 7.6).

Run:  python -m src.replay.mock_server            (fixture recs included)
      python -m src.replay.mock_server --no-recs  (B's racecontrol supplies recs)

Endpoints:
  ws  /stream        envelopes {"kind": "tick|detection|risk|rec|official", "data": {...}}
                     clients may send {"kind": "rec", ...}: it is rebroadcast to everyone
  ws  /stream?catchup=1   the same, after everything since the last seek or loop
                     (src.replay.hub), so a page opened mid-race shows the flags already out
  POST /replay       {"speed": 10, "seek_t": 4350.0, "race": "..."} (speed 0 pauses)
  GET /track         track map for the loaded race
  GET /official      all official events for the loaded race
  GET /status        current replay time, speed, clients, the cars in the race
  GET /cars          cars out of the race at the replay time (src.replay.cars)

A seek sends the tick at the new time first, then everything before it (the fixtures are a
precomputed race) to the pages that asked for catch-up, like the real server.
  GET /races         races available to load (the fixture race only)
  /                  the dashboard/ folder
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.replay.cars import Cars
from src.replay.hub import RESET_JUMP_S, Hub
from src.replay.static import NoCacheStaticFiles

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "fixtures"
DASHBOARD = ROOT / "dashboard"
STEP_S = 0.05            # wall-clock step of the playback loop
LOOP_PAUSE_S = 2.0       # pause before looping back to the start
STREAMS = {"tick": "ticks_sample.jsonl", "detection": "detections_sample.jsonl",
           "risk": "risk_sample.jsonl", "rec": "recs_sample.jsonl",
           "official": "official_sample.jsonl"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_events(no_recs: bool) -> list[tuple[float, int, dict]]:
    """All fixture rows as (t, order, envelope), sorted by time."""
    events = []
    for kind, name in STREAMS.items():
        if kind == "rec" and no_recs:
            continue
        for row in read_jsonl(FIX / name):
            events.append((float(row["t"]), len(events), {"kind": kind, "data": row}))
    events.sort(key=lambda e: (e[0], e[1]))
    return events


class Replay:
    def __init__(self, no_recs: bool = False, loop: bool = True) -> None:
        self.events = load_events(no_recs)
        self.times = [e[0] for e in self.events]
        self.t0, self.t1 = self.times[0], self.times[-1]
        self.sim_t = self.t0
        self.cursor = 0
        self.speed = 1.0
        self.loop = loop
        self.track = json.loads((FIX / "track_sample.json").read_text(encoding="utf-8"))
        self.official = read_jsonl(FIX / "official_sample.jsonl")
        self.out = Cars.from_ticks([env["data"] for _, _, env in self.events if env["kind"] == "tick"])
        self.cars = self.out.ids
        self.tick_t = self.t0
        self.pending: list[dict] | None = None      # history to push after the next tick (a seek)
        self.epoch = 0                              # a seek: a play loop in progress stops
        self.hub = Hub()

    def seek(self, t: float) -> None:
        t = min(max(t, self.t0), self.t1)
        self.sim_t = t
        self.cursor = bisect.bisect_left(self.times, self.sim_t)
        self.epoch += 1
        nxt = next((e[0] for e in self.events[self.cursor:] if e[2]["kind"] == "tick"), t)
        if abs(nxt - self.tick_t) > RESET_JUMP_S:   # the pages' rule: tick time against tick time
            self.hub.reset()              # the pages and race control wipe their state too
            self.pending = self.history()

    def history(self) -> list[dict]:
        """Every official message, detection and rec before the cursor, then the latest risk per car."""
        envs, risk = [], {}
        for _, _, env in self.events[:self.cursor]:
            if env["kind"] == "risk":
                risk[str(env["data"].get("drv"))] = env
            elif env["kind"] != "tick":
                envs.append(env)
        return envs + list(risk.values())

    async def play(self, envs: list[dict]) -> None:
        """Broadcast in order; after a seek the first tick goes out, then the history to the
        catch-up pages, then the rest."""
        epoch = self.epoch
        for env in envs:
            if self.epoch != epoch:
                return                    # a seek came in while sending: the rest is the old position
            if env["kind"] == "tick":
                self.tick_t = float(env["data"]["t"])
            await self.hub.broadcast(env)
            if env["kind"] == "tick" and self.pending is not None:
                await self.hub.push_catchup(self.hub.restart_history(self.pending))
                self.pending = None

    def due(self) -> list[dict]:
        """Envelopes with t <= sim_t not yet sent. Never anything from the future."""
        out = []
        while self.cursor < len(self.events) and self.events[self.cursor][0] <= self.sim_t:
            out.append(self.events[self.cursor][2])
            self.cursor += 1
        return out

    async def run(self) -> None:
        while True:
            await asyncio.sleep(STEP_S)
            if self.speed <= 0:
                continue
            self.sim_t += STEP_S * self.speed
            await self.play(self.due())
            if self.cursor >= len(self.events) and self.loop:
                await asyncio.sleep(LOOP_PAUSE_S)
                self.seek(self.t0)

    async def preview(self) -> None:
        """While paused, send the next tick from the current position, so a seek shows at
        once. The cursor does not move: playback sends it again when it resumes."""
        for _, _, env in self.events[self.cursor:]:
            if env["kind"] == "tick":
                await self.play([env])
                return

    def status(self) -> dict:
        return {"race": self.track["race"], "t": round(self.sim_t, 2), "speed": self.speed,
                "t_start": self.t0, "t_end": self.t1, "clients": len(self.hub.clients), "cars": self.cars}


def create_app(no_recs: bool = False, loop: bool = True, autoplay: bool = True) -> FastAPI:
    replay = Replay(no_recs=no_recs, loop=loop)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(replay.run()) if autoplay else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Fast Flag mock server", lifespan=lifespan)
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
                    await replay.hub.broadcast({"kind": "rec", "data": env["data"]})
        except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError):
            pass
        finally:
            replay.hub.leave(ws)

    @app.post("/replay")
    async def control(body: dict) -> dict:
        if "speed" in body:
            replay.speed = float(body["speed"])
        if body.get("seek_t") is not None:
            replay.seek(float(body["seek_t"]))
            if replay.speed <= 0:
                await replay.preview()
        return replay.status()

    @app.get("/status")
    async def status() -> dict:
        return replay.status()

    @app.get("/races")
    async def races() -> list[str]:
        return [replay.track["race"]]

    @app.get("/track")
    async def track() -> dict:
        return replay.track

    @app.get("/official")
    async def official() -> list[dict]:
        return replay.official

    @app.get("/cars")
    async def cars() -> dict:
        return replay.out.at(replay.tick_t)

    if (DASHBOARD / "index.html").exists():
        app.mount("/", NoCacheStaticFiles(directory=DASHBOARD, html=True), name="dashboard")
    else:
        @app.get("/", response_class=HTMLResponse)
        async def placeholder() -> str:
            return "<h1>Fast Flag mock server</h1><p>dashboard/index.html not built yet.</p>"

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-recs", action="store_true", help="do not play fixture recs")
    p.add_argument("--no-loop", action="store_true", help="stop at the end instead of looping")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    uvicorn.run(create_app(no_recs=a.no_recs, loop=not a.no_loop), host=a.host, port=a.port)


if __name__ == "__main__":
    main()

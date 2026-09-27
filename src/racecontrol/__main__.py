"""Race control client: feeds the WebSocket stream into RaceControl, sends recs back.

Run:  python -m src.racecontrol
      python -m src.racecontrol --url ws://localhost:8001/stream
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request

import websockets

from src.racecontrol.engine import RaceControl, race_laps_from_track

URL = "ws://localhost:8000/stream"
ws_url = URL             # the stream this client joined: GET /track comes from the same server
BACKOFF_START_S = 1.0
BACKOFF_CAP_S = 5.0      # local server: after a restart, recs flow again within 5 s


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


async def handle_message(rc: RaceControl, raw: str, ws) -> None:
    try:
        env = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(env, dict):
        return

    kind = env.get("kind")
    if kind in ("rec", "official"):
        return  # our own echo / dashboard-only, not for the engine

    data = env.get("data")
    if not isinstance(data, dict):
        return

    if kind == "tick":
        recs, did_reset = rc.on_tick(data)
        if did_reset:
            log(f"RESET at t={data['t']}")
            rc.race_laps = await race_laps(ws_url)   # a seek or loop may come with another race
    elif kind == "detection":
        recs = rc.on_detection(data)
    elif kind == "risk":
        recs = rc.on_risk(data)
    else:
        return

    for rec in recs:
        log(f"REC t={rec['t']:.2f} msector={rec['msector']} flag={rec['flag']} reason={rec['reason']}")
        await ws.send(json.dumps({"kind": "rec", "data": rec}, separators=(",", ":")))


async def race_laps(url: str) -> int | None:
    """The race length for the late-race red flag, from GET /track on the same server (the
    fewest laps that cover the race distance, known before the race). None if the server
    has no track yet: then there is no late-race red, everything else works."""
    track_url = url.replace("wss://", "https://").replace("ws://", "http://").rsplit("/", 1)[0] + "/track"

    def get() -> dict:
        with urllib.request.urlopen(track_url, timeout=5) as r:
            return json.loads(r.read())

    try:
        laps = race_laps_from_track(await asyncio.to_thread(get))
    except Exception as exc:
        log(f"no race length from {track_url} ({exc!r}): late-race red flag off")
        return None
    log(f"race length about {laps} laps (from the lap length): late-race red flag on")
    return laps


async def run_client(url: str) -> None:
    global ws_url
    ws_url = url
    rc = RaceControl()
    backoff = BACKOFF_START_S
    while True:
        try:
            async with websockets.connect(url) as ws:
                log(f"connected to {url}")
                rc.race_laps = await race_laps(url)
                backoff = BACKOFF_START_S
                async for raw in ws:
                    await handle_message(rc, raw, ws)
                log("server closed the connection")
        except Exception as exc:
            log(f"connection error: {exc!r}")
        log(f"reconnecting in {backoff:.0f}s")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, BACKOFF_CAP_S)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=URL, help=f"stream to join (default {URL})")
    a = p.parse_args()
    try:
        asyncio.run(run_client(a.url))
    except KeyboardInterrupt:
        log("shutting down (Ctrl+C)")


if __name__ == "__main__":
    main()

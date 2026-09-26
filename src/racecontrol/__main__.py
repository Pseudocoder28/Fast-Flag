"""Race control client: feeds the WebSocket stream into RaceControl, sends recs back.

Run:  python -m src.racecontrol
"""

from __future__ import annotations

import asyncio
import json
import sys

import websockets

from src.racecontrol.engine import RaceControl

URL = "ws://localhost:8000/stream"
BACKOFF_START_S = 1.0
BACKOFF_CAP_S = 30.0


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
    elif kind == "detection":
        recs = rc.on_detection(data)
    elif kind == "risk":
        recs = rc.on_risk(data)
    else:
        return

    for rec in recs:
        log(f"REC t={rec['t']:.2f} msector={rec['msector']} flag={rec['flag']} reason={rec['reason']}")
        await ws.send(json.dumps({"kind": "rec", "data": rec}, separators=(",", ":")))


async def run_client(url: str) -> None:
    rc = RaceControl()
    backoff = BACKOFF_START_S
    while True:
        try:
            async with websockets.connect(url) as ws:
                log(f"connected to {url}")
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
    try:
        asyncio.run(run_client(URL))
    except KeyboardInterrupt:
        log("shutting down (Ctrl+C)")


if __name__ == "__main__":
    main()

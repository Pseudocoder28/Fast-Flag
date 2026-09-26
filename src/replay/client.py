"""Small stream client for checking a running server (real or mock).

Run: python -m src.replay.client                    (listen 5 s on ws://localhost:8000/stream)
     python -m src.replay.client --seconds 10 --url ws://127.0.0.1:8765/stream

Prints envelope counts per kind, ticks per wall second, how fast replay time
advanced, and the first detection / rec / official seen.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter

import websockets

TICK_KEYS = {"t", "lap", "track_status", "cars"}
CAR_KEYS = {"drv", "x", "y", "dist", "lat_off", "speed", "throttle", "brake", "gear", "rpm",
            "msector", "gap_ahead_m", "in_pit"}


async def listen(url: str, seconds: float) -> dict:
    counts: Counter = Counter()
    first: dict[str, dict] = {}
    tick_ts: list[float] = []
    bad = 0
    async with websockets.connect(url, max_size=None) as ws:
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            try:
                env = json.loads(await asyncio.wait_for(ws.recv(), timeout=left))
            except asyncio.TimeoutError:
                break
            kind, data = env.get("kind"), env.get("data", {})
            counts[kind] += 1
            first.setdefault(kind, data)
            if kind == "tick":
                tick_ts.append(data["t"])
                if not TICK_KEYS <= data.keys() or any(not CAR_KEYS <= c.keys() for c in data["cars"]):
                    bad += 1
    return {"counts": counts, "first": first, "tick_ts": tick_ts, "bad": bad}


def report(r: dict, seconds: float) -> None:
    ts = r["tick_ts"]
    print("envelopes:", dict(r["counts"]))
    if ts:
        span = ts[-1] - ts[0]
        print(f"ticks: {len(ts) / seconds:.1f} per wall second, replay time {ts[0]:.2f} -> {ts[-1]:.2f} "
              f"({span / seconds:.1f}x real time), malformed ticks: {r['bad']}")
        print(f"cars in first tick: {len(r['first']['tick']['cars'])}")
    for kind in ("detection", "risk", "rec", "official"):
        if kind in r["first"]:
            print(f"first {kind}: {json.dumps(r['first'][kind])[:160]}")


def main() -> None:
    p = argparse.ArgumentParser(description="Listen to the stream and summarise it")
    p.add_argument("--url", default="ws://127.0.0.1:8000/stream")
    p.add_argument("--seconds", type=float, default=5.0)
    a = p.parse_args()
    report(asyncio.run(listen(a.url, a.seconds)), a.seconds)


if __name__ == "__main__":
    main()

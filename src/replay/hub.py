"""Connected WebSocket clients, and what a page that opens mid-race needs to catch up.

Recs only arrive when a flag changes, so a page opened mid-race (or reopened: switching
between the pit wall and the overlay reloads the page) would show no flag until the next
change. The hub keeps every detection, rec and official envelope since the last reset (a
seek over RESET_JUMP_S, a loop or a race switch), the latest risk per car and the latest
tick. A client that connects with ?catchup=1 gets those first, in that order, then
everything live. Clients without it (race control, the serial bridge, the voice) get live
envelopes only, as before. Both servers use it.

In a precomputed race (src.replay.timeline) a seek does not wipe the state: the server
sends the tick at the new time first (every page resets on the jump), then pushes every
detection, rec and official message up to that time to the catch-up clients only, so a
page after a seek shows exactly what it would after playing continuously to that time.
"""

from __future__ import annotations

import json
from collections import deque

from fastapi import WebSocket

RESET_JUMP_S = 2.0                                # same rule as the race control engine and the pages
CATCHUP_KINDS = ("detection", "rec", "official")  # kept in full since the last reset


class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.joining: dict[WebSocket, deque[str]] = {}   # clients still receiving their catch-up
        self.catchup_clients: set[WebSocket] = set()     # clients that asked for catch-up (the pages)
        self.history: list[str] = []
        self.risk: dict[str, str] = {}                   # drv -> latest risk envelope
        self.tick: str | None = None

    def reset(self) -> None:
        """A seek, loop or race switch: every client wipes its state at the next tick."""
        self.history.clear()
        self.risk.clear()
        self.tick = None

    def catchup(self) -> list[str]:
        return [*self.history, *self.risk.values(), *([self.tick] if self.tick else [])]

    def remember(self, env: dict, msg: str) -> None:
        kind = env.get("kind")
        if kind in CATCHUP_KINDS:
            self.history.append(msg)
        elif kind == "risk":
            self.risk[str(env.get("data", {}).get("drv"))] = msg
        elif kind == "tick":
            self.tick = msg

    async def broadcast(self, env: dict) -> None:
        msg = json.dumps(env, separators=(",", ":"))
        self.remember(env, msg)
        for queue in self.joining.values():
            queue.append(msg)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def join(self, ws: WebSocket, catchup: bool = False) -> None:
        """Register a client. With catchup, it first gets what it missed; envelopes broadcast
        meanwhile queue behind, so it sees each one once and in order."""
        if catchup:
            self.catchup_clients.add(ws)
            queue = self.joining[ws] = deque(self.catchup())
            try:
                while queue:
                    await ws.send_text(queue.popleft())
            finally:
                del self.joining[ws]
        self.clients.add(ws)              # no await since the queue ran empty: nothing is lost

    def leave(self, ws: WebSocket) -> None:
        self.clients.discard(ws)
        self.joining.pop(ws, None)
        self.catchup_clients.discard(ws)

    def restart_history(self, envs: list[dict]) -> list[str]:
        """After a seek in a precomputed race: everything up to the new time becomes the
        history (the tick at the new time, already broadcast, stays the latest tick)."""
        self.history.clear()
        self.risk.clear()
        msgs = []
        for env in envs:
            msg = json.dumps(env, separators=(",", ":"))
            self.remember(env, msg)
            msgs.append(msg)
        return msgs

    async def push_catchup(self, msgs: list[str]) -> None:
        """Send msgs to the catch-up clients only, in order (after the tick they reset on)."""
        for queue in self.joining.values():
            queue.extend(msgs)
        dead = []
        for ws in list(self.clients):
            if ws not in self.catchup_clients:
                continue
            try:
                for msg in msgs:
                    await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.leave(ws)

"""Race control voice: speaks Fast Flag recommendations like race control radio.

Subscribes to the stream and speaks only when a sector's flag level goes up
(YELLOW, DOUBLE_YELLOW, VSC, SC, RED), plus a short "Sector 7 clear" when it
returns to CLEAR. When the official message for a flag we already raised arrives
later, it says "Race control confirms ..., X.X seconds after Fast Flag". When
race control was first, nothing is said about lead time. Every phrase comes from
a template: flag, car numbers, detection type and sector. Nothing here decides
anything, it only reads out the rec envelopes it receives.

Speech uses macOS `say` in a non-blocking subprocess. A higher flag level
interrupts the current phrase, everything else waits for the current phrase and
for a 2 s gap. On a non-Mac (or with --print) the lines are printed instead.

Run: python -m src.lab.voice [--url ws://localhost:8000/stream] [--voice Daniel] [--print]
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Protocol

from src.ingest.sectors import sector_matches

URL = "ws://localhost:8000/stream"
VOICE = "Daniel"
RESET_JUMP_S = 2.0            # same rule as the race control engine and the dashboard
MIN_GAP_S = 2.0               # at most one non-interrupting call every 2 s
MAX_WAIT_S = 12.0             # a phrase still queued after this long (wall time) is stale
CONFIRM_WINDOW_S = 120.0      # an official message confirms a flag we raised within this long before it
STEP_S = 0.1
BACKOFF_START_S, BACKOFF_CAP_S = 1.0, 30.0

RANK = {"CLEAR": 0, "YELLOW": 1, "DOUBLE_YELLOW": 2, "VSC": 3, "SC": 4, "RED": 5}
GLOBAL_FLAGS = {"VSC", "SC", "RED"}
FLAG_WORDS = {"YELLOW": "Yellow flag", "DOUBLE_YELLOW": "Double yellow", "VSC": "Virtual Safety Car",
              "SC": "Safety Car", "RED": "Red flag"}
TYPE_WORDS = {"IMPACT": "impact", "STOPPED": "stopped", "SPIN": "spin", "DROPOUT": "no data",
              "MULTI": "multi-car", "ANOMALY": "anomaly"}
TRACK = "track"               # scope key for track-wide flags (VSC, SC, RED)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---- templates ---------------------------------------------------------------

def cars_phrase(cars: list[str]) -> str:
    if not cars:
        return ""
    if len(cars) == 1:
        return f"Car {cars[0]}"
    return f"Cars {', '.join(cars[:-1])} and {cars[-1]}"


def flag_phrase(flag: str, cars: list[str], dtype: str | None, msector: int) -> str:
    """'Safety Car. Car 23, impact, sector 7.'"""
    parts = [p for p in (cars_phrase(cars), TYPE_WORDS.get(dtype or ""), f"sector {msector}") if p]
    detail = ", ".join(parts)
    return f"{FLAG_WORDS[flag]}. {detail[0].upper() + detail[1:]}."


def clear_phrase(msector: int | None) -> str:
    return "Track clear." if msector is None else f"Sector {msector} clear."


def confirm_phrase(flag: str, lead_s: float) -> str:
    return f"Race control confirms {FLAG_WORDS[flag]}, {lead_s:.1f} seconds after Fast Flag."


def sort_car(drv: str) -> tuple[int, str]:
    return (int(drv), drv) if drv.isdigit() else (10**6, drv)


# ---- narrator: envelopes in, phrases out ----------------------------------------

@dataclass(order=True)
class Utterance:
    """What to say and how urgently. Heap order: highest rank first, then oldest."""
    sort_key: tuple[int, int] = field(init=False, repr=False)
    text: str = field(compare=False)
    rank: int = field(compare=False, default=0)
    interrupt: bool = field(compare=False, default=False)   # a higher level cuts current speech
    key: str | None = field(compare=False, default=None)     # queued lower phrases with this key are dropped
    seq: int = field(compare=False, default=0)

    def __post_init__(self) -> None:
        self.sort_key = (-self.rank, self.seq)


class Narrator:
    """Turns the stream into utterances. Pure logic, no I/O, so it is testable.

    Levels are tracked per sector (YELLOW, DOUBLE_YELLOW) and track-wide (VSC, SC,
    RED). An episode is one run of a scope from CLEAR up and back: its raises are
    kept so a later official message can be matched against when we first reached
    that level.
    """

    def __init__(self) -> None:
        self.seq = 0
        self.reset()

    def reset(self) -> None:
        self.t: float | None = None
        self.detections: dict[str, dict] = {}
        self.level: dict[int | str, int] = {}                        # msector or TRACK -> current rank
        self.episode: dict[int | str, list[tuple[float, int]]] = {}  # scope -> [(t, rank)] of this episode
        self.sector_cause: dict[int, tuple[list[str], str | None]] = {}
        self.confirmed: set[tuple[int | str, int, float]] = set()
        self.max_sector = 0

    # -- entry points

    def on_raw(self, raw: str) -> list[Utterance]:
        try:
            env = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        return self.on_envelope(env)

    def on_envelope(self, env: object) -> list[Utterance]:
        if not isinstance(env, dict) or not isinstance(env.get("data"), dict):
            return []
        try:
            handler = {"tick": self.on_tick, "detection": self.on_detection, "rec": self.on_rec,
                       "official": self.on_official}.get(str(env.get("kind")))
            return handler(env["data"]) if handler else []
        except (KeyError, TypeError, ValueError) as exc:     # a malformed envelope never kills the voice
            log(f"voice: skipped {env.get('kind')} envelope ({exc!r})")
            return []

    def on_tick(self, tick: dict) -> list[Utterance]:
        t = float(tick["t"])
        if self.t is not None and t < self.t - RESET_JUMP_S:
            self.reset()
        self.t = t
        for car in tick.get("cars", []):
            self.max_sector = max(self.max_sector, int(car["msector"]))
        return []

    def on_detection(self, det: dict) -> list[Utterance]:
        self.detections[str(det["id"])] = det
        return []

    def on_rec(self, rec: dict) -> list[Utterance]:
        flag, msector, t = str(rec["flag"]), int(rec["msector"]), float(rec["t"])
        if flag not in RANK:
            return []
        if flag == "CLEAR":
            return self.on_clear(rec, msector)
        cars, dtype = self.cause(rec, msector)
        rank = RANK[flag]
        scope: int | str = TRACK if flag in GLOBAL_FLAGS else msector
        if rank <= self.level.get(scope, 0):
            return []
        self.raise_scope(scope, t, rank)
        if scope == TRACK:
            self.raise_scope(msector, t, rank)      # the cause sector is covered by the track-wide flag too
        return [self.utterance(flag_phrase(flag, cars, dtype, msector), rank, interrupt=True,
                               key=f"{scope}")]

    def on_clear(self, rec: dict, msector: int) -> list[Utterance]:
        msg = str(rec.get("message", "")).upper()
        sector_clear = "SECTOR" in msg or (self.level.get(TRACK, 0) == 0 and self.level.get(msector, 0) > 0)
        scope: int | str = msector if sector_clear else TRACK
        if self.level.get(scope, 0) == 0:
            return []
        self.level[scope] = 0
        if sector_clear:
            self.sector_cause.pop(msector, None)
        return [self.utterance(clear_phrase(msector if sector_clear else None), 0, key=f"{scope}")]

    def on_official(self, off: dict) -> list[Utterance]:
        flag = str(off["flag"])
        if flag not in RANK or flag == "CLEAR":
            return []
        rank, t = RANK[flag], float(off["t"])
        msector = off.get("msector")
        scope: int | str = TRACK if msector is None else int(msector)
        found = self.our_first(rank, scope, t)
        if found is None:
            return []
        ours, matched = found
        if ours > t:                  # race control was first: nothing about lead time
            return []
        if (matched, rank, ours) in self.confirmed:
            return []
        self.confirmed.add((matched, rank, ours))
        return [self.utterance(confirm_phrase(flag, t - ours), rank)]

    # -- helpers

    def utterance(self, text: str, rank: int, interrupt: bool = False, key: str | None = None) -> Utterance:
        self.seq += 1
        return Utterance(text=text, rank=rank, interrupt=interrupt, key=key, seq=self.seq)

    def raise_scope(self, scope: int | str, t: float, rank: int) -> None:
        if self.level.get(scope, 0) == 0:
            self.episode[scope] = []            # a new episode: the old raises must not confirm this one
        if rank > self.level.get(scope, 0):
            self.level[scope] = rank
            self.episode[scope].append((t, rank))

    def cause(self, rec: dict, msector: int) -> tuple[list[str], str | None]:
        """Cars and detection type behind a rec: its source detections, else what
        this sector was last flagged for (a global escalation carries no sources)."""
        dets = [self.detections[i] for i in rec.get("source_detections", []) if i in self.detections]
        if dets:
            cars = sorted({str(d) for det in dets for d in det.get("drivers", [])}, key=sort_car)
            dtype = str(dets[-1].get("type")) if dets[-1].get("type") in TYPE_WORDS else None
            self.sector_cause[msector] = (cars, dtype)
            return cars, dtype
        return self.sector_cause.get(msector, ([], None))

    def our_first(self, rank: int, scope: int | str, t_official: float) -> tuple[float, int | str] | None:
        """When we first raised at least this level for this scope, within the confirm
        window: the track scope for a track-wide message, else a matching sector (the
        same, up to 2 downstream or 1 upstream). Returns (t, scope matched)."""
        n = self.max_sector or 1000
        best = None
        for s, raises in self.episode.items():
            if scope == TRACK:
                ok = s == TRACK
            else:
                ok = s != TRACK and sector_matches(int(s), int(scope), n)
            if not ok:
                continue
            times = [rt for rt, r in raises if r >= rank and rt >= t_official - CONFIRM_WINDOW_S]
            if times and (best is None or min(times) < best[0]):
                best = (min(times), s)
        return best


# ---- speaker: priority queue in front of one non-blocking voice ----------------

class Backend(Protocol):
    def start(self, text: str) -> object: ...
    def running(self, handle: object) -> bool: ...
    def stop(self, handle: object) -> None: ...


class SayBackend:
    """macOS `say`, one subprocess per phrase, never awaited."""

    def __init__(self, voice: str = VOICE) -> None:
        self.voice = voice

    def start(self, text: str) -> object:
        try:
            return subprocess.Popen(["say", "-v", self.voice, text], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            log(f"voice: say failed ({exc}), printing instead")
            print(text, flush=True)
            return None

    def running(self, handle: object) -> bool:
        return isinstance(handle, subprocess.Popen) and handle.poll() is None

    def stop(self, handle: object) -> None:
        if isinstance(handle, subprocess.Popen) and handle.poll() is None:
            handle.terminate()


class PrintBackend:
    """Non-Mac fallback: print the line, nothing runs."""

    def start(self, text: str) -> object:
        print(text, flush=True)
        return None

    def running(self, handle: object) -> bool:
        return False

    def stop(self, handle: object) -> None:
        return None


class Speaker:
    """Queued utterances and, on every step(now), what to start or cut.

    A phrase with interrupt set and a higher rank than the one being spoken (or
    the last one started) starts at once and cuts the current phrase. Everything
    else waits until nothing is speaking and min_gap_s has passed since the last
    start. Queued phrases older than max_wait_s are dropped as stale.
    """

    def __init__(self, backend: Backend, min_gap_s: float = MIN_GAP_S, max_wait_s: float = MAX_WAIT_S) -> None:
        self.backend = backend
        self.min_gap_s, self.max_wait_s = min_gap_s, max_wait_s
        self.heap: list[tuple[Utterance, float]] = []      # (utterance, wall time queued)
        self.current: Utterance | None = None
        self.handle: object = None
        self.last_start: float | None = None
        self.last_rank = -1
        self.spoken: list[str] = []

    def push(self, utt: Utterance, now: float) -> None:
        if utt.key is not None:       # a higher level for the same scope makes queued lower ones stale
            self.heap = [(u, q) for u, q in self.heap if not (u.key == utt.key and u.rank < utt.rank)]
            heapq.heapify(self.heap)
        heapq.heappush(self.heap, (utt, now))

    def step(self, now: float) -> list[str]:
        """Start the most urgent phrase when allowed. Returns the texts started."""
        if self.current is not None and not self.backend.running(self.handle):
            self.current, self.handle = None, None
        while self.heap and now - self.heap[0][1] > self.max_wait_s:
            heapq.heappop(self.heap)
        if not self.heap:
            return []
        top = self.heap[0][0]
        speaking = self.current is not None
        higher = top.interrupt and top.rank > (self.current.rank if speaking else self.last_rank)
        if speaking and not higher:
            return []
        if not higher and self.last_start is not None and now - self.last_start < self.min_gap_s:
            return []
        if speaking:
            self.backend.stop(self.handle)
        heapq.heappop(self.heap)
        self.current, self.last_start, self.last_rank = top, now, top.rank
        self.handle = self.backend.start(top.text)
        if not self.backend.running(self.handle):
            self.current, self.handle = None, None
        self.spoken.append(top.text)
        return [top.text]


def make_backend(force_print: bool, voice: str) -> Backend:
    if force_print or sys.platform != "darwin" or shutil.which("say") is None:
        return PrintBackend()
    return SayBackend(voice)


# ---- stream client -------------------------------------------------------------

async def run_speaker(speaker: Speaker) -> None:
    while True:
        try:
            for text in speaker.step(time.monotonic()):
                log(f"VOICE {text}")
        except Exception as exc:           # noqa: BLE001  never stop speaking because of one bad phrase
            log(f"voice: speaker error {exc!r}")
        await asyncio.sleep(STEP_S)


async def run_stream(url: str, narrator: Narrator, speaker: Speaker) -> None:
    import websockets

    backoff = BACKOFF_START_S
    while True:
        try:
            async with websockets.connect(url, max_size=None) as ws:
                log(f"voice: connected to {url}")
                backoff = BACKOFF_START_S
                async for raw in ws:
                    for utt in narrator.on_raw(raw):
                        speaker.push(utt, time.monotonic())
                log("voice: server closed the connection")
        except Exception as exc:           # noqa: BLE001
            log(f"voice: connection error {exc!r}")
        log(f"voice: reconnecting in {backoff:.0f} s")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, BACKOFF_CAP_S)


async def run(url: str, speaker: Speaker) -> None:
    await asyncio.gather(run_stream(url, Narrator(), speaker), run_speaker(speaker))


def main() -> None:
    p = argparse.ArgumentParser(description="Speak Fast Flag recommendations like race control radio")
    p.add_argument("--url", default=URL)
    p.add_argument("--voice", default=VOICE, help="macOS voice, see: say -v '?'")
    p.add_argument("--print", action="store_true", help="print the lines instead of speaking")
    a = p.parse_args()
    backend = make_backend(a.print, a.voice)
    log(f"voice: {'printing' if isinstance(backend, PrintBackend) else 'speaking with ' + a.voice}")
    try:
        asyncio.run(run(a.url, Speaker(backend)))
    except KeyboardInterrupt:
        log("voice: shutting down")


if __name__ == "__main__":
    main()

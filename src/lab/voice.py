"""Race control voice: speaks Fast Flag recommendations like race control radio.

Subscribes to the stream and speaks only when a sector's flag level goes up
(YELLOW, DOUBLE_YELLOW) or the track-wide level goes up (VSC, SC, RED), plus a
short "Sector 7 clear" when a level returns to CLEAR. When the official message
for a sector (or the track) we already escalated arrives later, it says
"Race control confirms Yellow flag, X.X seconds after Fast Flag", measured from
our first raise of that scope. When race control was first, nothing is said
about lead time, not even when race control re-issues the flag later. Every
phrase comes from a template: flag, car numbers, detection type and sector.
Nothing here decides anything, it only reads out the rec envelopes it receives.

It connects with ?catchup=1 and never speaks the past: what arrives before the first tick
after a (re)connect, and what the server replays from before the tick of a seek (the
catch-up: every flag already out), only sets the levels, so after a seek the voice knows
which flags are out and speaks only new calls. A rec re-sent with "recovery taking long"
(the race control engine's long-recovery advisory) keeps its flag and is never a new call.

Speech uses macOS `say` in a non-blocking subprocess. A higher flag level
interrupts the current phrase, everything else waits for the current phrase and
for a 2 s gap. On a non-Mac (or with --print, an extra option for checking the
output) the lines are printed instead, without pacing.

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

URL = "ws://localhost:8000/stream?catchup=1"
ADVISORY = "recovery taking long"   # src/racecontrol/engine.py: a flag re-sent as an advisory, not a new call
VOICE = "Daniel"
RESET_JUMP_S = 2.0            # same rule as the race control engine and the dashboard
MIN_GAP_S = 2.0               # at most one non-interrupting call every 2 s
MAX_WAIT_S = 12.0             # a confirmation or clear still queued after this long (wall time) is stale
CONFIRM_WINDOW_S = 120.0      # a cleared episode can still be confirmed this long after our first raise
MIN_LEAD_S = 0.05             # below the spoken resolution: not a lead
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
STEP_S = 0.1
BACKOFF_START_S, BACKOFF_CAP_S = 1.0, 5.0    # local server: same cap as the race control client

RANK = {"CLEAR": 0, "YELLOW": 1, "DOUBLE_YELLOW": 2, "VSC": 3, "SC": 4, "RED": 5}
GLOBAL_FLAGS = {"VSC", "SC", "RED"}
FLAG_WORDS = {"YELLOW": "Yellow flag", "DOUBLE_YELLOW": "Double yellow", "VSC": "Virtual Safety Car",
              "SC": "Safety Car", "RED": "Red flag"}
TYPE_WORDS = {"IMPACT": "impact", "STOPPED": "stopped", "SPIN": "spin", "DROPOUT": "no data",
              "MULTI": "multi-car", "ANOMALY": "anomaly"}
TRACK = "track"               # scope key for track-wide flags (VSC, SC, RED)

Scope = int | str


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


def with_catchup(url: str) -> str:
    """Ask the server for the flags already out (hub catch-up), unless the url already says."""
    return url if "catchup=" in url else url + ("&" if "?" in url else "?") + "catchup=1"


def clean_car(drv: object) -> str:
    """Car numbers only ever reach `say` as plain alphanumerics (it honours [[...]] commands)."""
    return "".join(ch for ch in str(drv) if ch.isalnum())[:8]


def sort_car(drv: str) -> tuple[int, str]:
    return (int(drv), drv) if drv.isdigit() else (10**6, drv)


# ---- narrator: envelopes in, phrases out ----------------------------------------

@dataclass(order=True)
class Utterance:
    """What to say and how urgently. Heap order: level-ups first, then highest rank, then oldest."""
    sort_key: tuple[int, int, int] = field(init=False, repr=False)
    text: str = field(compare=False)
    rank: int = field(compare=False, default=0)
    interrupt: bool = field(compare=False, default=False)   # a level-up: a higher one cuts current speech
    keys: frozenset[str] = field(compare=False, default=frozenset())   # scopes it covers, for dropping stale ones
    seq: int = field(compare=False, default=0)

    def __post_init__(self) -> None:
        self.sort_key = (0 if self.interrupt else 1, -self.rank, self.seq)


class Narrator:
    """Turns the stream into utterances. Pure logic, no I/O, so it is testable.

    Levels are tracked per sector (YELLOW, DOUBLE_YELLOW) and for the track
    (VSC, SC, RED), each in its own rank space. An episode is one run of a scope
    from CLEAR up and back: its raises are kept so a later official message can
    be matched against when we first raised that scope. A track-wide raise is
    also noted in its cause sector's episode (the sector was escalated by us),
    without changing that sector's own level.
    """

    def __init__(self) -> None:
        self.seq = 0
        self.max_sector = 0            # kept across resets: the track does not change on a backward jump
        self.reset()

    def reset(self) -> None:
        self.t: float | None = None
        self.quiet_before: float | None = None    # catch-up: envelopes stamped before this tick are the past
        self.detections: dict[str, dict] = {}
        self.level: dict[Scope, int] = {}                          # scope -> current rank
        self.open: set[Scope] = set()                              # scopes with an open episode
        self.episode: dict[Scope, list[tuple[float, int]]] = {}    # scope -> [(t, rank)] raises of the episode
        self.official: dict[Scope, list[tuple[float, int]]] = {}   # scope -> [(t, rank)] official messages seen
        self.sector_cause: dict[int, tuple[list[str], str | None]] = {}
        self.confirmed: set[tuple[Scope, float]] = set()

    # -- entry points

    def on_raw(self, raw: str | bytes) -> list[Utterance]:
        if len(raw) > MAX_MESSAGE_BYTES:
            log(f"voice: skipped a {len(raw)} byte message")
            return []
        try:
            env = json.loads(raw)
        except Exception as exc:       # noqa: BLE001  a bad message never kills the voice
            log(f"voice: skipped a message that is not JSON ({type(exc).__name__})")
            return []
        return self.on_envelope(env)

    def on_envelope(self, env: object) -> list[Utterance]:
        if not isinstance(env, dict) or not isinstance(env.get("data"), dict):
            return []
        try:
            kind = str(env.get("kind"))
            handler = {"tick": self.on_tick, "detection": self.on_detection, "rec": self.on_rec,
                       "official": self.on_official}.get(kind)
            if handler is None:
                return []
            past = kind != "tick" and self.is_past(env["data"])
            utts = handler(env["data"])
            return [] if past else utts          # the past only sets the levels
        except Exception as exc:       # noqa: BLE001  a malformed envelope never kills the voice
            log(f"voice: skipped {env.get('kind')} envelope ({exc!r})")
            return []

    def is_past(self, data: dict) -> bool:
        """Catch-up, not news: before the first tick after a (re)connect, or stamped before the
        tick of a seek while the server replays what came before it (until the next tick)."""
        if self.t is None:
            return True
        return self.quiet_before is not None and float(data.get("t", self.quiet_before)) < self.quiet_before

    def on_tick(self, tick: dict) -> list[Utterance]:
        t = float(tick["t"])
        if self.t is not None and abs(t - self.t) > RESET_JUMP_S:     # a seek or loop, back or forward
            self.reset()
            self.quiet_before = t               # the server's catch-up of this seek follows this tick
        elif self.t is None:
            self.quiet_before = t               # first tick after a (re)connect: same
        else:
            self.quiet_before = None            # the next tick: live again
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
        self.note_cause(rec, msector)             # every rec refreshes what the sector is flagged for
        rank = RANK[flag]
        scope: Scope = TRACK if flag in GLOBAL_FLAGS else msector
        if rank <= self.level.get(scope, 0):
            return []
        advisory = ADVISORY in str(rec.get("reason", ""))
        self.raise_scope(scope, t, rank)
        if advisory:
            return []                             # the flag is out (we missed its call), not a new call
        keys = {str(scope)}
        if scope == TRACK:
            self.note_raise(msector, t, rank)     # the cause sector is escalated by us too
            keys.add(str(msector))
        cars, dtype = self.sector_cause.get(msector, ([], None))
        return [self.utterance(flag_phrase(flag, cars, dtype, msector), rank, interrupt=True, keys=keys)]

    def on_clear(self, rec: dict, msector: int) -> list[Utterance]:
        msg = str(rec.get("message", "")).upper()
        sector_clear = msg.strip() != "TRACK CLEAR"      # contract: only TRACK CLEAR ends VSC, SC and RED
        scope: Scope = msector if sector_clear else TRACK
        if self.level.get(scope, 0) == 0:
            return []
        self.level[scope] = 0
        self.open.discard(scope)
        if sector_clear:
            self.sector_cause.pop(msector, None)
        return [self.utterance(clear_phrase(msector if sector_clear else None), 0, keys={str(scope)})]

    def on_official(self, off: dict) -> list[Utterance]:
        flag = str(off["flag"])
        if flag not in RANK:
            return []
        rank, t = RANK[flag], float(off["t"])
        msector = off.get("msector")
        scope: Scope = TRACK if msector is None else int(msector)
        ours = self.our_first(scope, t)
        rc_earlier = ours is not None and self.official_before(scope, ours)
        self.official.setdefault(scope, []).append((t, rank))
        if rank == 0 or ours is None or rc_earlier:
            return []
        if scope == TRACK and self.level_reached(TRACK, ours, t) < rank:
            return []                   # we never called this track-wide level (a VSC does not confirm a red flag)
        lead = t - ours
        once = (TRACK if scope == TRACK else "sector", ours)     # one confirmation per episode of ours
        if lead < MIN_LEAD_S or once in self.confirmed:          # a tie is not a lead
            return []
        self.confirmed.add(once)
        return [self.utterance(confirm_phrase(flag, lead), rank)]

    # -- helpers

    def utterance(self, text: str, rank: int, interrupt: bool = False, keys: set[str] | None = None) -> Utterance:
        self.seq += 1
        return Utterance(text=text, rank=rank, interrupt=interrupt, keys=frozenset(keys or ()), seq=self.seq)

    def raise_scope(self, scope: Scope, t: float, rank: int) -> None:
        self.note_raise(scope, t, rank)
        self.level[scope] = rank

    def note_raise(self, scope: Scope, t: float, rank: int) -> None:
        """Record a raise in a scope's episode, opening a new episode after a clear."""
        if scope not in self.open:
            self.open.add(scope)
            self.episode[scope] = []
        self.episode[scope].append((t, rank))

    def note_cause(self, rec: dict, msector: int) -> None:
        """Cars and detection type behind a rec: its source detections (the type of the
        last physical one, ANOMALY only when nothing else corroborates). A rec without
        sources (a global escalation from a sustained stop) keeps the sector's last cause."""
        dets = [self.detections[i] for i in rec.get("source_detections", []) if i in self.detections]
        if not dets:
            return
        cars = sorted({clean_car(d) for det in dets for d in det.get("drivers", []) if clean_car(d)}, key=sort_car)
        types = [str(det.get("type")) for det in dets if det.get("type") in TYPE_WORDS]
        physical = [x for x in types if x != "ANOMALY"]
        dtype = (physical or types or [None])[-1]
        self.sector_cause[msector] = (cars, dtype)

    def matching(self, scope: Scope) -> list[Scope]:
        """Our scopes that an official message for this scope refers to: the track for a
        track-wide message, else the same sector, 2 downstream or 1 upstream (sector_matches)."""
        if scope == TRACK:
            return [TRACK]
        n = self.max_sector or 1000
        return [s for s in self.episode if s != TRACK and sector_matches(int(s), int(scope), n)]

    def our_first(self, scope: Scope, t_official: float) -> float | None:
        """When we first raised a matching scope: the start of an open episode, or of a
        cleared one that started within the confirm window before the official message."""
        firsts = [self.episode[s][0][0] for s in self.matching(scope) if self.episode.get(s)
                  and (s in self.open or self.episode[s][0][0] >= t_official - CONFIRM_WINDOW_S)]
        return min(firsts) if firsts else None

    def level_reached(self, scope: Scope, since: float, until: float) -> int:
        """Highest level we raised for this scope between our first raise and the official message."""
        return max((r for rt, r in self.episode.get(scope, []) if since <= rt <= until), default=0)

    def official_before(self, scope: Scope, t: float) -> bool:
        """Did race control flag this scope (or the whole track) before t? Then race control
        was first and nothing about lead time is ever said for this episode."""
        covering = [TRACK] if scope == TRACK else [TRACK] + [s for s in self.official
                                                              if s != TRACK and sector_matches(int(scope), int(s), self.max_sector or 1000)]
        return any(t - CONFIRM_WINDOW_S <= ot < t and r > 0 for s in covering for ot, r in self.official.get(s, []))


# ---- speaker: priority queue in front of one non-blocking voice ----------------

class Backend(Protocol):
    instant: bool          # True when nothing runs (print): no pacing, no stale drop

    def start(self, text: str) -> object: ...
    def running(self, handle: object) -> bool: ...
    def stop(self, handle: object) -> None: ...
    def error(self, handle: object) -> str | None: ...


class SayBackend:
    """macOS `say`, one subprocess per phrase, never awaited."""

    instant = False

    def __init__(self, voice: str = VOICE) -> None:
        self.voice = voice

    def start(self, text: str) -> object:
        try:
            return subprocess.Popen(["say", "-v", self.voice, "--", text], stdin=subprocess.DEVNULL,
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

    def error(self, handle: object) -> str | None:
        if isinstance(handle, subprocess.Popen) and handle.poll() not in (None, 0, -15):   # -15: we cut it
            return f"say exited with {handle.returncode}"
        return None


class PrintBackend:
    """Non-Mac fallback: print the line, nothing runs, nothing waits."""

    instant = True

    def start(self, text: str) -> object:
        print(text, flush=True)
        return None

    def running(self, handle: object) -> bool:
        return False

    def stop(self, handle: object) -> None:
        return None

    def error(self, handle: object) -> str | None:
        return None


class Speaker:
    """Queued utterances and, on every step(now), what to start or cut.

    A level-up phrase with a higher rank than the one being spoken (or the last
    one started, when idle) starts at once and cuts the current phrase. Everything
    else waits until nothing is speaking and min_gap_s has passed since the last
    start. Level-ups are always spoken; confirmations and clears queued longer than
    max_wait_s are dropped as stale. A level-up for a scope drops queued lower
    phrases for that scope. With an instant backend (print) nothing waits.
    """

    def __init__(self, backend: Backend, min_gap_s: float = MIN_GAP_S, max_wait_s: float = MAX_WAIT_S) -> None:
        self.backend = backend
        self.instant = bool(getattr(backend, "instant", False))
        self.min_gap_s, self.max_wait_s = min_gap_s, max_wait_s
        self.heap: list[tuple[Utterance, float]] = []      # (utterance, wall time queued)
        self.current: Utterance | None = None
        self.handle: object = None
        self.last_start: float | None = None
        self.last_rank = -1
        self.spoken: list[str] = []

    def push(self, utt: Utterance, now: float) -> None:
        if utt.keys:      # a higher level for a scope makes queued lower phrases for it stale
            self.heap = [(u, q) for u, q in self.heap if not (u.keys & utt.keys and u.rank < utt.rank)]
            heapq.heapify(self.heap)
        heapq.heappush(self.heap, (utt, now))

    def step(self, now: float) -> list[str]:
        """Start the most urgent phrase when allowed. Returns the texts started."""
        self.reap()
        if not self.instant:
            self.drop_stale(now)
        if not self.heap:
            return []
        top = self.heap[0][0]
        speaking = self.current is not None
        higher = top.interrupt and top.rank > (self.current.rank if speaking else self.last_rank)
        if speaking and not higher:
            return []
        paced = not self.instant and not higher
        if paced and self.last_start is not None and now - self.last_start < self.min_gap_s:
            return []
        if speaking:
            self.backend.stop(self.handle)
        heapq.heappop(self.heap)
        self.current, self.last_start, self.last_rank = top, now, top.rank
        self.handle = self.backend.start(top.text)
        self.reap()
        self.spoken.append(top.text)
        return [top.text]

    def reap(self) -> None:
        if self.current is not None and not self.backend.running(self.handle):
            err = self.backend.error(self.handle)
            if err:
                log(f"voice: {err}, printing instead")
                print(self.current.text, flush=True)
            self.current, self.handle = None, None

    def drop_stale(self, now: float) -> None:
        fresh = [(u, q) for u, q in self.heap if u.interrupt or now - q <= self.max_wait_s]
        if len(fresh) != len(self.heap):
            self.heap = fresh
            heapq.heapify(self.heap)

    def shutdown(self) -> None:
        if self.current is not None:
            self.backend.stop(self.handle)


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
            async with websockets.connect(url, max_size=MAX_MESSAGE_BYTES) as ws:
                log(f"voice: connected to {url}")
                narrator.reset()                  # the catch-up replays everything since the last seek: never twice
                backoff = BACKOFF_START_S
                async for raw in ws:
                    try:
                        for utt in narrator.on_raw(raw):
                            speaker.push(utt, time.monotonic())
                    except Exception as exc:   # noqa: BLE001  one bad message never drops the socket
                        log(f"voice: skipped a message ({exc!r})")
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
    speaker = Speaker(backend)
    log(f"voice: {'printing' if speaker.instant else 'speaking with ' + a.voice}")
    try:
        asyncio.run(run(with_catchup(a.url), speaker))
    except KeyboardInterrupt:
        log("voice: shutting down")
    finally:
        speaker.shutdown()


if __name__ == "__main__":
    main()

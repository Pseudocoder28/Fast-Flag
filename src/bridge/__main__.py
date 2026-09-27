"""Serial bridge: relays rec envelopes from the stream to the Uno marshal panel.

Run:  python -m src.bridge                  (list serial ports and exit)
      python -m src.bridge COM3
      python -m src.bridge COM3 --url ws://localhost:8001/stream
      python -m src.bridge loop://          (no hardware: pyserial loopback, for testing)

Serial protocol (PROJECT_BRIEF.md 7.7), 115200 baud, ASCII, newline terminated:
  G,<GREEN|VSC|SC|RED>                        track-wide state
  Z,<zone 1-3>,<CLEAR|YELLOW|DOUBLE_YELLOW>   panel zone: highest flag across its sectors
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import httpx
import serial
import serial.tools.list_ports
import websockets

from src.racecontrol.engine import RESET_JUMP_S, SECTOR_RANK

URL = "ws://localhost:8000/stream"
BAUD = 115200
BACKOFF_START_S = 1.0
BACKOFF_CAP_S = 5.0
READY_TIMEOUT_S = 3.0     # the Uno resets when the port opens (DTR) and prints READY after setup()
READY_FALLBACK_S = 2.0
WRITE_TIMEOUT_S = 1.0
DEFAULT_SECTORS = 20
ZONES = (1, 2, 3)
GLOBAL_FLAGS = ("VSC", "SC", "RED")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fmt_t(t: object) -> str:
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return f"{t:.2f}"
    return "-"


def track_url_for(ws_url: str) -> str:
    u = urlsplit(ws_url)
    scheme = "https" if u.scheme == "wss" else "http"
    return urlunsplit((scheme, u.netloc, "/track", "", ""))


class Panel:
    """What the Uno is showing, and the per-sector flags behind each zone."""

    def __init__(self, ser: serial.SerialBase) -> None:
        self.ser = ser
        self.n_sectors = DEFAULT_SECTORS
        self.sector_flags: dict[int, str] = {}  # only sectors that are not CLEAR
        self.zone_sent: dict[int, str] = {}
        self.global_sent: str | None = None
        self.last_tick_t: float | None = None

    def send(self, cmd: str, t: object, flag: str) -> bool:
        try:
            self.ser.write(f"{cmd}\n".encode("ascii"))
        except serial.SerialException as exc:
            log(f"serial write failed ({cmd}): {exc}")
            return False
        log(f"t={fmt_t(t)} flag={flag} cmd={cmd}")
        return True

    def zone_for_sector(self, msector: int) -> int | None:
        if msector < 1:
            return None
        return min(3, math.ceil(msector * 3 / self.n_sectors))

    def zone_flag(self, zone: int) -> str:
        flags = [f for s, f in self.sector_flags.items() if self.zone_for_sector(s) == zone]
        return max(flags, key=SECTOR_RANK.__getitem__, default="CLEAR")

    def sync_zones(self, t: object) -> None:
        # a failed write leaves zone_sent unchanged, so the next sync retries it
        for zone in ZONES:
            flag = self.zone_flag(zone)
            if flag != self.zone_sent.get(zone) and self.send(f"Z,{zone},{flag}", t, flag):
                self.zone_sent[zone] = flag

    def set_global(self, flag: str, t: object) -> None:
        if flag != self.global_sent and self.send(f"G,{flag}", t, flag):
            self.global_sent = flag

    def full_reset(self, t: object) -> None:
        """Known state: G,GREEN and every zone CLEAR, sent unconditionally."""
        self.sector_flags.clear()
        self.global_sent = None
        self.zone_sent.clear()
        self.set_global("GREEN", t)
        self.sync_zones(t)

    def set_sector_count(self, n: int) -> None:
        if n != self.n_sectors:
            self.n_sectors = n
            self.sync_zones(self.last_tick_t)  # same sector flags, new zone split

    def on_rec(self, rec: dict) -> None:
        flag = rec.get("flag")
        t = rec.get("t")
        if flag in GLOBAL_FLAGS:
            self.set_global(flag, t)
            return
        if flag == "CLEAR" and "TRACK CLEAR" in str(rec.get("message", "")):
            self.set_global("GREEN", t)
            return
        msector = rec.get("msector")
        valid = (flag in SECTOR_RANK and isinstance(msector, int) and not isinstance(msector, bool)
                 and self.zone_for_sector(msector) is not None)
        if not valid:
            log(f"skipping rec {rec.get('id')}: flag={flag} msector={msector}")
            return
        if flag == "CLEAR":
            self.sector_flags.pop(msector, None)
        else:
            self.sector_flags[msector] = flag
        self.sync_zones(t)

    def on_tick_t(self, t: float) -> None:
        # the engine emits no CLEAR recs on reset, so the panel resets itself here
        if self.last_tick_t is not None and t < self.last_tick_t - RESET_JUMP_S:
            log(f"RESET at t={t:.2f}")
            self.full_reset(t)
        self.last_tick_t = t


def handle_message(panel: Panel, raw: str | bytes) -> None:
    try:
        env = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(env, dict):
        return
    data = env.get("data")
    if not isinstance(data, dict):
        return
    kind = env.get("kind")
    if kind == "rec":
        panel.on_rec(data)
    elif kind == "tick":
        t = data.get("t")  # only t is used, for reset detection
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            panel.on_tick_t(float(t))


async def fetch_sector_count(ws_url: str) -> int:
    track_url = track_url_for(ws_url)
    try:
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            r = await client.get(track_url)
            r.raise_for_status()
            n = len(r.json()["msectors"])
    except Exception as exc:
        log(f"GET {track_url} failed ({exc!r}), assuming {DEFAULT_SECTORS} sectors")
        return DEFAULT_SECTORS
    if n < 1:
        log(f"{track_url} has no msectors, assuming {DEFAULT_SECTORS} sectors")
        return DEFAULT_SECTORS
    log(f"track has {n} marshal sectors")
    return n


async def run_client(url: str, panel: Panel) -> None:
    backoff = BACKOFF_START_S
    while True:
        try:
            async with websockets.connect(url) as ws:
                log(f"connected to {url}")
                backoff = BACKOFF_START_S
                panel.set_sector_count(await fetch_sector_count(url))
                async for raw in ws:
                    handle_message(panel, raw)
                log("server closed the connection")
        except Exception as exc:
            log(f"connection error: {exc!r}")
        log(f"reconnecting in {backoff:.0f}s")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, BACKOFF_CAP_S)


def wait_for_ready(ser: serial.SerialBase) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while (left := deadline - time.monotonic()) > 0:
        ser.timeout = left
        try:
            line = ser.readline()
        except serial.SerialException as exc:
            log(f"serial read failed while waiting for READY: {exc}")
            break
        text = line.decode("ascii", errors="replace").strip()
        if "READY" in text:
            log("Uno READY")
            return
        if text:
            log(f"uno: {text}")
    log(f"no READY within {READY_TIMEOUT_S:.0f} s, waiting {READY_FALLBACK_S:.0f} s more")
    time.sleep(READY_FALLBACK_S)


def list_ports() -> None:
    ports = sorted(serial.tools.list_ports.comports(), key=lambda p: p.device)
    if not ports:
        print("no serial ports found")
    for p in ports:
        print(f"{p.device}  {p.description}")
    print("usage: python -m src.bridge <port>, e.g. python -m src.bridge COM3")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("port", nargs="?", help="serial port, e.g. COM3 (omit to list ports)")
    p.add_argument("--url", default=URL, help=f"stream to join (default {URL})")
    a = p.parse_args()
    if a.port is None:
        list_ports()
        return

    try:
        # serial_for_url opens plain port names exactly like serial.Serial, and also accepts loop://
        ser = serial.serial_for_url(a.port, baudrate=BAUD, timeout=0.1, write_timeout=WRITE_TIMEOUT_S)
    except (serial.SerialException, ValueError) as exc:
        log(f"cannot open {a.port}: {exc}")
        sys.exit(1)
    log(f"opened {a.port} at {BAUD} baud")

    panel = Panel(ser)
    try:
        wait_for_ready(ser)
        panel.full_reset(None)
        asyncio.run(run_client(a.url, panel))
    except KeyboardInterrupt:
        log("shutting down (Ctrl+C)")
    finally:
        ser.close()


if __name__ == "__main__":
    main()

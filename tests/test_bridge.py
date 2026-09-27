"""Tests for the serial bridge (src/bridge/__main__.py), using a fake serial port. Run: python -m pytest"""

from __future__ import annotations

import json

import pytest
import serial

from src.bridge.__main__ import MSG_MAX, Panel, handle_message


class FakeSerial:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.fail = False

    def write(self, data: bytes) -> int:
        if self.fail:
            raise serial.SerialException("unplugged")
        self.lines.append(data.decode("ascii").rstrip("\n"))
        return len(data)

    def take(self) -> list[str]:
        out, self.lines = self.lines, []
        return out


@pytest.fixture
def panel() -> tuple[Panel, FakeSerial]:
    ser = FakeSerial()
    p = Panel(ser)
    p.full_reset(None)
    ser.take()
    return p, ser


def rec(p: Panel, msector: int, flag: str, message: str = "", t: float = 100.0) -> None:
    data = {"id": "rec-x", "t": t, "msector": msector, "flag": flag, "confidence": 0.8,
            "reason": "test", "message": message, "source_detections": []}
    handle_message(p, json.dumps({"kind": "rec", "data": data}))


def tick(p: Panel, t: float) -> None:
    handle_message(p, json.dumps({"kind": "tick", "data": {"t": t, "lap": 1, "track_status": "1", "cars": []}}))


RESET_LINES = ["G,GREEN", "Z,1,CLEAR", "Z,2,CLEAR", "Z,3,CLEAR", "M,TRACK GREEN"]


def test_startup_reset_sends_known_state() -> None:
    ser = FakeSerial()
    Panel(ser).full_reset(None)
    assert ser.take() == RESET_LINES


def test_zone_keeps_highest_flag_across_its_sectors(panel) -> None:
    p, ser = panel
    rec(p, 3, "YELLOW")
    assert ser.take() == ["Z,1,YELLOW", "M,YELLOW S3"]
    rec(p, 5, "CLEAR")  # sector 5 clearing must not clear zone 1 while sector 3 is YELLOW
    assert ser.take() == []
    assert p.zone_sent[1] == "YELLOW"


def test_zone_drops_back_to_next_highest(panel) -> None:
    p, ser = panel
    rec(p, 3, "YELLOW")
    rec(p, 5, "DOUBLE_YELLOW")
    ser.take()
    rec(p, 5, "CLEAR")
    assert ser.take() == ["Z,1,YELLOW", "M,CLEAR S5"]
    rec(p, 3, "CLEAR")
    assert ser.take() == ["Z,1,CLEAR", "M,CLEAR S3"]


def test_dedup_sector_and_global(panel) -> None:
    p, ser = panel
    rec(p, 9, "YELLOW")
    ser.take()
    rec(p, 9, "YELLOW")  # e.g. an ANOMALY corroboration rec, same flag
    assert ser.take() == []
    rec(p, 9, "SC", "SAFETY CAR DEPLOYED")
    assert ser.take() == ["G,SC", "M,SAFETY CAR"]
    rec(p, 9, "SC", "SAFETY CAR DEPLOYED")
    assert ser.take() == []


def test_reset_on_backward_tick_only(panel) -> None:
    p, ser = panel
    tick(p, 100.0)
    rec(p, 9, "DOUBLE_YELLOW")
    rec(p, 9, "SC", "SAFETY CAR DEPLOYED")
    ser.take()

    tick(p, 101.0)
    tick(p, 99.5)  # 1.5 s wobble, not a reset
    assert ser.take() == []

    tick(p, 50.0)
    assert ser.take() == RESET_LINES
    assert p.sector_flags == {}


def test_track_clear_sends_green_but_sector_clear_does_not(panel) -> None:
    p, ser = panel
    rec(p, 9, "YELLOW")
    rec(p, 9, "SC", "SAFETY CAR DEPLOYED")
    ser.take()

    rec(p, 9, "CLEAR", "CLEAR IN TRACK SECTOR 9")
    assert ser.take() == ["Z,2,CLEAR", "M,CLEAR S9"]

    rec(p, 9, "CLEAR", "TRACK CLEAR")
    assert ser.take() == ["G,GREEN", "M,TRACK GREEN"]


@pytest.mark.parametrize("msector,flag,message,expected", [
    (9, "YELLOW", "YELLOW IN TRACK SECTOR 9", ["Z,2,YELLOW", "M,YELLOW S9"]),
    (9, "DOUBLE_YELLOW", "DOUBLE YELLOW IN TRACK SECTOR 9", ["Z,2,DOUBLE_YELLOW", "M,DBL YELLOW S9"]),
    (9, "SC", "SAFETY CAR DEPLOYED", ["G,SC", "M,SAFETY CAR"]),
    (9, "VSC", "VIRTUAL SAFETY CAR DEPLOYED", ["G,VSC", "M,VIRTUAL SC"]),
    (9, "RED", "RED FLAG", ["G,RED", "M,RED FLAG"]),
])
def test_m_line_text(panel, msector, flag, message, expected) -> None:
    p, ser = panel
    rec(p, msector, flag, message)
    assert ser.take() == expected


def test_m_line_truncated_to_lcd_width(panel) -> None:
    p, ser = panel
    p.message("X" * 40, 1.0, "YELLOW")
    (line,) = ser.take()
    assert line == "M," + "X" * MSG_MAX


def test_every_m_line_fits(panel) -> None:
    p, ser = panel
    for s in (1, 9, 20, 99):
        for flag in ("YELLOW", "DOUBLE_YELLOW", "CLEAR"):
            rec(p, s, flag)
    for flag in ("VSC", "SC", "RED"):
        rec(p, 9, flag)
    rec(p, 9, "CLEAR", "TRACK CLEAR")
    ms = [ln for ln in ser.take() if ln.startswith("M,")]
    assert ms and all(len(ln) - 2 <= MSG_MAX for ln in ms)


def test_failed_write_sends_no_m_line_and_retries(panel) -> None:
    p, ser = panel
    ser.fail = True
    rec(p, 3, "DOUBLE_YELLOW")
    assert ser.take() == []
    assert p.zone_sent[1] == "CLEAR"

    ser.fail = False
    rec(p, 4, "YELLOW")  # next change resends zone 1 at its true level
    assert ser.take() == ["Z,1,DOUBLE_YELLOW", "M,YELLOW S4"]

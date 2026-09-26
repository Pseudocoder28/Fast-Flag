"""Validate every fixture line against the contracts in PROJECT_BRIEF.md Section 7.

Plain Python checks only, no extra dependencies. Run: pytest
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIX = Path(__file__).resolve().parent.parent / "fixtures"

NUM = (int, float)
DET_TYPES = {"STOPPED", "IMPACT", "SPIN", "DROPOUT", "MULTI", "ANOMALY"}
FLAGS = {"CLEAR", "YELLOW", "DOUBLE_YELLOW", "VSC", "SC", "RED"}
TRACK_STATUS = {"1", "2", "3", "4", "5", "6", "7"}
KINDS = {"tick", "detection", "risk", "rec", "official"}

CAR = {"drv": str, "x": NUM, "y": NUM, "dist": NUM, "lat_off": NUM, "speed": NUM,
       "throttle": NUM, "brake": bool, "gear": int, "rpm": NUM, "msector": int,
       "gap_ahead_m": NUM, "in_pit": bool}
TICK = {"t": NUM, "lap": int, "track_status": str, "cars": list}
DETECTION = {"id": str, "t": NUM, "drivers": list, "msector": int, "type": str,
             "severity": NUM, "evidence": str}
RISK = {"t": NUM, "drv": str, "risk_10s": NUM, "risk_30s": NUM, "top_features": list}
REC = {"id": str, "t": NUM, "msector": int, "flag": str, "confidence": NUM,
       "reason": str, "message": str, "source_detections": list}
# official: one official race control event (Section 6.7), sent at its real time.
OFFICIAL = {"t": NUM, "category": str, "message": str, "flag": str,
            "scope": str, "msector": (int, type(None)), "drivers": list}
OFFICIAL_FLAGS = {"YELLOW", "DOUBLE_YELLOW", "RED", "SC", "VSC", "CLEAR"}


def check_shape(obj: dict, shape: dict, where: str) -> None:
    assert isinstance(obj, dict), f"{where}: not an object"
    for key, typ in shape.items():
        assert key in obj, f"{where}: missing key {key!r}"
        val = obj[key]
        # bool is a subclass of int: never accept it where a number is expected
        if typ is not bool and isinstance(val, bool):
            raise AssertionError(f"{where}: {key!r} is bool, expected {typ}")
        assert isinstance(val, typ), f"{where}: {key!r}={val!r} is not {typ}"


def check_tick(d: dict, where: str) -> None:
    check_shape(d, TICK, where)
    assert d["track_status"] in TRACK_STATUS, f"{where}: bad track_status"
    assert d["cars"], f"{where}: no cars"
    for i, car in enumerate(d["cars"]):
        check_shape(car, CAR, f"{where} car[{i}]")
        assert car["speed"] >= 0, f"{where} car[{i}]: negative speed"


def check_detection(d: dict, where: str) -> None:
    check_shape(d, DETECTION, where)
    assert d["type"] in DET_TYPES, f"{where}: bad type {d['type']}"
    assert 0.0 <= d["severity"] <= 1.0, f"{where}: severity out of range"
    assert d["drivers"] and all(isinstance(x, str) for x in d["drivers"])


def check_risk(d: dict, where: str) -> None:
    check_shape(d, RISK, where)
    for k in ("risk_10s", "risk_30s"):
        assert 0.0 <= d[k] <= 1.0, f"{where}: {k} out of range"
    assert all(isinstance(x, str) for x in d["top_features"])


def check_rec(d: dict, where: str) -> None:
    check_shape(d, REC, where)
    assert d["flag"] in FLAGS, f"{where}: bad flag {d['flag']}"
    assert 0.0 <= d["confidence"] <= 1.0, f"{where}: confidence out of range"
    assert all(isinstance(x, str) for x in d["source_detections"])


def check_official(d: dict, where: str) -> None:
    check_shape(d, OFFICIAL, where)
    assert d["flag"] in OFFICIAL_FLAGS, f"{where}: bad flag {d['flag']}"
    assert all(isinstance(x, str) for x in d["drivers"])


CHECKS = {
    "ticks_sample.jsonl": check_tick,
    "detections_sample.jsonl": check_detection,
    "risk_sample.jsonl": check_risk,
    "recs_sample.jsonl": check_rec,
    "official_sample.jsonl": check_official,
}


def read_jsonl(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_fixture_lines(name: str) -> None:
    path = FIX / name
    assert path.exists(), f"missing fixture {name}"
    rows = read_jsonl(path)
    assert rows, f"{name} is empty"
    for i, row in enumerate(rows):
        CHECKS[name](row, f"{name}:{i + 1}")


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_fixture_time_ordered(name: str) -> None:
    ts = [r["t"] for r in read_jsonl(FIX / name)]
    assert ts == sorted(ts), f"{name} is not sorted by t"


def test_track_sample() -> None:
    d = json.loads((FIX / "track_sample.json").read_text())
    check_shape(d, {"race": str, "ref_line": list, "msectors": list, "corners": list},
                "track_sample.json")
    assert len(d["ref_line"]) > 10
    for p in d["ref_line"]:
        assert len(p) == 2 and all(isinstance(v, NUM) for v in p)
    for s in d["msectors"]:
        check_shape(s, {"id": int, "start_dist": NUM, "end_dist": NUM}, "msector")
    for c in d["corners"]:
        check_shape(c, {"number": int, "x": NUM, "y": NUM}, "corner")


def test_cross_references() -> None:
    """Recs point at real detections, detection drivers exist in ticks."""
    det_ids = {d["id"] for d in read_jsonl(FIX / "detections_sample.jsonl")}
    for r in read_jsonl(FIX / "recs_sample.jsonl"):
        assert set(r["source_detections"]) <= det_ids, f"rec {r['id']} unknown detection"
    drivers = {c["drv"] for t in read_jsonl(FIX / "ticks_sample.jsonl") for c in t["cars"]}
    for d in read_jsonl(FIX / "detections_sample.jsonl"):
        assert set(d["drivers"]) <= drivers, f"detection {d['id']} unknown driver"


def test_envelope_kinds() -> None:
    assert KINDS == {"tick", "detection", "risk", "rec", "official"}


def test_fixture_size() -> None:
    total = sum(p.stat().st_size for p in FIX.iterdir() if p.is_file())
    assert total < 20 * 1024 * 1024, f"fixtures are {total / 1e6:.1f} MB, limit 20 MB"

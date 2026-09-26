"""Tests for the race control engine (src/racecontrol/engine.py). Run: python -m pytest"""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_contracts import check_rec

from src.racecontrol.engine import (
    ANOMALY_MAX_CONF,
    ANOMALY_MIN_SEV,
    GLOBAL_CLEAR_AFTER_S,
    GLOBAL_RANK,
    MIN_HOLD_S,
    RESET_JUMP_S,
    SC_STOPPED_HOLD_S,
    SECTOR_CLEAR_AFTER_S,
    SECTOR_RANK,
    STALE_CAR_S,
    VSC_STOPPED_HOLD_S,
    RaceControl,
)

FIX = Path(__file__).resolve().parent.parent / "fixtures"


def read_jsonl(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def make_car(drv: str, msector: int, speed: float = 200.0, lat_off: float = 0.0,
             in_pit: bool = False) -> dict:
    return {"drv": drv, "x": 0.0, "y": 0.0, "dist": 0.0, "lat_off": lat_off, "speed": speed,
            "throttle": 100.0, "brake": False, "gear": 6, "rpm": 10000, "msector": msector,
            "gap_ahead_m": -1.0, "in_pit": in_pit}


def make_tick(t: float, cars: list[dict], lap: int = 1, track_status: str = "1") -> dict:
    return {"t": t, "lap": lap, "track_status": track_status, "cars": cars}


def make_det(det_id: str, t: float, drv: str, msector: int, dtype: str, severity: float,
             evidence: str = "test") -> dict:
    return {"id": det_id, "t": t, "drivers": [drv], "msector": msector, "type": dtype,
            "severity": severity, "evidence": evidence}


def all_recs_valid(recs: list[dict]) -> None:
    for i, r in enumerate(recs):
        check_rec(r, f"rec[{i}]")


# --- fixture-driven tests --------------------------------------------------


def merged_fixture_events() -> list[tuple[float, str, dict]]:
    ticks = [(t["t"], "tick", t) for t in read_jsonl(FIX / "ticks_sample.jsonl")]
    dets = [(d["t"], "detection", d) for d in read_jsonl(FIX / "detections_sample.jsonl")]
    # ticks sort before detections at equal t, so car_state reflects "now" before the detection fires
    return sorted(ticks + dets, key=lambda e: (e[0], e[1] == "detection"))


def run_engine(events: list[tuple[float, str, dict]]) -> tuple[RaceControl, list[dict]]:
    rc = RaceControl()
    all_recs: list[dict] = []
    for _, kind, data in events:
        if kind == "tick":
            all_recs.extend(rc.on_tick(data))
        else:
            all_recs.extend(rc.on_detection(data))
    return rc, all_recs


def test_fixture_incident_yellow_or_stronger_before_official() -> None:
    rc, recs = run_engine(merged_fixture_events())
    all_recs_valid(recs)
    sector9 = [r for r in recs if r["msector"] == 9 and r["flag"] != "CLEAR"]
    assert sector9, "expected at least one flag rec for sector 9"
    first = sector9[0]
    assert SECTOR_RANK.get(first["flag"], GLOBAL_RANK.get(first["flag"], 0)) >= SECTOR_RANK["YELLOW"]
    assert first["t"] <= 4390.18


def test_no_flicker_across_full_fixture() -> None:
    rc, recs = run_engine(merged_fixture_events())
    sector9_flags = [r["flag"] for r in recs if r["msector"] == 9 and r["flag"] in SECTOR_RANK]
    ranks = [SECTOR_RANK[f] for f in sector9_flags]
    # drop a trailing CLEAR (rank 0), that is expected de-escalation, not flicker.
    # a repeated rank is fine (e.g. an ANOMALY corroboration confidence bump keeps the same flag),
    # a rank that drops and later rises again is not.
    while ranks and ranks[-1] == 0:
        ranks.pop()
    assert ranks == sorted(ranks), f"sector 9 flag ranks are not monotonic: {ranks}"


# --- reset behaviour --------------------------------------------------------


def test_detection_lag_does_not_reset() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(100.0, [make_car("1", 5)]))
    # a detection carrying an older t than the latest tick must not wipe state
    recs = rc.on_detection(make_det("det-1", 98.0, "1", 5, "IMPACT", 0.9))
    assert recs and recs[0]["flag"] == "YELLOW"
    assert rc.t == 100.0


def test_small_backward_tick_does_not_reset() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(100.0, [make_car("1", 5)]))
    rc.on_detection(make_det("det-1", 100.0, "1", 5, "IMPACT", 0.9))
    rc.on_tick(make_tick(100.0 - RESET_JUMP_S, [make_car("1", 5)]))
    assert rc.sectors[5].flag == "YELLOW", "wobble within RESET_JUMP_S must not reset"


def test_real_backward_jump_resets_and_replay_is_deterministic() -> None:
    rc = RaceControl()

    def drive() -> list[dict]:
        recs = []
        recs.extend(rc.on_tick(make_tick(100.0, [make_car("1", 5)])))
        recs.extend(rc.on_detection(make_det("det-1", 100.0, "1", 5, "IMPACT", 0.9)))
        return recs

    first = drive()
    assert first and first[0]["flag"] == "YELLOW"
    # loop-back style jump, well past RESET_JUMP_S
    rc.on_tick(make_tick(100.0 - RESET_JUMP_S - 50.0, [make_car("1", 5)]))
    assert rc.sectors == {}
    assert rc.global_.flag == "CLEAR"

    second = drive()
    assert second and second[0]["flag"] == "YELLOW"


# --- car state / staleness --------------------------------------------------


def test_missing_car_keeps_last_known_state() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9)]))
    rc.on_tick(make_tick(0.25, []))  # car 23 absent from this tick
    assert rc.car_state["23"]["msector"] == 9
    assert rc.car_state["23"]["t"] == 0.0


def test_stale_car_releases_sector_for_clearing() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9, lat_off=0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.85))
    assert rc.sectors[9].flag == "DOUBLE_YELLOW"

    # car 23 is never seen again: once it is stale it must stop pinning the sector,
    # which starts the clear countdown; the clear itself needs one more window after that
    stale_t = STALE_CAR_S + 1.0
    rc.on_tick(make_tick(stale_t, [make_car("99", 1)]))
    assert rc.sectors[9].empty_since_t == stale_t

    after = stale_t + SECTOR_CLEAR_AFTER_S + 1.0
    recs = rc.on_tick(make_tick(after, [make_car("99", 1)]))
    clears = [r for r in recs if r["msector"] == 9 and r["flag"] == "CLEAR"]
    assert clears, "stale car must release its sector so it can clear"


# --- escalate immediately, de-escalate only after hold ----------------------


def test_no_downgrade_before_hold_and_clear_window() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9, lat_off=0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.85))
    assert rc.sectors[9].flag == "DOUBLE_YELLOW"

    # a car missing from a tick does not mean it left the sector: it must actually be
    # observed in a different sector before the clear countdown can start. The clearing
    # check also will not start that countdown until the hold gate itself has passed, so
    # the countdown effectively begins at the first tick at/after MIN_HOLD_S.
    moved_t = MIN_HOLD_S + 0.1
    rc.on_tick(make_tick(moved_t, [make_car("23", 10, speed=200.0, lat_off=0.0)]))
    assert rc.sectors[9].empty_since_t == moved_t

    just_before = moved_t + SECTOR_CLEAR_AFTER_S - 0.5
    recs = rc.on_tick(make_tick(just_before, [make_car("23", 10)]))
    assert all(r["msector"] != 9 or r["flag"] != "CLEAR" for r in recs)
    assert rc.sectors[9].flag == "DOUBLE_YELLOW"

    after = moved_t + SECTOR_CLEAR_AFTER_S + 0.5
    recs = rc.on_tick(make_tick(after, [make_car("23", 10)]))
    assert any(r["msector"] == 9 and r["flag"] == "CLEAR" for r in recs)


# --- ANOMALY ------------------------------------------------------------


def test_anomaly_alone_capped_at_yellow_and_low_confidence() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9)]))
    recs = rc.on_detection(make_det("det-1", 0.0, "23", 9, "ANOMALY", 0.9))
    assert recs and recs[0]["flag"] == "YELLOW"
    assert recs[0]["confidence"] <= ANOMALY_MAX_CONF


def test_anomaly_below_threshold_does_nothing() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9)]))
    recs = rc.on_detection(make_det("det-1", 0.0, "23", 9, "ANOMALY", ANOMALY_MIN_SEV - 0.1))
    assert recs == []


def test_anomaly_corroboration_boosts_physical_rec_not_itself() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9)]))
    impact_recs = rc.on_detection(make_det("det-1", 0.0, "23", 9, "IMPACT", 0.9))
    base_conf = impact_recs[0]["confidence"]

    anomaly_recs = rc.on_detection(make_det("det-2", 1.0, "23", 9, "ANOMALY", 0.8))
    assert len(anomaly_recs) == 1
    boosted = anomaly_recs[0]
    assert boosted["flag"] == "YELLOW"  # unchanged, this is a confidence bump not an escalation
    assert boosted["confidence"] > base_conf
    assert "det-1" in boosted["source_detections"] and "det-2" in boosted["source_detections"]


# --- in_pit / DROPOUT ignoring -----------------------------------------


def test_stopped_ignored_when_in_pit() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9, in_pit=True)]))
    recs = rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.9))
    assert recs == []
    assert rc.sectors.get(9, None) is None or rc.sectors[9].flag == "CLEAR"


def test_dropout_ignored_while_global_sc() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9, lat_off=0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.85))
    rc.on_tick(make_tick(SC_STOPPED_HOLD_S, [make_car("23", 9, speed=0.0, lat_off=0.0)]))
    assert rc.global_.flag == "SC"

    recs = rc.on_detection(make_det("det-2", SC_STOPPED_HOLD_S, "44", 9, "DROPOUT", 0.95))
    assert recs == []


# --- sustained stop: on track vs off track ------------------------------


def test_sustained_stop_on_track_escalates_to_sc() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9, lat_off=0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.5))  # low severity, on-track only
    assert rc.sectors[9].flag == "DOUBLE_YELLOW"

    recs = rc.on_tick(make_tick(SC_STOPPED_HOLD_S - 0.1, [make_car("23", 9, speed=0.0, lat_off=0.0)]))
    assert rc.global_.flag == "CLEAR"

    recs = rc.on_tick(make_tick(SC_STOPPED_HOLD_S + 0.1, [make_car("23", 9, speed=0.0, lat_off=0.0)]))
    assert rc.global_.flag == "SC"
    assert any(r["flag"] == "SC" and r["msector"] == 9 for r in recs)


def test_sustained_stop_off_track_escalates_to_vsc_not_sc() -> None:
    rc = RaceControl()
    off_track = 20.0
    rc.on_tick(make_tick(0.0, [make_car("23", 9, lat_off=off_track)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.5))
    assert rc.sectors[9].flag == "YELLOW"  # low severity, off track

    rc.on_tick(make_tick(SC_STOPPED_HOLD_S + 0.1, [make_car("23", 9, speed=0.0, lat_off=off_track)]))
    assert rc.global_.flag == "CLEAR", "off track must not escalate at the on-track SC hold time"

    recs = rc.on_tick(make_tick(VSC_STOPPED_HOLD_S + 0.1, [make_car("23", 9, speed=0.0, lat_off=off_track)]))
    assert rc.global_.flag == "VSC"
    assert any(r["flag"] == "VSC" and r["msector"] == 9 for r in recs)


def test_second_stopped_detection_does_not_restart_timer() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9, lat_off=0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.5))
    rc.on_tick(make_tick(0.5, [make_car("23", 9, speed=0.0, lat_off=0.0)]))
    rc.on_detection(make_det("det-2", 0.5, "23", 9, "STOPPED", 0.6))  # same car, still stopped
    assert rc.sectors[9].stopped_since_t == 0.0, "stopped_since_t must not be overwritten"

    recs = rc.on_tick(make_tick(SC_STOPPED_HOLD_S + 0.1, [make_car("23", 9, speed=0.0, lat_off=0.0)]))
    assert rc.global_.flag == "SC"


# --- no re-emission at same/higher global level -----------------------------


def test_global_rec_not_reemitted_while_already_at_level() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9, lat_off=0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.85))
    rc.on_tick(make_tick(SC_STOPPED_HOLD_S + 0.1, [make_car("23", 9, speed=0.0, lat_off=0.0)]))
    assert rc.global_.flag == "SC"

    sc_recs = []
    for i in range(5):
        t = SC_STOPPED_HOLD_S + 0.1 + (i + 1) * 0.25
        recs = rc.on_tick(make_tick(t, [make_car("23", 9, speed=0.0, lat_off=0.0)]))
        sc_recs.extend(r for r in recs if r["flag"] == "SC")
    assert sc_recs == [], "SC must not be re-emitted every tick while already active"


# --- global de-escalation ---------------------------------------------------


def test_global_clears_after_cause_sector_clears_and_hold() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9, lat_off=0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.85))
    t = SC_STOPPED_HOLD_S + 0.1
    rc.on_tick(make_tick(t, [make_car("23", 9, speed=0.0, lat_off=0.0)]))
    assert rc.global_.flag == "SC"

    # car recovers and drives off, so sector 9 can start its own clear countdown
    t += 0.1
    rc.on_tick(make_tick(t, [make_car("23", 10, speed=200.0, lat_off=0.0)]))
    t += MIN_HOLD_S + SECTOR_CLEAR_AFTER_S + 1.0
    recs = rc.on_tick(make_tick(t, [make_car("23", 10)]))
    assert rc.sectors[9].flag == "CLEAR"

    # now global needs its own hold + clear-after on top of that
    t += MIN_HOLD_S + GLOBAL_CLEAR_AFTER_S + 1.0
    recs = rc.on_tick(make_tick(t, [make_car("23", 10)]))
    assert rc.global_.flag == "CLEAR"
    assert any(r["flag"] == "CLEAR" and r["msector"] == 9 for r in recs)


# --- rec contract shape ------------------------------------------------


def test_all_recs_match_contract_shape() -> None:
    rc, recs = run_engine(merged_fixture_events())
    assert recs
    all_recs_valid(recs)
    for r in recs:
        assert isinstance(r["msector"], int)

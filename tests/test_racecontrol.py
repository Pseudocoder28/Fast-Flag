"""Tests for the race control engine (src/racecontrol/engine.py). Run: python -m pytest"""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_contracts import check_rec

from src.racecontrol.engine import (
    ANOMALY_MAX_CONF,
    ANOMALY_MIN_SEV,
    GLOBAL_CLEAR_AFTER_S,
    GLOBAL_MIN_HOLD_S,
    GLOBAL_RANK,
    IMPACT_BEFORE_STOP_S,
    MIN_HOLD_S,
    PARKED_CAR_S,
    RESET_JUMP_S,
    SC_STOPPED_HOLD_S,
    SECTOR_CLEAR_AFTER_S,
    SECTOR_RANK,
    STALE_CAR_S,
    VSC_STOPPED_HOLD_S,
    RaceControl,
)

FIX = Path(__file__).resolve().parent.parent / "fixtures"
TICK_S = 0.25   # the real stream's tick spacing


def read_jsonl(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def make_car(drv: str, msector: int, speed: float = 200.0, lat_off: float = 0.0,
             in_pit: bool = False, x: float = 0.0, y: float = 0.0) -> dict:
    return {"drv": drv, "x": x, "y": y, "dist": 0.0, "lat_off": lat_off, "speed": speed,
            "throttle": 100.0, "brake": False, "gear": 6, "rpm": 10000, "msector": msector,
            "gap_ahead_m": -1.0, "in_pit": in_pit}


def moving_car(drv: str, msector: int, t: float) -> dict:
    """A car at racing speed: its position changes every tick."""
    return make_car(drv, msector, speed=200.0, x=55.0 * t)


def make_tick(t: float, cars: list[dict], lap: int = 1, track_status: str = "1") -> dict:
    return {"t": t, "lap": lap, "track_status": track_status, "cars": cars}


def tick_until(rc: RaceControl, t_from: float, t_to: float, cars) -> list[dict]:
    """Tick every TICK_S after t_from up to and including t_to, like the real stream (a gap
    over RESET_JUMP_S would count as a seek). cars is a list, or a function of t. Returns recs."""
    recs: list[dict] = []
    for i in range(1, round((t_to - t_from) / TICK_S) + 1):
        t = t_from + i * TICK_S
        out, did_reset = rc.on_tick(make_tick(t, cars(t) if callable(cars) else cars))
        assert not did_reset
        recs.extend(out)
    return recs


def make_det(det_id: str, t: float, drv: str, msector: int, dtype: str, severity: float,
             evidence: str = "test") -> dict:
    return {"id": det_id, "t": t, "drivers": [drv], "msector": msector, "type": dtype,
            "severity": severity, "evidence": evidence}


def crash(rc: RaceControl, t: float, drv: str, msector: int, stop_sev: float = 0.85) -> list[dict]:
    """An impact, then a stop, as the detectors report a crash (Albon, 2023_Australian:
    IMPACT at 4387.75, STOPPED at 4388.75). Returns the recs."""
    recs = rc.on_detection(make_det(f"det-i{drv}-{t}", t - 1.0, drv, msector, "IMPACT", 0.9))
    return recs + rc.on_detection(make_det(f"det-s{drv}-{t}", t, drv, msector, "STOPPED", stop_sev))


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
            recs, _ = rc.on_tick(data)
            all_recs.extend(recs)
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
    _, did_reset = rc.on_tick(make_tick(100.0 - RESET_JUMP_S, [make_car("1", 5)]))
    assert did_reset is False
    assert rc.sectors[5].flag == "YELLOW", "wobble within RESET_JUMP_S must not reset"


def test_on_tick_returns_did_reset_flag() -> None:
    rc = RaceControl()
    _, did_reset = rc.on_tick(make_tick(100.0, [make_car("1", 5)]))
    assert did_reset is False  # first tick ever, nothing to reset from

    _, did_reset = rc.on_tick(make_tick(100.0 - RESET_JUMP_S, [make_car("1", 5)]))
    assert did_reset is False  # within tolerance

    _, did_reset = rc.on_tick(make_tick(100.0 - RESET_JUMP_S - 50.0, [make_car("1", 5)]))
    assert did_reset is True  # real backward jump


def test_real_backward_jump_resets_and_replay_is_deterministic() -> None:
    rc = RaceControl()

    def drive() -> list[dict]:
        recs = []
        tick_recs, _ = rc.on_tick(make_tick(100.0, [make_car("1", 5)]))
        recs.extend(tick_recs)
        recs.extend(rc.on_detection(make_det("det-1", 100.0, "1", 5, "IMPACT", 0.9)))
        return recs

    first = drive()
    assert first and first[0]["flag"] == "YELLOW"
    # loop-back style jump, well past RESET_JUMP_S
    _, did_reset = rc.on_tick(make_tick(100.0 - RESET_JUMP_S - 50.0, [make_car("1", 5)]))
    assert did_reset is True
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
    others = lambda t: [moving_car("99", 1, t)]  # noqa: E731
    tick_until(rc, 0.0, STALE_CAR_S, others)
    assert rc.sectors[9].empty_since_t is None
    stale_t = STALE_CAR_S + TICK_S
    tick_until(rc, STALE_CAR_S, stale_t, others)
    assert rc.sectors[9].empty_since_t == stale_t

    recs = tick_until(rc, stale_t, stale_t + SECTOR_CLEAR_AFTER_S, others)
    clears = [r for r in recs if r["msector"] == 9 and r["flag"] == "CLEAR"]
    assert clears, "stale car must release its sector so it can clear"


# --- escalate immediately, de-escalate only after hold ----------------------


def test_no_downgrade_before_hold_and_clear_window() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9, lat_off=0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.85))
    assert rc.sectors[9].flag == "DOUBLE_YELLOW"

    # the car drives on into sector 10 at once. A car missing from a tick does not mean it
    # left (test_missing_car_keeps_last_known_state): it has to be seen in another sector.
    # The clear countdown cannot start before the MIN_HOLD_S gate, so it begins at the
    # first tick at MIN_HOLD_S, and the clear needs SECTOR_CLEAR_AFTER_S more.
    gone = lambda t: [moving_car("23", 10, t)]  # noqa: E731
    tick_until(rc, 0.0, MIN_HOLD_S - TICK_S, gone)
    assert rc.sectors[9].empty_since_t is None
    tick_until(rc, MIN_HOLD_S - TICK_S, MIN_HOLD_S, gone)
    assert rc.sectors[9].empty_since_t == MIN_HOLD_S

    clear_t = MIN_HOLD_S + SECTOR_CLEAR_AFTER_S
    recs = tick_until(rc, MIN_HOLD_S, clear_t - TICK_S, gone)
    assert all(r["msector"] != 9 or r["flag"] != "CLEAR" for r in recs)
    assert rc.sectors[9].flag == "DOUBLE_YELLOW"

    recs = tick_until(rc, clear_t - TICK_S, clear_t, gone)
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
    stopped = [make_car("23", 9, speed=0.0, lat_off=0.0)]
    rc.on_tick(make_tick(0.0, stopped))
    crash(rc, 0.0, "23", 9)
    tick_until(rc, 0.0, SC_STOPPED_HOLD_S, stopped)
    assert rc.global_.flag == "SC"

    recs = rc.on_detection(make_det("det-2", SC_STOPPED_HOLD_S, "44", 9, "DROPOUT", 0.95))
    assert recs == []


# --- sustained stop: SC after an impact, VSC without one -------------------------


def test_stop_after_impact_escalates_to_sc() -> None:
    rc = RaceControl()
    stopped = [make_car("23", 9, speed=0.0, lat_off=0.0)]
    rc.on_tick(make_tick(0.0, stopped))
    crash(rc, 0.0, "23", 9, stop_sev=0.5)  # low severity stop, but after an impact
    assert rc.sectors[9].flag == "DOUBLE_YELLOW"

    tick_until(rc, 0.0, SC_STOPPED_HOLD_S - TICK_S, stopped)
    assert rc.global_.flag == "CLEAR"

    recs = tick_until(rc, SC_STOPPED_HOLD_S - TICK_S, SC_STOPPED_HOLD_S, stopped)
    assert rc.global_.flag == "SC"
    sc = [r for r in recs if r["flag"] == "SC"]
    assert len(sc) == 1 and sc[0]["msector"] == 9 and "after an impact" in sc[0]["reason"]


def test_stop_without_impact_escalates_to_vsc_not_sc() -> None:
    """FastF1 puts stopped cars on the racing line even in run-off, so lateral offset cannot
    pick VSC or SC. A car that stopped without an impact gets a VSC, after the longer hold."""
    rc = RaceControl()
    stopped = [make_car("23", 9, speed=0.0, lat_off=0.0)]  # on the line, like stopped cars in the data
    rc.on_tick(make_tick(0.0, stopped))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.5))
    assert rc.sectors[9].flag == "DOUBLE_YELLOW"

    tick_until(rc, 0.0, VSC_STOPPED_HOLD_S - TICK_S, stopped)
    assert rc.global_.flag == "CLEAR", "no impact: no SC at the SC hold time, and no VSC yet"

    recs = tick_until(rc, VSC_STOPPED_HOLD_S - TICK_S, VSC_STOPPED_HOLD_S, stopped)
    assert rc.global_.flag == "VSC"
    vsc = [r for r in recs if r["flag"] == "VSC"]
    assert len(vsc) == 1 and vsc[0]["msector"] == 9 and "no impact" in vsc[0]["reason"]


def test_brief_stop_without_impact_escalates_nothing() -> None:
    """Wet 2023 Monaco: cars stopped for 3 to 5 s and rejoined while race control kept double
    yellows. Without an impact, a stop shorter than VSC_STOPPED_HOLD_S escalates nothing."""
    rc = RaceControl()

    def car(t: float) -> list[dict]:
        if t < 5.0:
            return [make_car("23", 9, speed=0.0, x=100.0)]
        return [make_car("23", 9, speed=120.0, x=100.0 + 33.0 * (t - 5.0))]

    rc.on_tick(make_tick(0.0, car(0.0)))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.85))
    recs = tick_until(rc, 0.0, 30.0, car)
    assert not any(r["flag"] in ("VSC", "SC", "RED") for r in recs)


def test_multi_with_the_car_counts_as_an_impact() -> None:
    rc = RaceControl()
    stopped = [make_car("23", 9, speed=0.0), make_car("16", 9, speed=0.0, x=40.0)]
    rc.on_tick(make_tick(0.0, stopped))
    multi = {"id": "det-1", "t": 0.0, "drivers": ["16", "23"], "msector": 9, "type": "MULTI",
             "severity": 0.6, "evidence": "test"}  # below MULTI_SC_SEV: no SC from the MULTI itself
    rc.on_detection(multi)
    assert rc.global_.flag == "CLEAR"

    rc.on_detection(make_det("det-2", 0.5, "23", 9, "STOPPED", 0.85))
    recs = tick_until(rc, 0.0, 0.5 + SC_STOPPED_HOLD_S, stopped)
    assert [r["flag"] for r in recs if r["flag"] in ("VSC", "SC")] == ["SC"]


def test_late_impact_upgrades_vsc_to_sc() -> None:
    """A car can be hit after it stopped (a MULTI with a second car). The VSC already out
    for the stopped car is upgraded to an SC."""
    rc = RaceControl()
    stopped = [make_car("23", 9, speed=0.0), make_car("44", 9, speed=0.0, x=40.0)]
    rc.on_tick(make_tick(0.0, stopped))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.85))
    tick_until(rc, 0.0, VSC_STOPPED_HOLD_S, stopped)
    assert rc.global_.flag == "VSC"

    multi = {"id": "det-2", "t": VSC_STOPPED_HOLD_S, "drivers": ["44", "23"], "msector": 9, "type": "MULTI",
             "severity": 0.6, "evidence": "test"}
    rc.on_detection(multi)
    recs = tick_until(rc, VSC_STOPPED_HOLD_S, VSC_STOPPED_HOLD_S + TICK_S, stopped)
    assert [r["flag"] for r in recs if r["flag"] in ("VSC", "SC")] == ["SC"]
    assert rc.global_.flag == "SC"


def test_old_impact_does_not_count() -> None:
    """A knock long before the stop (the car drove on) does not make the stop an SC."""
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [moving_car("23", 9, 0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "IMPACT", 0.9))
    t_stop = IMPACT_BEFORE_STOP_S + 10.0
    tick_until(rc, 0.0, t_stop, lambda t: [moving_car("23", 12, t)])

    stopped = [make_car("23", 12, speed=0.0, x=5000.0)]
    rc.on_detection(make_det("det-2", t_stop, "23", 12, "STOPPED", 0.85))
    recs = tick_until(rc, t_stop, t_stop + VSC_STOPPED_HOLD_S, stopped)
    assert [r["flag"] for r in recs if r["flag"] in ("VSC", "SC")] == ["VSC"]


def test_impact_on_another_car_does_not_count() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [make_car("23", 9, speed=0.0), moving_car("44", 9, 0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "44", 9, "IMPACT", 0.9))
    rc.on_detection(make_det("det-2", 0.0, "23", 9, "STOPPED", 0.85))
    recs = tick_until(rc, 0.0, VSC_STOPPED_HOLD_S,
                      lambda t: [make_car("23", 9, speed=0.0), moving_car("44", 10, t)])
    assert [r["flag"] for r in recs if r["flag"] in ("VSC", "SC")] == ["VSC"]


def test_second_stopped_detection_does_not_restart_timer() -> None:
    rc = RaceControl()
    stopped = [make_car("23", 9, speed=0.0, lat_off=0.0)]
    rc.on_tick(make_tick(0.0, [make_car("23", 9, lat_off=0.0)]))
    crash(rc, 0.0, "23", 9, stop_sev=0.5)
    tick_until(rc, 0.0, 0.5, stopped)
    rc.on_detection(make_det("det-2", 0.5, "23", 9, "STOPPED", 0.6))  # same car, still stopped
    stop = rc.sectors[9].stops["23"]
    assert stop.since_t == 0.0, "a second STOPPED for a car already on hold must not restart it"
    assert stop.first_t == 0.0

    tick_until(rc, 0.5, SC_STOPPED_HOLD_S, stopped)
    assert rc.global_.flag == "SC"


def test_speed_blip_does_not_cancel_sustained_stop() -> None:
    """Leclerc, 2023_Australian lap 1: impact and stop at t=3762, one 70 km/h sample at
    t=3766, then stopped again. The blip used to cancel the SC hold for good (no SC at all)."""
    rc = RaceControl()

    def car16(t: float) -> list[dict]:
        return [make_car("16", 5, speed=70.0 if t == 1.0 else 0.0, lat_off=0.0, x=100.0, y=100.0)]

    rc.on_tick(make_tick(0.0, car16(0.0)))
    crash(rc, 0.0, "16", 5)
    restart_t = 1.0 + TICK_S       # first tick stopped again after the blip
    recs = tick_until(rc, 0.0, restart_t + SC_STOPPED_HOLD_S - TICK_S, car16)
    assert not any(r["flag"] == "SC" for r in recs), "the hold restarts when the car stops again"

    recs = tick_until(rc, restart_t + SC_STOPPED_HOLD_S - TICK_S, restart_t + SC_STOPPED_HOLD_S, car16)
    assert [(r["flag"], r["msector"]) for r in recs] == [("SC", 5)]


# --- no re-emission at same/higher global level -----------------------------


def test_global_rec_not_reemitted_while_already_at_level() -> None:
    rc = RaceControl()
    stopped = [make_car("23", 9, speed=0.0, lat_off=0.0)]
    rc.on_tick(make_tick(0.0, stopped))
    crash(rc, 0.0, "23", 9)
    tick_until(rc, 0.0, SC_STOPPED_HOLD_S, stopped)
    assert rc.global_.flag == "SC"

    recs = tick_until(rc, SC_STOPPED_HOLD_S, SC_STOPPED_HOLD_S + 5.0, stopped)
    sc_recs = [r for r in recs if r["flag"] == "SC"]
    assert sc_recs == [], "SC must not be re-emitted every tick while already active"


# --- clearing and hysteresis (PROJECT_BRIEF.md Section 6.5) ------------------


def test_global_clears_after_cause_sector_clears_and_hold() -> None:
    rc = RaceControl()
    stopped = [make_car("23", 9, speed=0.0, lat_off=0.0)]
    rc.on_tick(make_tick(0.0, stopped))
    crash(rc, 0.0, "23", 9)
    sc_t = SC_STOPPED_HOLD_S
    tick_until(rc, 0.0, sc_t, stopped)
    assert rc.global_.flag == "SC"

    # car recovers and drives off, so sector 9 can start its own clear countdown
    gone = lambda t: [moving_car("23", 10, t)]  # noqa: E731
    sector_clear_t = sc_t + TICK_S + SECTOR_CLEAR_AFTER_S
    tick_until(rc, sc_t, sector_clear_t, gone)
    assert rc.sectors[9].flag == "CLEAR"
    assert rc.global_.flag == "SC", "a track-wide flag holds at least GLOBAL_MIN_HOLD_S"

    # the SC holds GLOBAL_MIN_HOLD_S from deployment, then needs GLOBAL_CLEAR_AFTER_S more
    clear_t = sc_t + GLOBAL_MIN_HOLD_S + GLOBAL_CLEAR_AFTER_S
    recs = tick_until(rc, sector_clear_t, clear_t - TICK_S, gone)
    assert rc.global_.flag == "SC" and not any(r["message"] == "TRACK CLEAR" for r in recs)
    recs = tick_until(rc, clear_t - TICK_S, clear_t, gone)
    assert rc.global_.flag == "CLEAR"
    assert any(r["flag"] == "CLEAR" and r["msector"] == 9 and r["message"] == "TRACK CLEAR" for r in recs)


def test_parked_car_releases_its_sector() -> None:
    """A retired car keeps reporting the same position for the rest of the session. After
    PARKED_CAR_S without moving it counts as recovered, even with a little GPS jitter."""
    rc = RaceControl()

    def wreck(t: float) -> list[dict]:
        jitter = 0.4 if round(t / TICK_S) % 2 else 0.0
        return [make_car("23", 9, speed=0.0, lat_off=30.0, x=500.0 + jitter, y=500.0)]

    rc.on_tick(make_tick(0.0, wreck(0.0)))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "IMPACT", 0.7))
    tick_until(rc, 0.0, PARKED_CAR_S - TICK_S, wreck)
    assert rc.sectors[9].flag == "YELLOW", "a car stopped for less than PARKED_CAR_S still holds its sector"

    recs = tick_until(rc, PARKED_CAR_S - TICK_S, PARKED_CAR_S + SECTOR_CLEAR_AFTER_S, wreck)
    assert any(r["msector"] == 9 and r["flag"] == "CLEAR" for r in recs)
    assert rc.on_detection(make_det("det-2", rc.t, "23", 9, "STOPPED", 0.9)) == [], \
        "a recovered car must not flag its sector again"


def test_two_incidents_second_escalates_after_first_clears() -> None:
    """The A9 bug: the wreck of the first incident kept reporting its position forever, so
    its sector and its SC never cleared and the second incident could never escalate."""
    rc = RaceControl()
    wreck = make_car("23", 9, speed=0.0, lat_off=0.0, x=500.0, y=500.0)
    first = lambda t: [wreck, moving_car("16", 2, t)]  # noqa: E731

    # incident 1: car 23 crashes in sector 9 and never moves again
    recs, _ = rc.on_tick(make_tick(0.0, first(0.0)))
    recs += crash(rc, 0.0, "23", 9)
    recs += tick_until(rc, 0.0, 300.0, first)
    assert [r["msector"] for r in recs if r["flag"] == "SC"] == [9]
    track_clear = [r for r in recs if r["message"] == "TRACK CLEAR"]
    assert len(track_clear) == 1, "the first SC must clear once the wreck counts as recovered"
    assert track_clear[0]["t"] >= PARKED_CAR_S
    assert rc.global_.flag == "CLEAR" and rc.sectors[9].flag == "CLEAR"

    # incident 2: car 16 crashes in sector 5, well after the first cleared
    second = [wreck, make_car("16", 5, speed=0.0, lat_off=0.0, x=900.0, y=900.0)]
    t2 = 320.0
    recs2 = tick_until(rc, 300.0, t2, second)
    recs2 += crash(rc, t2, "16", 5)
    recs2 += tick_until(rc, t2, t2 + SC_STOPPED_HOLD_S, second)
    assert [(r["flag"], r["msector"]) for r in recs2] == [("YELLOW", 5), ("DOUBLE_YELLOW", 5), ("SC", 5)]
    assert recs2[-1]["t"] > track_clear[0]["t"]

    track_wide = [r["message"] if r["message"] == "TRACK CLEAR" else r["flag"]
                  for r in recs + recs2 if r["flag"] in ("VSC", "SC", "RED") or r["message"] == "TRACK CLEAR"]
    assert track_wide == ["SC", "TRACK CLEAR", "SC"]


def test_sc_waits_for_every_sector_that_supports_it() -> None:
    """A second crash while the SC is out supports it: the SC must not clear when the first
    sector clears while the second one is still flagged."""
    rc = RaceControl()
    car23 = make_car("23", 9, speed=0.0, lat_off=0.0, x=500.0, y=500.0)
    rc.on_tick(make_tick(0.0, [car23, moving_car("16", 2, 0.0)]))
    crash(rc, 0.0, "23", 9)
    tick_until(rc, 0.0, 10.0, lambda t: [car23, moving_car("16", 2, t)])
    assert rc.global_.flag == "SC"

    car16 = make_car("16", 5, speed=0.0, lat_off=0.0, x=900.0, y=900.0)
    crash(rc, 10.0, "16", 5)
    recs = tick_until(rc, 10.0, 20.0, [car23, car16])
    assert not any(r["flag"] == "SC" for r in recs), "no second SC rec while the SC is out"
    assert rc.global_.cause_sectors == {9, 5}

    # car 23 is recovered and driven off, car 16 is still stopped: the SC stays
    tick_until(rc, 20.0, 120.0, lambda t: [moving_car("23", 10, t), car16])
    assert rc.sectors[9].flag == "CLEAR" and rc.sectors[5].flag == "DOUBLE_YELLOW"
    assert rc.global_.flag == "SC", "sector 5 still supports the SC"

    # car 16 drives off too, then the SC clears
    tick_until(rc, 120.0, 140.0, lambda t: [moving_car("23", 10, t), moving_car("16", 6, t)])
    assert rc.sectors[5].flag == "CLEAR" and rc.global_.flag == "CLEAR"


# --- several stopped cars in one sector ----------------------------------------


def test_crash_in_a_sector_where_another_car_stopped_without_impact_gets_sc() -> None:
    """Each sector tracks every stopped car. Car 44 crashes 6 s after car 23 stopped without
    an impact (more than the detector's 5 s MULTI window, so no MULTI): car 44's SC must not
    be lost behind car 23's slower VSC."""
    rc = RaceControl()
    car23 = make_car("23", 9, speed=0.0, x=100.0)
    rc.on_tick(make_tick(0.0, [car23, moving_car("44", 9, 0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "STOPPED", 0.85))
    tick_until(rc, 0.0, 6.0, lambda t: [car23, moving_car("44", 9, t)])
    crash(rc, 6.0, "44", 9)
    recs = tick_until(rc, 6.0, 6.0 + VSC_STOPPED_HOLD_S, [car23, make_car("44", 9, speed=0.0, x=300.0)])
    esc = [(r["t"], r["flag"]) for r in recs if r["flag"] in ("VSC", "SC")]
    assert esc == [(6.0 + SC_STOPPED_HOLD_S, "SC")], "car 23's later VSC call only supports the SC"
    assert "car 44" in next(r["reason"] for r in recs if r["flag"] == "SC")


def test_crashed_car_keeps_its_sc_when_another_car_stops_during_its_blip() -> None:
    """Car 23 crashes; one noisy 40 km/h sample pauses its hold just as car 44 stops without an
    impact. Car 23 must still get its SC; car 44 must not take its place."""
    rc = RaceControl()

    def cars(t: float) -> list[dict]:
        return [make_car("23", 9, speed=40.0 if t == 1.0 else 0.0, x=100.0), make_car("44", 9, speed=0.0, x=300.0)]

    rc.on_tick(make_tick(0.0, cars(0.0)))
    crash(rc, 0.0, "23", 9)
    tick_until(rc, 0.0, 1.0, cars)
    rc.on_detection(make_det("det-44", 1.0, "44", 9, "STOPPED", 0.85))
    recs = tick_until(rc, 1.0, 1.0 + TICK_S + SC_STOPPED_HOLD_S, cars)
    esc = [r for r in recs if r["flag"] in ("VSC", "SC")]
    assert [r["flag"] for r in esc] == ["SC"] and "car 23" in esc[0]["reason"]


def test_second_stopped_car_holds_the_sector() -> None:
    """Every car with a detection in a flagged sector counts as a flagged car: the sector
    stays flagged while car 44 is stopped there, even after car 23 is driven away."""
    rc = RaceControl()
    car44 = make_car("44", 9, speed=0.0, x=300.0)
    rc.on_tick(make_tick(0.0, [make_car("23", 9, speed=0.0, x=100.0), car44]))
    crash(rc, 0.0, "23", 9)
    rc.on_detection(make_det("det-44", 0.5, "44", 9, "STOPPED", 0.85))
    tick_until(rc, 0.0, 30.0, lambda t: [moving_car("23", 10, t), car44])
    assert rc.sectors[9].flag == "DOUBLE_YELLOW"


# --- the impact lookback -------------------------------------------------------


def test_impact_lookback_edge() -> None:
    """An IMPACT exactly IMPACT_BEFORE_STOP_S before the stop still counts (the car limped on
    in the sector, then stopped); one tick earlier it does not."""

    def run(t_imp: float) -> list[tuple[float, str]]:
        rc = RaceControl()
        t_stop = 50.0
        limping = lambda t: [make_car("23", 9, speed=40.0, x=10.0 * t)]  # noqa: E731
        rc.on_tick(make_tick(t_imp, limping(t_imp)))
        rc.on_detection(make_det("det-1", t_imp, "23", 9, "IMPACT", 0.9))
        tick_until(rc, t_imp, t_stop, limping)
        rc.on_detection(make_det("det-2", t_stop, "23", 9, "STOPPED", 0.85))
        recs = tick_until(rc, t_stop, t_stop + VSC_STOPPED_HOLD_S, [make_car("23", 9, speed=0.0, x=500.0)])
        return [(r["t"], r["flag"]) for r in recs if r["flag"] in ("VSC", "SC")]

    assert IMPACT_BEFORE_STOP_S == 30.0, "PROJECT_BRIEF.md 6.5 documents 30 s: update it with the constant"
    assert run(50.0 - 30.0) == [(50.0 + SC_STOPPED_HOLD_S, "SC")]
    assert run(50.0 - 30.0 - TICK_S) == [(50.0 + VSC_STOPPED_HOLD_S, "VSC")]


def test_lookback_starts_at_the_first_stop_not_the_latest_hold() -> None:
    """Impact at 0, stop at 1, then the car twitches (a 45 km/h sample every 2.5 s) until 40 s,
    so its hold keeps restarting long after IMPACT_BEFORE_STOP_S. It is still the same stop."""
    rc = RaceControl()

    def car(t: float) -> list[dict]:
        twitch = 1.0 < t < 40.0 and round(t / TICK_S) % 10 == 0
        return [make_car("23", 9, speed=45.0 if twitch else 0.0, x=100.0)]

    rc.on_tick(make_tick(0.0, car(0.0)))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "IMPACT", 0.9))
    tick_until(rc, 0.0, 1.0, car)
    rc.on_detection(make_det("det-2", 1.0, "23", 9, "STOPPED", 0.85))
    recs = tick_until(rc, 1.0, 55.0, car)
    esc = [r for r in recs if r["flag"] in ("VSC", "SC")]
    assert [r["flag"] for r in esc] == ["SC"] and "after an impact" in esc[0]["reason"]
    assert esc[0]["t"] > 40.0


def test_same_car_stopping_again_keeps_its_first_stop() -> None:
    """Impact at 0, stop at 5, the car creeps on at 40 km/h inside the sector and gets a new
    STOPPED at 40 s. The detector judges speed against the reference, so that STOPPED can
    arrive while the engine still sees the car above STOPPED_SPEED_KMH (hold paused). The
    lookback still runs from the first stop at 5 s."""
    rc = RaceControl()

    def car(t: float) -> list[dict]:
        creeping = 6.0 < t <= 40.0
        return [make_car("23", 9, speed=40.0 if creeping else 0.0, x=100.0 + 10.0 * min(max(t - 6.0, 0.0), 34.0))]

    rc.on_tick(make_tick(0.0, car(0.0)))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "IMPACT", 0.9))
    tick_until(rc, 0.0, 5.0, car)
    rc.on_detection(make_det("det-2", 5.0, "23", 9, "STOPPED", 0.85))
    tick_until(rc, 5.0, 40.0, car)
    rc.on_detection(make_det("det-3", 40.0, "23", 9, "STOPPED", 0.85))
    assert rc.sectors[9].stops["23"].first_t == 5.0
    recs = tick_until(rc, 40.0, 55.0, car)
    assert [(r["t"], r["flag"]) for r in recs if r["flag"] in ("VSC", "SC")] == [(40.0 + SC_STOPPED_HOLD_S, "SC")]


def test_newer_impact_counts_after_an_old_one() -> None:
    """A knock at 0 (the car drove on), then a real crash at 100 s: the fresh IMPACT counts."""
    rc = RaceControl()
    rc.on_tick(make_tick(0.0, [moving_car("23", 9, 0.0)]))
    rc.on_detection(make_det("det-1", 0.0, "23", 9, "IMPACT", 0.9))
    tick_until(rc, 0.0, 100.0, lambda t: [moving_car("23", 12, t)])
    crash(rc, 100.0, "23", 12)
    recs = tick_until(rc, 100.0, 100.0 + SC_STOPPED_HOLD_S, [make_car("23", 12, speed=0.0, x=9000.0)])
    assert [(r["t"], r["flag"]) for r in recs if r["flag"] in ("VSC", "SC")] == [(100.0 + SC_STOPPED_HOLD_S, "SC")]


def test_reset_forgets_impacts() -> None:
    """After a seek back, an impact the engine saw later in session time is from the future:
    it must not turn an earlier stop into an SC (no future leakage)."""
    rc = RaceControl()
    rc.on_tick(make_tick(100.0, [moving_car("23", 9, 100.0)]))
    rc.on_detection(make_det("det-1", 100.0, "23", 9, "IMPACT", 0.9))
    _, did_reset = rc.on_tick(make_tick(50.0, [moving_car("23", 9, 50.0)]))
    assert did_reset and rc.impact_t == {}

    tick_until(rc, 50.0, 60.0, lambda t: [moving_car("23", 9, t)])
    rc.on_detection(make_det("det-2", 60.0, "23", 9, "STOPPED", 0.85))
    recs = tick_until(rc, 60.0, 60.0 + VSC_STOPPED_HOLD_S, [make_car("23", 9, speed=0.0, x=1.0)])
    assert [r["flag"] for r in recs if r["flag"] in ("VSC", "SC")] == ["VSC"]


# --- seeks ---------------------------------------------------------------


def test_forward_seek_resets() -> None:
    rc = RaceControl()
    rc.on_tick(make_tick(100.0, [make_car("1", 5)]))
    rc.on_detection(make_det("det-1", 100.0, "1", 5, "IMPACT", 0.9))
    _, did_reset = rc.on_tick(make_tick(100.0 + RESET_JUMP_S, [make_car("1", 5)]))
    assert did_reset is False and rc.sectors[5].flag == "YELLOW"

    _, did_reset = rc.on_tick(make_tick(100.0 + RESET_JUMP_S + 50.0, [make_car("1", 5)]))
    assert did_reset is True, "a jump forward (seek) wipes flags from before the jump"
    assert rc.sectors == {}


# --- rec contract shape ------------------------------------------------


def test_all_recs_match_contract_shape() -> None:
    rc, recs = run_engine(merged_fixture_events())
    assert recs
    all_recs_valid(recs)
    for r in recs:
        assert isinstance(r["msector"], int)

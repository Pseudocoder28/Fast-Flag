"""Race control engine: per-sector flag state machine + global (VSC/SC/RED) state.

Pure logic, no I/O. Clearing and hysteresis follow PROJECT_BRIEF.md Section 6.5:
- Flags escalate at once and only come down after a hold, so they do not flicker.
- A sector returns to CLEAR once no flagged car has remained in it for
  SECTOR_CLEAR_AFTER_S. A flagged car stops counting as in the sector when it
  drives out, enters the pit lane, sends no data for STALE_CAR_S, or has not moved
  for PARKED_CAR_S. The last rule matters: a retired car keeps reporting its last
  position for the rest of the session, so without it its sector, and any SC it
  caused, would never clear and a later crash could never escalate.
- A track-wide flag (VSC, SC, RED) holds at least GLOBAL_MIN_HOLD_S and clears
  GLOBAL_CLEAR_AFTER_S after every sector that caused or supported it is CLEAR.
- A stopped car gets an SC after SC_STOPPED_HOLD_S when our IMPACT or MULTI
  detection involved it (from IMPACT_BEFORE_STOP_S before its stop onward), and a
  VSC after VSC_STOPPED_HOLD_S otherwise. A later IMPACT or MULTI upgrades that VSC
  to an SC. Every stopped car in a sector is tracked and the highest call wins, so
  a crash is never hidden behind another car's stop. Lateral offset cannot make this
  call: FastF1 positions of stopped cars sit on the racing line even in run-off
  (docs/charts/escalation.md).
- ANOMALY is an advisory: it never raises a flag on its own (86 of its 89 alerts on
  the training races matched no official incident, docs/charts/detect_eval.md). It
  only raises the confidence of a sector that a physical detection flagged within
  ANOMALY_CORROBORATION_WINDOW_S. The dashboard shows it as a watch marker.
- A jump of more than RESET_JUMP_S in tick time, back or forward (seek or loop),
  wipes all flag state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- thresholds -------------------------------------------------------------

RESET_JUMP_S = 2.0
STOPPED_HIGH_SEV = 0.8
ON_TRACK_LAT_OFF_M = 8.0
DROPOUT_HIGH_SEV = 0.9
MULTI_SC_SEV = 0.85
RED_SEV = 0.95
ANOMALY_CORROBORATION_WINDOW_S = 5.0
ANOMALY_CORROBORATION_BOOST = 0.15
SC_STOPPED_HOLD_S = 3.0      # stopped after an impact: SC after this long
VSC_STOPPED_HOLD_S = 10.0    # stopped without one: VSC after this long
IMPACT_BEFORE_STOP_S = 30.0  # an IMPACT or MULTI this long before the stop still counts
STOPPED_SPEED_KMH = 30.0
STALE_CAR_S = 120.0
PARKED_MOVE_M = 3.0        # a car that stays within this distance of one spot ...
PARKED_CAR_S = 180.0       # ... for this long counts as recovered (about recovery time)
MIN_HOLD_S = 2.0
SECTOR_CLEAR_AFTER_S = 5.0
GLOBAL_MIN_HOLD_S = 60.0
GLOBAL_CLEAR_AFTER_S = 5.0
RISK_SC_SUPPORT = 0.5
RISK_CONF_BOOST = 0.1
MAX_CONF = 0.95

SECTOR_RANK = {"CLEAR": 0, "YELLOW": 1, "DOUBLE_YELLOW": 2}
GLOBAL_RANK = {"CLEAR": 0, "VSC": 1, "SC": 2, "RED": 3}


@dataclass
class Stop:
    """One stopped car in a sector."""
    first_t: float            # when this stop was first detected: the impact lookback starts here
    since_t: float | None     # start of the current hold, None while the car moves (paused)
    severity: float


@dataclass
class SectorState:
    flag: str = "CLEAR"
    since_t: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    cause_drivers: set[str] = field(default_factory=set)
    cause_detection_ids: list[str] = field(default_factory=list)
    stops: dict[str, Stop] = field(default_factory=dict)    # every stopped car in the sector, by drv
    empty_since_t: float | None = None


@dataclass
class GlobalState:
    flag: str = "CLEAR"
    since_t: float = 0.0
    cause_sector: int | None = None                           # first cause, used as the rec msector
    cause_sectors: set[int] = field(default_factory=set)      # every sector that caused or supports it
    confidence: float = 0.0
    reason: str = ""
    empty_since_t: float | None = None


def _sector_message(flag: str, msector: int) -> str:
    if flag == "CLEAR":
        return f"CLEAR IN TRACK SECTOR {msector}"
    return f"{flag.replace('_', ' ')} IN TRACK SECTOR {msector}"


def _global_message(flag: str) -> str:
    return {
        "SC": "SAFETY CAR DEPLOYED",
        "VSC": "VIRTUAL SAFETY CAR DEPLOYED",
        "RED": "RED FLAG",
        "CLEAR": "TRACK CLEAR",
    }[flag]


class RaceControl:
    """Consumes tick/detection/risk envelopes, emits rec envelopes (Section 7.4)."""

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self.t: float | None = None
        self.sectors: dict[int, SectorState] = {}
        self.global_ = GlobalState()
        self.car_state: dict[str, dict] = {}
        self.risk: dict[str, dict] = {}
        self.impact_t: dict[str, float] = {}      # drv -> latest IMPACT or MULTI detection with that car
        self.last_physical_by_sector: dict[int, list[tuple[float, str]]] = {}
        # rec id counter is not reset here, ids stay unique across a reset

    # rec ids are assigned lazily so a fresh instance and a reset instance
    # both keep counting up, never reusing an id within one process lifetime
    _seq: int = 1

    def _next_rec_id(self) -> str:
        rec_id = f"rec-{self._seq:06d}"
        self._seq += 1
        return rec_id

    def _make_rec(
        self,
        t: float,
        msector: int,
        flag: str,
        confidence: float,
        reason: str,
        message: str,
        source_detections: list[str],
    ) -> dict:
        return {
            "id": self._next_rec_id(),
            "t": t,
            "msector": msector,
            "flag": flag,
            "confidence": round(min(confidence, MAX_CONF), 4),
            "reason": reason,
            "message": message,
            "source_detections": list(source_detections),
        }

    # --- public API ----------------------------------------------------

    def on_tick(self, tick: dict) -> tuple[list[dict], bool]:
        new_t = tick["t"]
        did_reset = self.t is not None and abs(new_t - self.t) > RESET_JUMP_S
        if did_reset:
            self._reset_state()
        self.t = new_t

        for car in tick["cars"]:
            self._update_car(car, new_t)

        recs: list[dict] = []
        recs.extend(self._check_sustained_stop())
        recs.extend(self._check_sector_clearing())
        recs.extend(self._check_global_clearing())
        return recs, did_reset

    def on_detection(self, det: dict) -> list[dict]:
        det_t = det["t"]
        msector = det["msector"]
        dtype = det["type"]

        if dtype == "ANOMALY":
            return self._on_anomaly(det)

        if dtype in ("IMPACT", "MULTI"):
            for d in det["drivers"]:
                self.impact_t[d] = max(det_t, self.impact_t.get(d, det_t))

        sec = self.sectors.setdefault(msector, SectorState())
        drv = det["drivers"][0]
        car = self.car_state.get(drv, {})

        if (car.get("in_pit") or self._parked(car)) and dtype in ("STOPPED", "DROPOUT"):
            return []   # in the pit lane, or a retired car that already counts as recovered
        if dtype == "DROPOUT" and self.global_.flag in ("SC", "VSC", "RED"):
            return []

        self._record_physical(msector, det_t, det["id"])

        target, conf, reason = self._physical_target(det, sec, car, drv)

        if dtype == "STOPPED":
            stop = sec.stops.get(drv)
            if stop is None:
                sec.stops[drv] = Stop(first_t=det_t, since_t=det_t, severity=det["severity"])
            elif stop.since_t is None:
                stop.since_t = det_t        # stopped again: restart the hold, keep when the stop began
                stop.severity = max(stop.severity, det["severity"])

        # every car with a physical detection here is a flagged car: the sector waits for it to leave
        sec.cause_drivers |= set(det["drivers"])

        recs: list[dict] = []
        if SECTOR_RANK[target] > SECTOR_RANK[sec.flag]:
            sec.flag = target
            sec.since_t = det_t
            sec.confidence = min(MAX_CONF, conf)
            sec.reason = reason
            sec.cause_detection_ids.append(det["id"])
            recs.append(
                self._make_rec(
                    det_t, msector, target, sec.confidence, reason,
                    _sector_message(target, msector), sec.cause_detection_ids,
                )
            )

        if dtype == "MULTI":
            recs.extend(self._maybe_escalate_global_from_multi(det, msector))

        return recs

    def on_risk(self, risk: dict) -> list[dict]:
        self.risk[risk["drv"]] = risk
        return []

    # --- car state --------------------------------------------------------

    def _update_car(self, car: dict, t: float) -> None:
        prev = self.car_state.get(car["drv"])
        x, y = car["x"], car["y"]
        if prev is None or math.hypot(x - prev["anchor"][0], y - prev["anchor"][1]) > PARKED_MOVE_M:
            anchor, moved_t = (x, y), t
        else:
            anchor, moved_t = prev["anchor"], prev["moved_t"]
        self.car_state[car["drv"]] = {
            "msector": car["msector"],
            "speed": car["speed"],
            "lat_off": car["lat_off"],
            "in_pit": car["in_pit"],
            "t": t,
            "anchor": anchor,       # where the car was when it last moved more than PARKED_MOVE_M
            "moved_t": moved_t,
        }

    def _stale(self, car: dict) -> bool:
        return self.t - car["t"] > STALE_CAR_S

    def _parked(self, car: dict) -> bool:
        """Not moved for PARKED_CAR_S: a retired car, treated as recovered."""
        return "moved_t" in car and self.t - car["moved_t"] >= PARKED_CAR_S

    def _holds_sector(self, drv: str, msector: int) -> bool:
        """True while a flagged car still counts as being in the sector."""
        car = self.car_state.get(drv)
        if car is None or car["in_pit"] or car["msector"] != msector:
            return False
        return not self._stale(car) and not self._parked(car)

    # --- detection handling ---------------------------------------------

    def _record_physical(self, msector: int, t: float, det_id: str) -> None:
        self.last_physical_by_sector.setdefault(msector, []).append((t, det_id))

    def _recent_physical(self, msector: int, t: float) -> tuple[float, str] | None:
        entries = self.last_physical_by_sector.get(msector, [])
        for entry_t, det_id in reversed(entries):
            if abs(t - entry_t) <= ANOMALY_CORROBORATION_WINDOW_S:
                return entry_t, det_id
        return None

    def _on_anomaly(self, det: dict) -> list[dict]:
        """Advisory only: confirms a sector a physical detection flagged moments ago,
        never raises a flag by itself."""
        msector = det["msector"]
        det_t = det["t"]
        sec = self.sectors.setdefault(msector, SectorState())

        recent = self._recent_physical(msector, det_t)
        if recent is not None and sec.flag != "CLEAR":
            sec.confidence = min(MAX_CONF, sec.confidence + ANOMALY_CORROBORATION_BOOST)
            sec.cause_detection_ids.append(det["id"])
            reason = "confirmed by anomaly detector"
            sec.reason = reason
            return [
                self._make_rec(
                    det_t, msector, sec.flag, sec.confidence, reason,
                    _sector_message(sec.flag, msector), sec.cause_detection_ids,
                )
            ]

        return []

    def _physical_target(
        self, det: dict, sec: SectorState, car: dict, drv: str
    ) -> tuple[str, float, str]:
        dtype = det["type"]
        sev = det["severity"]

        if dtype == "STOPPED":
            on_track = abs(car.get("lat_off", 0.0)) <= ON_TRACK_LAT_OFF_M
            if sev >= STOPPED_HIGH_SEV or on_track:
                where = "on track" if on_track else "high severity"
                return "DOUBLE_YELLOW", min(MAX_CONF, sev), f"car {drv} stopped ({where})"
            return "YELLOW", min(MAX_CONF, sev), f"car {drv} stopped in run-off"

        if dtype == "MULTI":
            return "DOUBLE_YELLOW", min(MAX_CONF, sev), "multi-car incident"

        if dtype == "DROPOUT":
            if sev >= DROPOUT_HIGH_SEV:
                return "DOUBLE_YELLOW", min(MAX_CONF, sev), f"car {drv} dropout, high severity"
            return "YELLOW", min(MAX_CONF, sev), f"car {drv} dropout"

        # IMPACT, SPIN
        return "YELLOW", min(MAX_CONF, sev), f"car {drv} {dtype.lower()} signature"

    def _maybe_escalate_global_from_multi(self, det: dict, msector: int) -> list[dict]:
        sev = det["severity"]
        if sev >= RED_SEV:
            target = "RED"
        elif sev >= MULTI_SC_SEV:
            target = "SC"
        else:
            return []

        if GLOBAL_RANK[target] <= GLOBAL_RANK[self.global_.flag]:
            self.global_.cause_sectors.add(msector)   # supports the flag already out
            return []

        reason = "multi-car incident"
        conf = min(MAX_CONF, sev)
        return self._escalate_global(det["t"], msector, target, conf, reason, det["id"])

    def _escalate_global(
        self, t: float, msector: int, target: str, conf: float, reason: str, det_id: str | None
    ) -> list[dict]:
        if self.global_.flag == "CLEAR":
            self.global_.cause_sector = msector
            self.global_.cause_sectors = set()
        self.global_.cause_sectors.add(msector)
        self.global_.flag = target
        self.global_.since_t = t
        self.global_.confidence = conf
        self.global_.reason = reason
        self.global_.empty_since_t = None
        source = [det_id] if det_id else []
        return [self._make_rec(t, msector, target, conf, reason, _global_message(target), source)]

    # --- tick-driven checks ----------------------------------------------

    def _check_sustained_stop(self) -> list[dict]:
        """Every stopped car in every sector calls for SC after an impact, VSC without one,
        once its hold has run. The highest call is escalated, once per tick; every sector
        whose call the flag already out covers supports that flag (it holds it up)."""
        calls = []   # (rank, target, msector, drv, stop, elapsed, impact), in sector then drv order
        for msector, sec in self.sectors.items():
            for drv in sorted(sec.stops):
                stop = sec.stops[drv]
                car = self.car_state.get(drv)
                if car is None:
                    continue
                if not self._holds_sector(drv, msector):
                    del sec.stops[drv]   # drove out, pitted, no data or parked long enough to count as recovered
                    continue
                if car["speed"] > STOPPED_SPEED_KMH:
                    # moving again: pause the hold but keep watching the car, so one noisy
                    # speed sample cannot cancel the escalation of a car that stays put
                    stop.since_t = None
                    continue
                if stop.since_t is None:
                    stop.since_t = self.t

                elapsed = self.t - stop.since_t
                impact = self._impact_since(drv, stop.first_t - IMPACT_BEFORE_STOP_S)
                if impact and elapsed >= SC_STOPPED_HOLD_S:
                    calls.append((GLOBAL_RANK["SC"], "SC", msector, drv, stop, elapsed, impact))
                elif not impact and elapsed >= VSC_STOPPED_HOLD_S:
                    calls.append((GLOBAL_RANK["VSC"], "VSC", msector, drv, stop, elapsed, impact))
        if not calls:
            return []

        recs: list[dict] = []
        rank, target, msector, drv, stop, elapsed, impact = max(calls, key=lambda c: c[0])
        if rank > GLOBAL_RANK[self.global_.flag]:
            conf = min(MAX_CONF, stop.severity)
            if self.risk.get(drv, {}).get("risk_30s", 0.0) >= RISK_SC_SUPPORT:
                conf = min(MAX_CONF, conf + RISK_CONF_BOOST)
            reason = (f"car {drv} stopped for {elapsed:.1f} s after an impact" if impact
                      else f"car {drv} stopped for {elapsed:.1f} s, no impact detected")
            recs = self._escalate_global(self.t, msector, target, conf, reason, None)
        for c in calls:
            if c[0] <= GLOBAL_RANK[self.global_.flag]:
                self.global_.cause_sectors.add(c[2])
        return recs

    def _impact_since(self, drv: str, t0: float) -> bool:
        """Our IMPACT or MULTI detection involved the car at or after t0 (and, as detections
        only arrive up to now, at or before now)."""
        t_imp = self.impact_t.get(drv)
        return t_imp is not None and t_imp >= t0

    def _check_sector_clearing(self) -> list[dict]:
        recs: list[dict] = []
        for msector, sec in self.sectors.items():
            if sec.flag == "CLEAR":
                continue
            if self.t - sec.since_t < MIN_HOLD_S:
                continue

            holds = any(self._holds_sector(drv, msector) for drv in sec.cause_drivers)
            if holds:
                sec.empty_since_t = None
                continue
            if sec.empty_since_t is None:
                sec.empty_since_t = self.t
                continue
            if self.t - sec.empty_since_t < SECTOR_CLEAR_AFTER_S:
                continue

            recs.append(
                self._make_rec(
                    self.t, msector, "CLEAR", 1.0, "no flagged car remains in sector",
                    _sector_message("CLEAR", msector), [],
                )
            )
            sec.flag = "CLEAR"
            sec.since_t = self.t
            sec.confidence = 0.0
            sec.reason = ""
            sec.cause_drivers = set()
            sec.cause_detection_ids = []
            sec.stops.clear()
            sec.empty_since_t = None
        return recs

    def _check_global_clearing(self) -> list[dict]:
        if self.global_.flag == "CLEAR":
            return []
        if self.t - self.global_.since_t < GLOBAL_MIN_HOLD_S:
            return []

        cause_sector = self.global_.cause_sector
        still_flagged = any(
            s in self.sectors and self.sectors[s].flag != "CLEAR" for s in self.global_.cause_sectors
        )
        if still_flagged:
            self.global_.empty_since_t = None
            return []
        if self.global_.empty_since_t is None:
            self.global_.empty_since_t = self.t
            return []
        if self.t - self.global_.empty_since_t < GLOBAL_CLEAR_AFTER_S:
            return []

        rec = self._make_rec(
            self.t, cause_sector, "CLEAR", 1.0, "incident cleared",
            _global_message("CLEAR"), [],
        )
        self.global_.flag = "CLEAR"
        self.global_.since_t = self.t
        self.global_.cause_sectors = set()
        self.global_.confidence = 0.0
        self.global_.reason = ""
        self.global_.empty_since_t = None
        return [rec]

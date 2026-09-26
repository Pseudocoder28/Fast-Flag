"""Race control engine: per-sector flag state machine + global (VSC/SC/RED) state.

Pure logic, no I/O. See src/racecontrol/PLAN_B1.md for the design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- thresholds -------------------------------------------------------------

RESET_JUMP_S = 2.0
STOPPED_HIGH_SEV = 0.8
ON_TRACK_LAT_OFF_M = 8.0
DROPOUT_HIGH_SEV = 0.9
MULTI_SC_SEV = 0.85
RED_SEV = 0.95
ANOMALY_MIN_SEV = 0.75
ANOMALY_MAX_CONF = 0.6
ANOMALY_CORROBORATION_WINDOW_S = 5.0
ANOMALY_CORROBORATION_BOOST = 0.15
SC_STOPPED_HOLD_S = 3.0
VSC_STOPPED_HOLD_S = 10.0
STOPPED_SPEED_KMH = 30.0
STALE_CAR_S = 120.0
MIN_HOLD_S = 2.0
SECTOR_CLEAR_AFTER_S = 5.0
GLOBAL_CLEAR_AFTER_S = 5.0
RISK_SC_SUPPORT = 0.5
RISK_CONF_BOOST = 0.1
MAX_CONF = 0.95

SECTOR_RANK = {"CLEAR": 0, "YELLOW": 1, "DOUBLE_YELLOW": 2}
GLOBAL_RANK = {"CLEAR": 0, "VSC": 1, "SC": 2, "RED": 3}


@dataclass
class SectorState:
    flag: str = "CLEAR"
    since_t: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    cause_drivers: set[str] = field(default_factory=set)
    cause_detection_ids: list[str] = field(default_factory=list)
    stopped_since_t: float | None = None
    stopped_driver: str | None = None
    stopped_severity: float | None = None
    empty_since_t: float | None = None


@dataclass
class GlobalState:
    flag: str = "CLEAR"
    since_t: float = 0.0
    cause_sector: int | None = None
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
        did_reset = self.t is not None and new_t < self.t - RESET_JUMP_S
        if did_reset:
            self._reset_state()
        self.t = new_t

        for car in tick["cars"]:
            self.car_state[car["drv"]] = {
                "msector": car["msector"],
                "speed": car["speed"],
                "lat_off": car["lat_off"],
                "in_pit": car["in_pit"],
                "t": new_t,
            }

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

        sec = self.sectors.setdefault(msector, SectorState())
        drv = det["drivers"][0]
        car = self.car_state.get(drv, {})

        if car.get("in_pit") and dtype in ("STOPPED", "DROPOUT"):
            return []
        if dtype == "DROPOUT" and self.global_.flag in ("SC", "VSC", "RED"):
            return []

        self._record_physical(msector, det_t, det["id"])

        target, conf, reason = self._physical_target(det, sec, car, drv)

        if dtype == "STOPPED" and sec.stopped_since_t is None:
            sec.stopped_since_t = det_t
            sec.stopped_driver = drv
            sec.stopped_severity = det["severity"]

        recs: list[dict] = []
        if SECTOR_RANK[target] > SECTOR_RANK[sec.flag]:
            sec.flag = target
            sec.since_t = det_t
            sec.confidence = min(MAX_CONF, conf)
            sec.reason = reason
            sec.cause_drivers |= set(det["drivers"])
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

        if det["severity"] < ANOMALY_MIN_SEV:
            return []

        target = "YELLOW"
        conf = min(ANOMALY_MAX_CONF, det["severity"])
        if SECTOR_RANK[target] <= SECTOR_RANK[sec.flag]:
            return []

        drv = det["drivers"][0]
        reason = f"anomaly score high for car {drv}"
        sec.flag = target
        sec.since_t = det_t
        sec.confidence = conf
        sec.reason = reason
        sec.cause_drivers |= set(det["drivers"])
        sec.cause_detection_ids.append(det["id"])
        return [
            self._make_rec(
                det_t, msector, target, conf, reason,
                _sector_message(target, msector), sec.cause_detection_ids,
            )
        ]

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
            return []

        reason = "multi-car incident"
        conf = min(MAX_CONF, sev)
        return self._escalate_global(det["t"], msector, target, conf, reason, det["id"])

    def _escalate_global(
        self, t: float, msector: int, target: str, conf: float, reason: str, det_id: str | None
    ) -> list[dict]:
        self.global_.flag = target
        self.global_.since_t = t
        self.global_.cause_sector = msector
        self.global_.confidence = conf
        self.global_.reason = reason
        self.global_.empty_since_t = None
        source = [det_id] if det_id else []
        return [self._make_rec(t, msector, target, conf, reason, _global_message(target), source)]

    # --- tick-driven checks ----------------------------------------------

    def _check_sustained_stop(self) -> list[dict]:
        recs: list[dict] = []
        for msector, sec in self.sectors.items():
            if sec.stopped_since_t is None or sec.stopped_driver is None:
                continue
            car = self.car_state.get(sec.stopped_driver)
            if car is None:
                continue
            if car["in_pit"]:
                sec.stopped_since_t = None
                sec.stopped_driver = None
                sec.stopped_severity = None
                continue
            if self.t - car["t"] > STALE_CAR_S:
                continue
            if car["speed"] > STOPPED_SPEED_KMH:
                sec.stopped_since_t = None
                sec.stopped_driver = None
                sec.stopped_severity = None
                continue

            elapsed = self.t - sec.stopped_since_t
            on_track = abs(car["lat_off"]) <= ON_TRACK_LAT_OFF_M
            target: str | None = None
            if on_track and elapsed >= SC_STOPPED_HOLD_S:
                target = "SC"
            elif (not on_track) and elapsed >= VSC_STOPPED_HOLD_S:
                target = "VSC"

            if target is None or GLOBAL_RANK[target] <= GLOBAL_RANK[self.global_.flag]:
                continue

            conf = min(MAX_CONF, sec.stopped_severity or 0.0)
            risk = self.risk.get(sec.stopped_driver, {})
            if risk.get("risk_30s", 0.0) >= RISK_SC_SUPPORT:
                conf = min(MAX_CONF, conf + RISK_CONF_BOOST)
            reason = f"car {sec.stopped_driver} stopped {'on track' if on_track else 'off track'} for {elapsed:.1f} s"
            recs.extend(self._escalate_global(self.t, msector, target, conf, reason, None))
        return recs

    def _check_sector_clearing(self) -> list[dict]:
        recs: list[dict] = []
        for msector, sec in self.sectors.items():
            if sec.flag == "CLEAR":
                continue
            if self.t - sec.since_t < MIN_HOLD_S:
                continue

            holds = False
            for drv in sec.cause_drivers:
                car = self.car_state.get(drv)
                if car is None:
                    continue
                if self.t - car["t"] > STALE_CAR_S:
                    continue
                if car["msector"] == msector:
                    holds = True
                    break

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
            sec.stopped_since_t = None
            sec.stopped_driver = None
            sec.stopped_severity = None
            sec.empty_since_t = None
        return recs

    def _check_global_clearing(self) -> list[dict]:
        if self.global_.flag == "CLEAR":
            return []
        if self.t - self.global_.since_t < MIN_HOLD_S:
            return []

        cause_sector = self.global_.cause_sector
        cause = self.sectors.get(cause_sector) if cause_sector is not None else None
        cause_flag = cause.flag if cause is not None else "CLEAR"

        if cause_flag != "CLEAR":
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
        self.global_.confidence = 0.0
        self.global_.reason = ""
        self.global_.empty_since_t = None
        return [rec]

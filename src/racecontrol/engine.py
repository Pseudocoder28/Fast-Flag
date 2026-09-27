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
- A stopped car that rolls on into another sector at FOLLOW_STOP_KMH or less (a car
  losing power crawls over a sector boundary before it stops) takes its stop and its
  sector flag along, so its VSC or SC still comes. Before this (27 Sept), its old sector
  cleared behind it and no VSC came: 3 of race control's neutralisations on the training
  races (2023 Australia, 2025 Netherlands, 2025 Miami) were missed that way.
- ANOMALY is an advisory: it never raises a flag on its own (86 of its 89 alerts on
  the training races matched no official incident, docs/charts/detect_eval.md). It
  only raises the confidence of a sector that a physical detection flagged within
  ANOMALY_CORROBORATION_WINDOW_S. The dashboard shows it as a watch marker.
- A car that stopped after an impact is a crash site. It holds its sector, and so any
  VSC or SC it caused, even after it has not moved for PARKED_CAR_S: FastF1 keeps
  reporting a crashed car's last position long after marshals recover it, so "not
  moving" says nothing about the recovery (docs/charts/escalation.md). A crash site is
  released when the car moves CRASH_SITE_MOVE_M away, its data stops, or race control's
  own track status (in every tick) has been green for OFFICIAL_GREEN_S after race control
  reacted to it. If race control never reacts, it is released after PARKED_CAR_S as
  before.
- Our track-wide flag never clears while race control's own track status (in every tick)
  is SC, VSC or red (OFFICIAL_NEUTRAL): lifting a neutralisation is race control's call,
  only raising one is ours. Without this, a car that stopped with no impact (2023
  Australia, Magnussen: TRACK CLEAR 5 s before race control's red flag) or a crashed car
  moved a few metres by a crane would let us recommend green under race control's SC.
- Red flag, late in the race, which never blocks: while our VSC or SC is out for an
  incident that is still there and LATE_RED_MIN_LAPS to LATE_RED_LAPS laps are left, a RED
  rec (confidence LATE_RED_CONF): a red flag lets the race restart and finish racing instead
  of ending behind the Safety Car. The race length comes from the lap length before the race
  (race_laps_from_track). It is a soft overlay: the hard flag every other escalation compares
  against stays our VSC or SC, so a new incident elsewhere still escalates on its own. It
  ends with our track-wide flag (TRACK CLEAR).
  Why this rule (docs/lab/red_flag.md, training races): race control's reds follow late
  neutralisations or damage car data cannot see (barrier, gravel, debris, rain). The old red
  by time at a crash site caught 2 mid-race reds but sent 11 reds race control never called;
  a MULTI at RED_SEV caught 1 red and sent 7 it never called. A crash site still stopped
  RED_CRASH_SITE_S after its stop, once race control reacted, now only gets an advisory:
  its sector flag re-sent unchanged, "recovery taking long, race control may need a red
  flag". RED_BY_TIME and RED_FROM_MULTI bring the old reds back for comparison. With
  RED_BY_TIME, the red by time is tied to its crash site: released (race control green
  again, the car moved, no data), another crash site that waited as long takes it over;
  otherwise our VSC or SC is re-sent as the downgrade.
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
FOLLOW_STOP_KMH = 80.0       # a stopped car that rolls into another sector this slowly takes its flag along
STALE_CAR_S = 120.0
PARKED_MOVE_M = 3.0        # a car that stays within this distance of one spot ...
PARKED_CAR_S = 180.0       # ... for this long counts as recovered (about recovery time)
CRASH_SITE_MOVE_M = 20.0   # a crashed car this far from where it stopped was driven or craned away
OFFICIAL_GREEN_S = 20.0    # race control's track status green this long after it reacted: incident over
RED_CRASH_SITE_S = 120.0   # a crashed car still at its crash site this long after stopping ...
RED_BY_TIME = False        # ... was a red flag (soft); now only an advisory (docs/lab/red_flag.md)
RED_CRASH_SITE_CONF = 0.7  # a judgement call on time alone, not a physical signal: lower confidence
RED_FROM_MULTI = False     # a MULTI at RED_SEV or more called red: 1 of 8 matched a race control red
LATE_RED_LAPS = 4          # our VSC or SC out for a live incident with this many laps or fewer left: red
LATE_RED_MIN_LAPS = 2      # ... but not with fewer than this left: too late to restart the race
LATE_RED_CONF = 0.75
RACE_DISTANCE_M = 305_000.0      # an F1 race: the fewest laps that cover this distance ...
MONACO_DISTANCE_M = 260_000.0    # ... except Monaco
OFFICIAL_NEUTRAL = {"4", "5", "6", "7"}   # race control's track status: SC, red, VSC, VSC ending
GREEN_REASON_SLACK_S = 5.0  # a sector clear this soon after a green release is credited to race control
MIN_HOLD_S = 2.0
SECTOR_CLEAR_AFTER_S = 5.0
GLOBAL_MIN_HOLD_S = 60.0
GLOBAL_CLEAR_AFTER_S = 5.0
RISK_SC_SUPPORT = 0.5
RISK_CONF_BOOST = 0.1
MAX_CONF = 0.95

LATE_RED_KEY = "late-race"     # self.soft_red for a late-race red (a crash site's red holds its drv)

SECTOR_RANK = {"CLEAR": 0, "YELLOW": 1, "DOUBLE_YELLOW": 2}
GLOBAL_RANK = {"CLEAR": 0, "VSC": 1, "SC": 2, "RED": 3}


@dataclass
class Stop:
    """One stopped car in a sector."""
    first_t: float            # when this stop was first detected: the impact lookback starts here
    since_t: float | None     # start of the current hold, None while the car moves (paused)
    severity: float
    crash_site_done: bool = False   # this stop already made its crash site (at most one per stop)


@dataclass
class CrashSite:
    """A car that stopped after an impact: a hazard until it is moved, its data stops, or
    race control's own track status is back to green after it reacted to the crash."""
    msector: int
    stop_t: float             # when the car stopped at the site
    x: float
    y: float
    reacted: bool = False     # race control's track status left green after the crash
    green_since: float | None = None


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
    held_for_race_control: bool = False   # a crash site kept it out until race control's status was green


def _sector_message(flag: str, msector: int) -> str:
    if flag == "CLEAR":
        return f"CLEAR IN TRACK SECTOR {msector}"
    return f"{flag.replace('_', ' ')} IN TRACK SECTOR {msector}"


def race_laps_from_track(track: dict) -> int | None:
    """The scheduled race length, known before the race: the fewest laps that cover 305 km
    (260 km at Monaco), from the lap length in GET /track. It comes out 1 to 3 laps long on
    most circuits (the track's sector ends fall short of the real lap), so late-race rules
    see about that many laps too many left."""
    ends = [max(float(s.get("start_dist", 0.0)), float(s.get("end_dist", 0.0))) for s in track.get("msectors", [])]
    length = max(ends, default=0.0)
    if length < 1000.0:
        return None
    distance = MONACO_DISTANCE_M if "monaco" in str(track.get("race", "")).lower() else RACE_DISTANCE_M
    return math.ceil(distance / length)


def _global_message(flag: str) -> str:
    return {
        "SC": "SAFETY CAR DEPLOYED",
        "VSC": "VIRTUAL SAFETY CAR DEPLOYED",
        "RED": "RED FLAG",
        "CLEAR": "TRACK CLEAR",
    }[flag]


class RaceControl:
    """Consumes tick/detection/risk envelopes, emits rec envelopes (Section 7.4)."""

    def __init__(self, race_laps: int | None = None) -> None:
        self.race_laps = race_laps    # scheduled race length (race_laps_from_track); None: no late-race red
        self._reset_state()

    def _reset_state(self) -> None:
        self.t: float | None = None
        self.lap: int | None = None
        self.sectors: dict[int, SectorState] = {}
        self.global_ = GlobalState()
        self.car_state: dict[str, dict] = {}
        self.risk: dict[str, dict] = {}
        self.impact_t: dict[str, float] = {}      # drv -> latest IMPACT or MULTI detection with that car
        self.official_status = "1"                # race control's track status in the latest tick
        self.crash_sites: dict[str, CrashSite] = {}
        self.green_released: dict[int, float] = {}   # sector -> when race control's green released its crash site
        self.soft_red: str | None = None          # our soft red: LATE_RED_KEY, or (by time) the crash site's drv
        self.advised: set[str] = set()            # crash sites that already got the long-recovery advisory
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
        self.official_status = str(tick.get("track_status", "1"))
        if isinstance(tick.get("lap"), int):
            self.lap = tick["lap"]
        released = self._update_crash_sites()

        recs: list[dict] = []
        if self.soft_red is not None and self.soft_red in released:
            recs.extend(self._end_soft_red(self.soft_red, released[self.soft_red]))
        recs.extend(self._check_sustained_stop())
        recs.extend(self._check_sector_clearing())
        recs.extend(self._check_global_clearing())
        recs.extend(self._check_soft_red())
        recs.extend(self._check_recovery_advisory())
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
        if dtype == "IMPACT" and self.official_status in OFFICIAL_NEUTRAL:
            # under race control's own SC or VSC a lone impact signature (a car braking behind the
            # SC) raises no sector flag; it still counts if the car then stops (2021 Baku car 22)
            return []

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
        """True while a flagged car still counts as being in the sector. A crash site holds
        it even when parked (see the module docstring)."""
        car = self.car_state.get(drv)
        if car is None or car["in_pit"] or car["msector"] != msector:
            return False
        if self._stale(car):
            return False
        site = self.crash_sites.get(drv)
        if site is not None and site.msector == msector:
            return True
        return not self._parked(car)

    def _update_crash_sites(self) -> dict[str, int]:
        """Follow race control's track status for every crash site and release the sites
        that are over: moved away, no data, race control green again, or never reacted.
        Returns the released sites as {drv: msector}."""
        released: dict[str, int] = {}
        green = self.official_status == "1"
        for drv in list(self.crash_sites):
            site = self.crash_sites[drv]
            car = self.car_state.get(drv)
            if not green:
                site.reacted, site.green_since = True, None
            elif site.green_since is None:
                site.green_since = self.t
            moved = car is not None and math.hypot(car["anchor"][0] - site.x,
                                                   car["anchor"][1] - site.y) > CRASH_SITE_MOVE_M
            over = (site.reacted and site.green_since is not None
                    and self.t - site.green_since >= OFFICIAL_GREEN_S)
            ignored = not site.reacted and self.t - site.stop_t >= PARKED_CAR_S
            if car is None or self._stale(car) or car["in_pit"] or moved or over or ignored:
                if over:
                    self.green_released[site.msector] = self.t
                released[drv] = site.msector
                del self.crash_sites[drv]
        return released

    def _soft_red_candidate(self) -> tuple[str, float] | None:
        """The longest-standing crash site that qualifies for the soft red: race control has
        reacted, the car is still stopped, and it stopped RED_CRASH_SITE_S ago or more."""
        for drv, site in sorted(self.crash_sites.items(), key=lambda kv: (kv[1].stop_t, kv[0])):
            car = self.car_state.get(drv)
            if car is None or not site.reacted or car["speed"] > STOPPED_SPEED_KMH:
                continue
            if self.t - site.stop_t >= RED_CRASH_SITE_S:
                return drv, self.t - site.stop_t
        return None

    def laps_left(self) -> int | None:
        if self.race_laps is None or self.lap is None:
            return None
        return max(self.race_laps - self.lap, 0)

    def _live_cause_sector(self) -> int | None:
        """A cause sector of our track-wide flag that is still flagged and still held by a
        flagged car: the neutralisation is for an incident that is still there."""
        g = self.global_
        for s in sorted(g.cause_sectors):
            sec = self.sectors.get(s)
            if sec is not None and sec.flag != "CLEAR" and any(self._holds_sector(d, s) for d in sec.cause_drivers):
                return s
        return None

    def _check_soft_red(self) -> list[dict]:
        """Our red flag, kept apart from the hard flag (self.global_.flag) so it never blocks
        another incident's escalation. Late in the race: our VSC or SC is out for an incident
        that is still there and LATE_RED_LAPS laps or fewer are left, so the race would likely
        end behind the neutralisation. Race control's reds in the data come mostly from such
        late neutralisations (docs/lab/red_flag.md); a mid-race red needs barrier damage,
        gravel or debris that car data cannot see. RED_BY_TIME brings back the old red by
        time at a crash site (off: 13 of its 16 reds were not race control's)."""
        if self.soft_red is not None or self.global_.flag in ("CLEAR", "RED"):
            return []
        left = self.laps_left()
        if left is not None and LATE_RED_MIN_LAPS <= left <= LATE_RED_LAPS and self.global_.flag in ("VSC", "SC"):
            where = self._live_cause_sector()
            if where is not None:
                self.soft_red = LATE_RED_KEY
                g = self.global_.flag
                return [self._make_rec(self.t, where, "RED", LATE_RED_CONF,
                                       f"{g} out for a live incident with about {left} laps left: a red flag "
                                       f"lets the race restart and finish racing, not behind the {g}",
                                       _global_message("RED"), [])]
        if not RED_BY_TIME:
            return []
        cand = self._soft_red_candidate()
        if cand is None:
            return []
        drv, elapsed = cand
        self.soft_red = drv
        site = self.crash_sites[drv]
        return [self._make_rec(self.t, site.msector, "RED", RED_CRASH_SITE_CONF,
                               f"car {drv} still stopped at its crash site {elapsed:.0f} s after it stopped",
                               _global_message("RED"), [])]

    def _check_recovery_advisory(self) -> list[dict]:
        """A crashed car still at its crash site RED_CRASH_SITE_S after it stopped, once race
        control has reacted: the recovery is taking long. Only an advisory: the crash
        sector's own flag is re-sent unchanged with that reason (every client treats a
        same-level re-send as an update, never a new call). Race control decides whether
        that needs a red flag; car data cannot see barrier damage or gravel on the track."""
        if RED_BY_TIME:
            return []
        cand = self._soft_red_candidate()
        if cand is None or cand[0] in self.advised:
            return []
        drv, elapsed = cand
        self.advised.add(drv)
        site = self.crash_sites[drv]
        sec = self.sectors.get(site.msector)
        if sec is None or sec.flag == "CLEAR":
            return []
        return [self._make_rec(self.t, site.msector, sec.flag, sec.confidence,
                               f"car {drv} still at its crash site {elapsed:.0f} s after it stopped: "
                               "recovery taking long, race control may need a red flag (advisory, flag unchanged)",
                               _sector_message(sec.flag, site.msector), sec.cause_detection_ids)]

    def _end_soft_red(self, drv: str, msector: int) -> list[dict]:
        """The soft red's crash site was released. Another qualifying crash site takes the
        red over and nothing is sent (our flag stays RED). Otherwise, if our hard VSC or SC
        stays out (race control still neutralises, or a car still holds a cause sector), it
        is re-sent as the downgrade; if nothing keeps it, TRACK CLEAR follows within seconds."""
        self.soft_red = None
        heir = self._soft_red_candidate()
        if heir is not None:
            self.soft_red = heir[0]
            return []
        g = self.global_
        if g.flag not in ("VSC", "SC"):
            return []
        holding = sorted(s for s in g.cause_sectors if s in self.sectors and self.sectors[s].flag != "CLEAR"
                         and any(self._holds_sector(d, s) for d in self.sectors[s].cause_drivers))
        if not holding and self.official_status not in OFFICIAL_NEUTRAL:
            return []
        where = holding[0] if holding else (g.cause_sector if g.cause_sector is not None else msector)
        still = f"sector {where}" if holding else "race control's neutralisation"
        return [self._make_rec(self.t, where, g.flag, g.confidence,
                               f"red flag call for car {drv} ended, {g.flag} still out for {still} "
                               "(a downgrade, not a new call)",
                               _global_message(g.flag), [])]

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
        if sev >= RED_SEV and RED_FROM_MULTI:
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
            self.global_.held_for_race_control = False
        self.global_.cause_sectors.add(msector)
        self.global_.flag = target
        if target == "RED":
            self.soft_red = None          # a hard red supersedes the soft one
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
        moves = []   # (from sector, drv, stop): stopped cars that rolled on into another sector
        for msector, sec in self.sectors.items():
            for drv in sorted(sec.stops):
                stop = sec.stops[drv]
                car = self.car_state.get(drv)
                if car is None:
                    continue
                if not self._holds_sector(drv, msector):
                    if self._rolled_on(car, msector):
                        moves.append((msector, drv, stop))
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
                if impact and elapsed >= SC_STOPPED_HOLD_S and not stop.crash_site_done:
                    ax, ay = car["anchor"]
                    self.crash_sites[drv] = CrashSite(msector, stop.first_t, ax, ay)
                    stop.crash_site_done = True
                if impact and elapsed >= SC_STOPPED_HOLD_S:
                    calls.append((GLOBAL_RANK["SC"], "SC", msector, drv, stop, elapsed, impact))
                elif not impact and elapsed >= VSC_STOPPED_HOLD_S:
                    calls.append((GLOBAL_RANK["VSC"], "VSC", msector, drv, stop, elapsed, impact))
        recs: list[dict] = []
        for msector, drv, stop in moves:
            recs.extend(self._follow_stop(msector, drv, stop))
        if not calls:
            return recs

        rank, target, msector, drv, stop, elapsed, impact = max(calls, key=lambda c: c[0])
        if rank > GLOBAL_RANK[self.global_.flag]:
            conf = min(MAX_CONF, stop.severity)
            if self.risk.get(drv, {}).get("risk_30s", 0.0) >= RISK_SC_SUPPORT:
                conf = min(MAX_CONF, conf + RISK_CONF_BOOST)
            if impact:
                reason = f"car {drv} stopped for {elapsed:.1f} s after an impact"
            else:
                reason = f"car {drv} stopped for {elapsed:.1f} s, no impact detected"
            recs.extend(self._escalate_global(self.t, msector, target, conf, reason, None))
        for c in calls:
            if c[0] <= GLOBAL_RANK[self.global_.flag]:
                self.global_.cause_sectors.add(c[2])
        return recs

    def _rolled_on(self, car: dict, msector: int) -> bool:
        """A stopped car that left its sector still slow, on track and with fresh data: it
        rolled on (a car losing power often crawls over a sector boundary before it stops),
        so it is still the hazard. Driving off at speed, pitting or retiring is not."""
        return (car["msector"] != msector and not car["in_pit"] and not self._stale(car)
                and not self._parked(car) and car["speed"] <= FOLLOW_STOP_KMH)

    def _follow_stop(self, old: int, drv: str, stop: Stop) -> list[dict]:
        """Move a stopped car's stop, and its sector's flag, to the sector it rolled into.
        The hold keeps running, so the VSC or SC comes as if it had stopped there; the old
        sector clears as usual once no flagged car holds it."""
        new = self.car_state[drv]["msector"]
        src = self.sectors[old]
        dst = self.sectors.setdefault(new, SectorState())
        dst.stops.setdefault(drv, stop)
        dst.cause_drivers.add(drv)
        dst.empty_since_t = None
        if SECTOR_RANK[src.flag] <= SECTOR_RANK[dst.flag]:
            return []
        dst.flag = src.flag
        dst.since_t = self.t
        dst.confidence = src.confidence
        dst.reason = f"car {drv} stopped, rolled on from sector {old}"
        dst.cause_detection_ids = list(src.cause_detection_ids)
        return [self._make_rec(self.t, new, dst.flag, dst.confidence, dst.reason,
                               _sector_message(dst.flag, new), dst.cause_detection_ids)]

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

            released_t = self.green_released.pop(msector, None)
            by_race_control = (released_t is not None
                               and self.t - released_t <= SECTOR_CLEAR_AFTER_S + GREEN_REASON_SLACK_S)
            if by_race_control and msector in self.global_.cause_sectors:
                self.global_.held_for_race_control = True
            reason = ("crash site held until race control's track status was green" if by_race_control
                      else "no flagged car remains in sector")
            recs.append(
                self._make_rec(
                    self.t, msector, "CLEAR", 1.0, reason,
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
        if self.official_status in OFFICIAL_NEUTRAL:
            # every cause is gone, but race control's own SC, VSC or red is still out
            self.global_.empty_since_t = None
            self.global_.held_for_race_control = True
            return []
        if self.global_.empty_since_t is None:
            self.global_.empty_since_t = self.t
            return []
        if self.t - self.global_.empty_since_t < GLOBAL_CLEAR_AFTER_S:
            return []

        reason = ("incident cleared, held until race control's track status was green"
                  if self.global_.held_for_race_control else "incident cleared")
        rec = self._make_rec(
            self.t, cause_sector, "CLEAR", 1.0, reason,
            _global_message("CLEAR"), [],
        )
        self.global_.held_for_race_control = False
        self.soft_red = None
        self.global_.flag = "CLEAR"
        self.global_.since_t = self.t
        self.global_.cause_sectors = set()
        self.global_.confidence = 0.0
        self.global_.reason = ""
        self.global_.empty_since_t = None
        return [rec]

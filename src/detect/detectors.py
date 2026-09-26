"""Incident detectors (Section 6.3): STOPPED, IMPACT, SPIN, DROPOUT, MULTI, ANOMALY.

One replay-engine Processor. It sees one tick at a time and keeps a small state
per car, so it is causal by construction. Thresholds live in Config and are tuned
on training races only (src.eval.run).

Context gates:
- suspended (red flag, or most of the field in the pit lane): nothing fires.
- grid phase (most of the field below 30 km/h, or the whole field crawling at
  under 30% of reference speed: standing start, forming up for a restart) and
  grid_hold_s after it: no STOPPED, IMPACT, SPIN or ANOMALY.
- lap_ratio (speed vs the same car at the same point one lap earlier) confirms
  STOPPED and IMPACT: after normal braking a car matches its previous lap, after a
  crash it is far below it.
- speed is always judged relative to the field: a car's speed ratio (speed /
  reference) is divided by the field's median ratio. Under SC / VSC, in the slow
  laps after a red flag or behind a formation, every car is slow and nobody stands
  out, but a car stopped among them still does. ANOMALY only runs in green or
  yellow running (track status 1, 2), which is what it was trained on.

Run: python -m src.detect.detectors 2023_Australian   (prints every detection)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.ingest.sectors import sector_offset

NEUTRAL = {"4", "6", "7"}
COLS = ["drv", "speed", "rel_ratio", "lap_ratio", "dspeed_1s", "dspeed_min_2s", "brake", "lat_off", "x",
        "msector", "in_pit",
        "heading_err", "ref_speed", "track_status", "suspended", "field_slow", "field_ratio"]
CRASH_TYPES = {"STOPPED", "IMPACT", "SPIN"}


@dataclass
class Config:
    stop_ratio: float = 0.30          # speed / reference below this counts as stopped or crawling
    stop_sustain_s: float = 1.5
    recover_ratio: float = 0.60
    recover_s: float = 5.0
    impact_drop_kmh: float = 80.0     # speed lost within 1 s
    impact_end_ratio: float = 0.40    # while braking, an impact must end this far below reference
    impact_confirm_s: float = 0.5     # still slow this long after, so data glitches do not fire
    impact_hard_kmh: float = 180.0    # harder than 99.99% of braking in green running: fires even while braking
    impact_hard_confirm_ratio: float = 0.85
    spin_heading_deg: float = 35.0    # path this far off the reference path (green running p99.99: 11 deg) ...
    spin_min_kmh: float = 25.0
    spin_loss_kmh: float = 30.0       # ... while losing this much speed within 1 s
    spin_lat_m: float = 10.0          # or this far off the racing line (99.999% of green running: < 7.5 m)
    spin_sustain_s: float = 0.5
    slowdown_lap_ratio: float | None = 0.6   # or a sudden slowdown: below this share of last lap's speed ...
    slowdown_loss_kmh: float = 40.0          # ... after losing this much speed within 2 s
    slowdown_sustain_s: float = 1.0
    slowdown_rel_ratio: float = 0.6          # ... and slow compared with the field right now
    dropout_s: float = 2.0            # no data for this long (on top of the 1 s staleness in ingest)
    dropout_min_kmh: float = 30.0     # car was moving when the data stopped
    field_live_min: float = 0.6       # DROPOUT only while most of the field still has data
    multi_window_s: float = 5.0
    cooldown_s: float = 20.0          # same car, same type
    grid_field_slow: float = 0.5
    grid_field_ratio: float = 0.3     # whole field this slow = procession / forming up
    grid_hold_s: float = 15.0
    stop_lap_ratio: float = 0.5       # STOPPED also needs speed below this share of last lap's
    impact_lap_ratio: float = 0.6     # IMPACT is confirmed only below this share of last lap's
    anomaly_sustain: int = 2          # ticks above threshold
    on_line_m: float = 8.0


@dataclass
class CarState:
    slow_since: float | None = None
    stopped: bool = False
    recover_since: float | None = None
    impact_cand: tuple[float, float, float, bool] | None = None   # (t, drop, speed before, hard)
    spin_since: float | None = None
    slowdown_since: float | None = None
    last_seen: float | None = None
    last_speed: float = 0.0
    last_in_pit: bool = True
    last_msector: int = 0
    dropped: bool = False
    anomaly_run: int = 0
    last_emit: dict[str, float] = field(default_factory=dict)


class DetectorSuite:
    def __init__(self, cfg: Config | None = None, anomaly_threshold: float | None = None,
                 n_sectors: int | None = None) -> None:
        self.cfg = cfg or Config()
        self.anomaly_threshold = anomaly_threshold
        self.n_sectors = n_sectors or 1000    # sector numbers wrap at the line when known
        self.next_id = 1                      # never reset, so ids stay unique across seeks
        self.reset()

    def reset(self) -> None:
        self.cars: dict[str, CarState] = {}
        self.grid_until = -np.inf
        self.recent: list[dict] = []          # crash-type detections in the last multi window
        self.multi_recent: list[dict] = []    # MULTI detections within the cooldown

    # ---- helpers
    def car(self, drv: str) -> CarState:
        return self.cars.setdefault(drv, CarState())

    def can_emit(self, c: CarState, kind: str, t: float) -> bool:
        return t - c.last_emit.get(kind, -np.inf) >= self.cfg.cooldown_s

    def det(self, t: float, drivers: list[str], msector: int, kind: str, severity: float,
            evidence: str) -> dict:
        d = {"id": f"det-{self.next_id:06d}", "t": round(float(t), 2), "drivers": drivers,
             "msector": int(msector), "type": kind, "severity": round(float(np.clip(severity, 0, 1)), 2),
             "evidence": evidence}
        self.next_id += 1
        return d

    # ---- main entry
    def on_tick(self, t: float, frame: pd.DataFrame, tick: dict) -> list[dict]:
        cols = {c: frame[c].to_numpy() for c in COLS if c in frame}
        score = frame["anomaly_score"].to_numpy() if "anomaly_score" in frame else None
        if bool(cols["suspended"][0]):
            for c in self.cars.values():
                c.slow_since = c.spin_since = c.impact_cand = None
            return []
        status = str(cols["track_status"][0])
        if (float(cols["field_slow"][0]) >= self.cfg.grid_field_slow
                or float(cols["field_ratio"][0]) < self.cfg.grid_field_ratio):
            self.grid_until = t + self.cfg.grid_hold_s
        grid = t < self.grid_until
        neutral = status in NEUTRAL
        speed = cols["speed"].astype(float)
        live = np.isfinite(speed) & np.isfinite(cols["x"].astype(float))
        live_share = float(np.isfinite(speed).mean())
        out: list[dict] = []
        for i in range(len(speed)):
            out.extend(self.car_tick(t, i, cols, score, live[i], grid, neutral, status, live_share))
        out.extend(self.multi(t, out))
        return [{"kind": "detection", "data": d} for d in out]

    def car_tick(self, t: float, i: int, cols: dict, score, live: bool, grid: bool, neutral: bool,
                 status: str, live_share: float) -> list[dict]:
        cfg, drv = self.cfg, str(cols["drv"][i])
        c = self.car(drv)
        speed = float(cols["speed"][i])
        in_pit = bool(cols["in_pit"][i])
        msector = int(cols["msector"][i])
        out = self.dropout(t, c, drv, speed, in_pit, msector, live_share, grid)
        if not live:
            c.slow_since = c.spin_since = None
            return out
        if in_pit:
            c.stopped, c.slow_since, c.spin_since, c.impact_cand, c.anomaly_run = False, None, None, None, 0
            c.slowdown_since = None
            return out
        ratio = float(cols["rel_ratio"][i])            # speed ratio relative to the field (ingest)
        lap = float(cols["lap_ratio"][i])              # speed vs own previous lap at this point
        lat = float(cols["lat_off"][i])
        if not grid:
            out += self.stopped(t, c, drv, speed, ratio, lap, lat, msector, float(cols["ref_speed"][i]))
            out += self.impact(t, c, drv, speed, ratio, lap, msector, float(cols["dspeed_1s"][i]),
                               bool(cols["brake"][i] > 0))
            out += self.spin(t, c, drv, speed, ratio, lat, msector, float(cols["heading_err"][i])
                             if "heading_err" in cols else np.nan, float(cols["dspeed_1s"][i]))
            out += self.slowdown(t, c, drv, speed, ratio, lap, msector, float(cols["dspeed_min_2s"][i]))
            if score is not None and status in ("1", "2") and self.anomaly_threshold is not None:
                out += self.anomaly(t, c, drv, msector, float(score[i]))
        return out

    # ---- detectors
    def stopped(self, t, c, drv, speed, ratio, lap, lat, msector, ref) -> list[dict]:
        cfg = self.cfg
        below_last_lap = not np.isfinite(lap) or lap < cfg.stop_lap_ratio
        if ratio < cfg.stop_ratio and below_last_lap:
            c.slow_since = t if c.slow_since is None else c.slow_since
        else:
            c.slow_since = None
        if c.stopped:
            c.recover_since = (c.recover_since or t) if ratio > cfg.recover_ratio else None
            if c.recover_since is not None and t - c.recover_since >= cfg.recover_s:
                c.stopped, c.recover_since = False, None
            return []
        if c.slow_since is None or t - c.slow_since < cfg.stop_sustain_s:
            return []
        c.stopped = True
        c.last_emit["STOPPED"] = t
        on_line = abs(lat) < cfg.on_line_m
        recent_hit = t - c.last_emit.get("IMPACT", -np.inf) < 10
        sev = 0.45 + 0.25 * on_line + 0.15 * (speed < 5) + 0.15 * recent_hit
        where = "on the racing line" if on_line else f"{abs(lat):.0f} m off line"
        return [self.det(t, [drv], msector, "STOPPED", sev,
                         f"speed {speed:.0f} km/h vs ref {ref:.0f} km/h, {where}")]

    def impact(self, t, c, drv, speed, ratio, lap, msector, dspeed, brake) -> list[dict]:
        cfg = self.cfg
        if c.impact_cand is not None:
            t0, drop, before, hard = c.impact_cand
            if t - t0 < cfg.impact_confirm_s:
                return []
            c.impact_cand = None
            still_slow = ratio < (cfg.impact_hard_confirm_ratio if hard else 0.6)
            if np.isfinite(lap):
                still_slow = still_slow and lap < cfg.impact_lap_ratio
            if still_slow and self.can_emit(c, "IMPACT", t):
                c.last_emit["IMPACT"] = t
                sev = 0.5 + min(drop, 250) / 500 + 0.2 * (ratio < cfg.stop_ratio)
                why = "harder than braking" if hard else ("with brakes on" if brake else "without braking")
                return [self.det(t, [drv], msector, "IMPACT", sev,
                                 f"lost {drop:.0f} km/h in 1 s ({before:.0f} to {speed:.0f} km/h), {why}")]
            return []
        drop = -dspeed
        if np.isfinite(drop) and ((drop >= cfg.impact_drop_kmh and (not brake or ratio < cfg.impact_end_ratio))
                                  or drop >= cfg.impact_hard_kmh):
            c.impact_cand = (t, drop, speed + drop, drop >= cfg.impact_hard_kmh)
        return []

    def spin(self, t, c, drv, speed, ratio, lat, msector, heading, dspeed) -> list[dict]:
        """SPIN / OFF: leaving the track direction while losing speed, or far off the
        racing line. Position data has no yaw, so a spin shows up as the car's path
        leaving the track direction and the speed dropping, not as rotation."""
        cfg = self.cfg
        turned = (np.isfinite(heading) and heading > cfg.spin_heading_deg and speed > cfg.spin_min_kmh
                  and np.isfinite(dspeed) and -dspeed >= cfg.spin_loss_kmh)
        off = abs(lat) > cfg.spin_lat_m
        if not (turned or off):
            c.spin_since = None
            return []
        c.spin_since = t if c.spin_since is None else c.spin_since
        if t - c.spin_since < cfg.spin_sustain_s or not self.can_emit(c, "SPIN", t):
            return []
        c.last_emit["SPIN"] = t
        sev = 0.4 + 0.2 * (abs(lat) < cfg.on_line_m) + 0.2 * (ratio < cfg.stop_ratio) + 0.1 * turned
        why = (f"travelling {heading:.0f} deg off the track direction, lost {-dspeed:.0f} km/h in 1 s"
               if turned else f"{abs(lat):.0f} m off the racing line")
        return [self.det(t, [drv], msector, "SPIN", sev, why)]

    def slowdown(self, t, c, drv, speed, ratio, lap, msector, dspeed_min) -> list[dict]:
        """A car suddenly far slower than itself at the same point last lap and slower
        than the field (off, spin, damage) that has not stopped. Reported as SPIN
        (the brief's SPIN / OFF)."""
        cfg = self.cfg
        if cfg.slowdown_lap_ratio is None:
            return []
        hit = (np.isfinite(lap) and lap < cfg.slowdown_lap_ratio and ratio < cfg.slowdown_rel_ratio
               and (c.slowdown_since is not None or (np.isfinite(dspeed_min) and -dspeed_min >= cfg.slowdown_loss_kmh)))
        if not hit:
            c.slowdown_since = None
            return []
        c.slowdown_since = t if c.slowdown_since is None else c.slowdown_since
        if t - c.slowdown_since < cfg.slowdown_sustain_s or not self.can_emit(c, "SPIN", t) or c.stopped:
            return []
        c.last_emit["SPIN"] = t
        return [self.det(t, [drv], msector, "SPIN", 0.4 + 0.3 * (1 - lap),
                         f"sudden slowdown: {speed:.0f} km/h, {lap:.0%} of its speed here last lap")]

    def dropout(self, t, c, drv, speed, in_pit, msector, live_share, grid) -> list[dict]:
        cfg = self.cfg
        if np.isfinite(speed):
            c.last_seen, c.last_speed, c.last_in_pit, c.last_msector, c.dropped = t, speed, in_pit, msector, False
            return []
        if c.dropped or c.last_seen is None or c.last_in_pit or grid:
            return []
        if t - c.last_seen < cfg.dropout_s or c.last_speed < cfg.dropout_min_kmh or live_share < cfg.field_live_min:
            return []
        c.dropped = True
        c.last_emit["DROPOUT"] = t
        return [self.det(t, [drv], c.last_msector, "DROPOUT", 0.5,
                         f"no data for {t - c.last_seen + 1:.1f} s, last seen at {c.last_speed:.0f} km/h")]

    def anomaly(self, t, c, drv, msector, score) -> list[dict]:
        if not np.isfinite(score) or score < self.anomaly_threshold:
            c.anomaly_run = 0
            return []
        c.anomaly_run += 1
        if c.anomaly_run < self.cfg.anomaly_sustain or not self.can_emit(c, "ANOMALY", t):
            return []
        c.last_emit["ANOMALY"] = t
        sev = 0.3 + min(0.5, (score - self.anomaly_threshold) * 5)
        return [self.det(t, [drv], msector, "ANOMALY", sev,
                         f"anomaly score {score:.2f} (threshold {self.anomaly_threshold:.2f})")]

    def near(self, a: int, b: int) -> bool:
        return abs(sector_offset(a, b, self.n_sectors)) <= 1

    def multi(self, t: float, new: list[dict]) -> list[dict]:
        """2+ cars with crash-type detections in the same or adjacent sectors within
        multi_window_s. One MULTI per group: skipped when a recent MULTI nearby
        already covers one of the same cars."""
        cfg = self.cfg
        self.recent = [d for d in self.recent if t - d["t"] <= cfg.multi_window_s]
        self.multi_recent = [m for m in self.multi_recent if t - m["t"] < cfg.cooldown_s]
        out = []
        for d in new:
            if d["type"] not in CRASH_TYPES:
                continue
            others = [r for r in self.recent if r["drivers"][0] != d["drivers"][0] and self.near(r["msector"], d["msector"])]
            drivers = sorted({d["drivers"][0], *(r["drivers"][0] for r in others)})
            covered = any(self.near(m["msector"], d["msector"]) and set(m["drivers"]) & set(drivers)
                          for m in self.multi_recent)
            if others and not covered:
                sev = min(1.0, max([d["severity"]] + [r["severity"] for r in others]) + 0.15)
                m = self.det(t, drivers, d["msector"], "MULTI", sev,
                             f"cars {', '.join(drivers)} in sectors "
                             f"{min(r['msector'] for r in others + [d])}-{max(r['msector'] for r in others + [d])}"
                             f" within {cfg.multi_window_s:.0f} s")
                self.multi_recent.append(m)
                out.append(m)
            self.recent.append(d)
        return out


class NaiveThreshold:
    """Baseline to beat (Section 6.3): any car below a fixed speed outside the pit lane.
    Gets the same suspension and grid gating as the real detectors, so the comparison
    is about the signal, not the plumbing."""

    def __init__(self, kmh: float = 50.0, sustain_s: float = 1.0) -> None:
        self.kmh, self.sustain_s, self.next_id = kmh, sustain_s, 1
        self.reset()

    def reset(self) -> None:
        self.slow_since: dict[str, float] = {}
        self.latched: set[str] = set()
        self.grid_until = -np.inf

    def on_tick(self, t: float, frame: pd.DataFrame, tick: dict) -> list[dict]:
        if bool(frame["suspended"].iat[0]):
            return []
        if float(frame["field_slow"].iat[0]) >= 0.5:
            self.grid_until = t + 15.0
        out = []
        for drv, speed, in_pit, ms in zip(frame["drv"], frame["speed"], frame["in_pit"], frame["msector"]):
            slow = np.isfinite(speed) and speed < self.kmh and not in_pit and t >= self.grid_until
            if not slow:
                self.slow_since.pop(drv, None)
                if np.isfinite(speed) and speed > self.kmh + 30:
                    self.latched.discard(drv)
                continue
            t0 = self.slow_since.setdefault(drv, t)
            if t - t0 >= self.sustain_s and drv not in self.latched:
                self.latched.add(drv)
                out.append({"kind": "detection", "data": {
                    "id": f"base-{self.next_id:06d}", "t": round(float(t), 2), "drivers": [str(drv)],
                    "msector": int(ms), "type": "STOPPED", "severity": 0.5,
                    "evidence": f"speed {speed:.0f} km/h < {self.kmh:.0f} km/h"}})
                self.next_id += 1
        return out


def main() -> None:
    from src.replay.engine import Engine, RaceData
    race = RaceData.load(sys.argv[1] if len(sys.argv) > 1 else "2023_Australian")
    eng = Engine(race, [DetectorSuite(n_sectors=len(race.track.get("msectors", [])) or None)])
    for env in eng.advance(eng.t_end):
        if env["kind"] == "detection":
            d = env["data"]
            print(f"{d['t']:8.1f} {d['type']:<8} sector {d['msector']:>2} cars {','.join(d['drivers']):<8} "
                  f"sev {d['severity']:.2f}  {d['evidence']}")


if __name__ == "__main__":
    main()

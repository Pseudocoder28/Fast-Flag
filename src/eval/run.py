"""Run detectors over built training races and score them against official incidents.

Metrics per race and in total (definitions in src.eval.incidents):
- recall: matched incidents / incidents, where matched = an alert in the match
  window, or a STOPPED alert we were still holding in a matching sector ("held")
- recall_window: only alerts in the match window
- lead time: t_official - t_ours for window matches (positive = we were earlier)
- false alarms per race hour: our alerts that overlap no incident, where an alert is
  a car's first detection after 60 s without one (MULTI: per sector), and race hours
  exclude time suspended under a red flag
- the same for the naive speed-threshold baseline

Run: python -m src.eval.run                       (all built training races, detectors vs baseline)
     python -m src.eval.run 2023_Australian 2024_Canadian --verbose
Writes docs/charts/detect_eval.csv and docs/charts/detect_eval.md.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from src.detect.anomaly import attach_scores, train
from src.detect.detectors import Config, DetectorSuite, NaiveThreshold
from src.eval.incidents import (MATCH_AFTER_S, MATCH_BEFORE_S, Incident, alert_matches, build_incidents,
                                race_files, suspended_times)
from src.ingest.holdout import assert_not_holdout
from src.ingest.sectors import sector_matches
from src.replay.engine import Engine, RaceData, available_races

CHARTS = Path("docs/charts")
ALERT_GAP_S = 60.0
FA_AFTER_S = 60.0         # an alert this long after an incident's last message still belongs to it
MOVING_KMH = 60.0         # a held STOPPED alert ends when the car moves again ...
MAX_HOLD_S = 900.0        # ... enters the pit lane, loses data, or after 15 minutes


@dataclass
class RaceResult:
    race: str
    system: str
    incidents: int
    matched: int
    held: int
    lead_times: list[float]
    alerts: int
    false_alarms: int
    hours: float
    skipped: int
    rows: list[dict]           # per-incident detail
    alert_rows: list[dict]     # per-alert detail (type, true / false alarm)


def alerts_from(dets: list[dict]) -> list[dict]:
    """Collapse detections into alerts: a car's first detection after ALERT_GAP_S of quiet."""
    last: dict[str, float] = {}
    out = []
    for d in sorted(dets, key=lambda d: d["t"]):
        key = f"sector-{d['msector']}" if d["type"] == "MULTI" else d["drivers"][0]
        if d["t"] - last.get(key, -np.inf) >= ALERT_GAP_S:
            out.append(d)
        last[key] = d["t"]
    return out


def belongs(alert: dict, inc: Incident, n: int) -> bool:
    if not inc.t - MATCH_BEFORE_S <= alert["t"] <= inc.t_last + FA_AFTER_S:
        return False
    return inc.track_wide_only or any(sector_matches(alert["msector"], s, n) for s in inc.sectors)


def held_intervals(race: RaceData, dets: list[dict]) -> list[tuple[float, float, int]]:
    """(start, end, msector) while each STOPPED car stays stopped on track."""
    cols = race.frame[["t", "drv", "speed", "in_pit"]]
    by_drv = {drv: g for drv, g in cols.groupby("drv")}
    out = []
    for d in dets:
        g = by_drv.get(d["drivers"][0]) if d["type"] == "STOPPED" else None
        if g is None:
            continue
        t, v = g["t"].to_numpy(), g["speed"].to_numpy(float)
        ended = (t > d["t"]) & ((v > MOVING_KMH) | g["in_pit"].to_numpy() | ~np.isfinite(v))
        end = float(t[ended.argmax()]) if ended.any() else float(t[-1])
        out.append((d["t"], min(end, d["t"] + MAX_HOLD_S), d["msector"]))
    return out


def score(race: RaceData, system: str, dets: list[dict]) -> RaceResult:
    official, meta = race_files(race.race)
    n = meta["n_msectors"]
    susp = suspended_times(race.frame)
    incidents, skipped = build_incidents(official, meta, susp, n)
    dets = sorted(dets, key=lambda d: d["t"])
    held = held_intervals(race, dets)
    rows, leads = [], []
    for inc in incidents:
        hit = next((d for d in dets if alert_matches(inc, d["t"], d["msector"], n)), None)
        lead = round(inc.t - hit["t"], 2) if hit else None
        if hit:
            leads.append(lead)
        was_held = hit is None and any(
            a <= inc.t <= b and (inc.track_wide_only or any(sector_matches(ms, s, n) for s in inc.sectors))
            for a, b, ms in held)
        rows.append({"race": race.race, "system": system, "t_official": inc.t, "flag": inc.top_flag,
                     "sectors": sorted(inc.sectors), "matched": hit is not None or was_held,
                     "held": was_held, "lead_s": lead,
                     "our_type": hit["type"] if hit else None, "our_evidence": hit["evidence"] if hit else None})
    alerts = alerts_from(dets)
    alert_rows = [{"race": race.race, "system": system, "t": a["t"], "type": a["type"], "msector": a["msector"],
                   "drivers": ",".join(a["drivers"]), "evidence": a["evidence"],
                   "false_alarm": not any(belongs(a, i, n) for i in incidents)} for a in alerts]
    fa = sum(r["false_alarm"] for r in alert_rows)
    dt = np.diff(susp.index.to_numpy(), append=susp.index[-1])
    hours = float(dt[~susp.to_numpy()].sum()) / 3600
    return RaceResult(race.race, system, len(incidents), sum(r["matched"] for r in rows),
                      sum(r["held"] for r in rows), leads, len(alerts), fa, hours, skipped, rows, alert_rows)


def run_race(rid: str, cfg: Config | None = None, anomaly: bool = True,
             anomaly_q: float | None = None) -> list[RaceResult]:
    """Score one race. The ANOMALY model is trained leave-one-race-out: on every
    other training race, never on the race being scored (and never on the holdout)."""
    assert_not_holdout(rid=rid)
    race = RaceData.load(rid)
    n = len(race.track.get("msectors", [])) or None
    threshold = None
    if anomaly:
        model = train([r for r in available_races() if r != rid])
        attach_scores(race.frame, model)
        threshold = model.threshold_at(anomaly_q) if anomaly_q else model.threshold
    out = []
    suite = DetectorSuite(cfg, anomaly_threshold=threshold, n_sectors=n)
    for system, proc in (("detectors", suite), ("baseline", NaiveThreshold())):
        eng = Engine(race, [proc])
        dets = [e["data"] for e in eng.advance(eng.t_end) if e["kind"] == "detection"]
        out.append(score(race, system, dets))
    return out


def summarise(results: list[RaceResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({"race": r.race, "system": r.system, "incidents": r.incidents, "matched": r.matched,
                     "held": r.held, "recall": r.matched / r.incidents if r.incidents else np.nan,
                     "recall_window": (r.matched - r.held) / r.incidents if r.incidents else np.nan,
                     "median_lead_s": float(np.median(r.lead_times)) if r.lead_times else np.nan,
                     "alerts": r.alerts, "false_alarms": r.false_alarms,
                     "fa_per_hour": r.false_alarms / r.hours if r.hours else np.nan, "hours": r.hours})
    return pd.DataFrame(rows)


def totals(df: pd.DataFrame, results: list[RaceResult]) -> pd.DataFrame:
    out = []
    for system, g in df.groupby("system"):
        leads = [x for r in results if r.system == system for x in r.lead_times]
        out.append({"system": system, "races": len(g), "incidents": int(g["incidents"].sum()),
                    "matched": int(g["matched"].sum()), "held": int(g["held"].sum()),
                    "recall": g["matched"].sum() / g["incidents"].sum(),
                    "recall_window": (g["matched"].sum() - g["held"].sum()) / g["incidents"].sum(),
                    "median_lead_s": float(np.median(leads)) if leads else np.nan,
                    "earlier_than_official": float(np.mean([x > 0 for x in leads])) if leads else np.nan,
                    "false_alarms": int(g["false_alarms"].sum()),
                    "fa_per_hour": g["false_alarms"].sum() / g["hours"].sum()})
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Score detectors on training races")
    p.add_argument("races", nargs="*")
    p.add_argument("--verbose", action="store_true", help="print every incident")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-anomaly", action="store_true", help="skip the ANOMALY model (faster)")
    p.add_argument("--anomaly-q", type=float, help="alert threshold quantile, e.g. 0.9999")
    a = p.parse_args()
    races = a.races or available_races()
    job = partial(run_race, anomaly=not a.no_anomaly, anomaly_q=a.anomaly_q)
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        results = [r for rs in ex.map(job, races) for r in rs]
    df = summarise(results)
    tot = totals(df, results)
    pd.set_option("display.width", 200)
    print(df.pivot(index="race", columns="system", values=["recall", "median_lead_s", "fa_per_hour"]).round(2))
    print()
    print(tot.round(3).to_string(index=False))
    al = pd.DataFrame([row for r in results for row in r.alert_rows if r.system == "detectors"])
    if len(al):
        print("\ndetector alerts by type:")
        print(al.groupby("type")["false_alarm"].agg(alerts="size", false_alarms="sum").to_string())
    if a.verbose:
        detail = pd.DataFrame([row for r in results if r.system == "detectors" for row in r.rows])
        print(detail.to_string(index=False))
    if not a.no_save:
        CHARTS.mkdir(parents=True, exist_ok=True)
        df.to_csv(CHARTS / "detect_eval.csv", index=False, encoding="utf-8")
        pd.DataFrame([row for r in results for row in r.rows]).to_csv(CHARTS / "detect_incidents.csv", index=False, encoding="utf-8")
        pd.DataFrame([row for r in results for row in r.alert_rows]).to_csv(CHARTS / "detect_alerts.csv", index=False, encoding="utf-8")
        (CHARTS / "detect_eval.md").write_text(
            "# Detection eval (training races)\n\nReplay of historical FastF1 data. Matching and metric "
            "definitions: src/eval/incidents.py and src/eval/run.py.\n\n" + tot.round(3).to_markdown(index=False)
            + "\n\n" + df.round(3).to_markdown(index=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Delay-cost curve: how many cars pass a crash site at racing speed for every second
a flag is delayed (training races plus the 2021 Azerbaijan crashes, never a holdout).

For each official incident with an identifiable onset (the onset rule and the
onset-car selection are imported from src.eval.onset and src.eval.latency_by_type,
not copied) the curve counts, for every delay d from 0 to 60 s in 0.5 s steps, the
cars that passed the crash location (where the onset car came to rest, src.eval.case_study)
between the onset and onset + d at 80% or more of their own normal speed at that point
(the median of their previous 3 clean laps). Vertical markers: our first alert, our
escalation recommendation (VSC, SC or RED from the race control engine), the official
yellow and the official escalation.

Counterfactual: no model of driver reactions. The cars are counted as they actually
drove; a flag shown at delay d would have changed what they did after it.

Run: python -m src.lab.delay_cost                   (all training races + 2021_Azerbaijan)
     python -m src.lab.delay_cost 2023_Australian    (one race)
     python -m src.lab.delay_cost --replot           (redraw the PNGs from docs/lab/delay_cost.json)
Writes docs/lab/delay_cost.json, docs/lab/delay_cost.md and one PNG per incident.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.detect.anomaly import attach_scores, train as train_anomaly  # noqa: E402
from src.detect.detectors import DetectorSuite  # noqa: E402
from src.detect.pipeline import load_config  # noqa: E402
from src.eval.case_study import COUNTERFACTUAL, OUT_OF_SAMPLE, driver_names, passes, recommendations, rest_position  # noqa: E402
from src.eval.incidents import build_incidents, race_files, suspended_times  # noqa: E402
from src.eval.latency_by_type import ALERT_AFTER_S, ALERT_BEFORE_S, ONSET_AFTER_S, ONSET_BEFORE_S, in_sectors  # noqa: E402
from src.eval.onset import RULE, add_own_ratio, onsets  # noqa: E402
from src.ingest.holdout import assert_not_holdout  # noqa: E402
from src.replay.engine import Engine, RaceData, available_races, race_dir  # noqa: E402

OUT_DIR = Path("docs/lab")
CASE_RACES = ["2021_Azerbaijan"]           # replayable case studies, never listed as training races
ESCALATIONS = ("VSC", "SC", "RED")
SECTOR_FLAGS = ("YELLOW", "DOUBLE_YELLOW")
RACING_SHARE = 0.8                          # racing speed: at least this share of the car's own normal speed
MAX_DELAY_S = 60.0
STEP_S = 0.5
TRACE_BEFORE_S, TRACE_AFTER_S = 15.0, 45.0      # the onset car's speed trace kept for the cards
MAX_LISTED = 8                                  # detections and recommendations kept per incident
NOTE = "Replay of historical FastF1 data. Counterfactual: no model of driver reactions."
DPI = 110

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GOLD = "#2a78d6", "#eb6834", "#b8960c"
MARKER_STYLE = {"our first alert": (ORANGE, "-"), "our escalation": (BLUE, "-"),
                "official yellow": (GOLD, "--"), "official escalation": (INK, "--")}


# ---- pipeline: detections and recommendations for one race

def run_engine(race: RaceData) -> tuple[list[dict], list[dict]]:
    """Production detector settings with an ANOMALY model trained on the other
    training races (the same setup as src.eval.latency_by_type.detections_loro), then
    the race control engine on the same envelopes."""
    others = [r for r in available_races() if r != race.race]
    model = train_anomaly(others)
    attach_scores(race.frame, model)
    n = len(race.track.get("msectors", [])) or None
    eng = Engine(race, [DetectorSuite(load_config(), anomaly_threshold=model.threshold, n_sectors=n)])
    envs = eng.advance(eng.t_end)
    dets = sorted((e["data"] for e in envs if e["kind"] == "detection"), key=lambda d: d["t"])
    recs, _ = recommendations(envs)
    return dets, recs or []


# ---- the curve

def delays(max_delay_s: float = MAX_DELAY_S, step_s: float = STEP_S) -> list[float]:
    return [round(float(d), 2) for d in np.arange(0.0, max_delay_s + step_s / 2, step_s)]


def curve(pass_list: list[dict], share: float = RACING_SHARE, max_delay_s: float = MAX_DELAY_S,
          step_s: float = STEP_S) -> list[int]:
    """Cars that passed at racing speed within d seconds of the onset, for each delay d.
    A pass with no own reference (no clean lap yet) cannot be judged and is not counted."""
    racing = sorted(p["t_after_onset"] for p in pass_list
                    if p["pct_of_own_normal"] is not None and p["pct_of_own_normal"] >= 100 * share)
    return [int(np.searchsorted(racing, d, side="right")) for d in delays(max_delay_s, step_s)]


def cars_by(pass_list: list[dict], t_after_onset: float | None, share: float = RACING_SHARE) -> int | None:
    if t_after_onset is None:
        return None
    return sum(1 for p in pass_list if p["pct_of_own_normal"] is not None
               and p["pct_of_own_normal"] >= 100 * share and p["t_after_onset"] <= t_after_onset)


def summary_line(inc: dict) -> str:
    """'each second of delay here averaged X cars', plus where the real flags fell."""
    m, n60 = inc["markers"], inc["curve"][-1]
    rate = n60 / inc["max_delay_s"]
    parts = [f"{inc['race'].replace('_', ' ')}, {inc['driver']} (car {inc['car']}) at {inc['onset_t']:.1f} s: "
             f"each second of delay here averaged {rate:.2f} cars ({n60} cars passed at racing speed in the "
             f"{inc['max_delay_s']:.0f} s after onset)"]
    if m["our_first_alert"]:
        parts.append(f"our first alert {m['our_first_alert']['t_after_onset']:+.1f} s")
    if m["our_escalation"]:
        e = m["our_escalation"]
        parts.append(f"our {e['flag']} {e['t_after_onset']:+.1f} s ({e['cars_by_then']} cars by then)")
    if m["official_yellow"] is not None:
        parts.append(f"official yellow {m['official_yellow']:+.1f} s ({m['cars_by_official_yellow']} cars by then)")
    if m["official_escalation"]:
        o = m["official_escalation"]
        parts.append(f"official {o['flag']} {o['t_after_onset']:+.1f} s ({o['cars_by_then']} cars by then)")
    else:
        parts.append("no official escalation")
    return "; ".join(parts) + "."


# ---- incidents of one race

def race_incidents(rid: str) -> list[dict]:
    assert_not_holdout(rid=rid)
    race = RaceData.load(rid)
    official, meta = race_files(rid, race_dir(rid))
    n, length = int(meta["n_msectors"]), float(meta["lap_length_m"])
    frame = add_own_ratio(race.frame, length)
    ons = onsets(frame, length)
    dets, recs = run_engine(race)
    names = driver_names(race)
    incidents, _ = build_incidents(official, meta, suspended_times(race.frame), n)
    out, seen = [], set()
    for inc in incidents:
        cand = ons[(ons["t"] >= inc.t - ONSET_BEFORE_S) & (ons["t"] <= inc.t + ONSET_AFTER_S)]
        cand = cand[cand["msector"].apply(lambda m: in_sectors(m, inc, n))]
        if cand.empty:
            continue
        onset = cand.iloc[0]
        key = (str(onset["drv"]), float(onset["t"]))
        if key in seen:                     # a re-flag of the same crash is the same curve
            continue
        seen.add(key)
        out.append(incident(rid, inc, onset, set(cand["drv"]), frame, dets, recs, names, n, length))
    return out


def incident(rid: str, inc, onset, cars: set[str], frame, dets: list[dict], recs: list[dict],
             names: dict[str, str], n: int, length: float) -> dict:
    t0 = float(onset["t"])
    rel = lambda t: round(float(t) - t0, 2)  # noqa: E731
    crash_dist = rest_position(frame, onset["drv"], onset)
    match = lambda m: in_sectors(int(m), inc, n)  # noqa: E731
    alert = next((d for d in dets if t0 - ALERT_BEFORE_S <= d["t"] <= t0 + ALERT_AFTER_S
                  and (cars & set(d["drivers"]) or match(d["msector"]))), None)
    esc = next((r for r in recs if r["flag"] in ESCALATIONS and t0 - ALERT_BEFORE_S <= r["t"] <= t0 + ALERT_AFTER_S
                and match(r["msector"])), None)
    yellow = min((m["t"] for m in inc.messages if m["flag"] in SECTOR_FLAGS), default=None)
    off_esc = min((m for m in inc.messages if m["flag"] in ESCALATIONS), key=lambda m: m["t"], default=None)
    # passes are listed up to the last marker (an official escalation can come minutes later), the curve stops at 60 s
    horizon = max([MAX_DELAY_S] + [t - t0 + 1 for t in (yellow, off_esc and off_esc["t"], alert and alert["t"],
                                                          esc and esc["t"]) if t is not None])
    raw = passes(frame, cars, crash_dist, t0, t0 + horizon, length, names)
    pass_list = [{"car": p["car"], "driver": p["driver"], "t_after_onset": rel(p["t"]), "speed_kmh": p["speed_kmh"],
                  "own_normal_kmh": p["own_normal_kmh"], "pct_of_own_normal": p["pct_of_own_normal"]} for p in raw]
    markers = {
        "our_first_alert": ({"t_after_onset": rel(alert["t"]), "type": alert["type"], "evidence": alert["evidence"],
                             "cars_by_then": cars_by(pass_list, rel(alert["t"]))} if alert else None),
        "our_escalation": ({"t_after_onset": rel(esc["t"]), "flag": esc["flag"], "reason": esc["reason"],
                            "cars_by_then": cars_by(pass_list, rel(esc["t"]))} if esc else None),
        "official_yellow": rel(yellow) if yellow is not None else None,
        "cars_by_official_yellow": cars_by(pass_list, rel(yellow) if yellow is not None else None),
        "official_escalation": ({"t_after_onset": rel(off_esc["t"]), "flag": off_esc["flag"],
                                 "cars_by_then": cars_by(pass_list, rel(off_esc["t"]))} if off_esc else None),
    }
    car = frame[frame["drv"] == str(onset["drv"])].sort_values("t")
    win = car[(car["t"] >= t0 - TRACE_BEFORE_S) & (car["t"] <= t0 + TRACE_AFTER_S)]
    at = car[(car["t"] - t0).abs() < 0.13]
    in_window = lambda t: t0 - ALERT_BEFORE_S <= t <= t0 + ALERT_AFTER_S  # noqa: E731
    rec = {"race": rid, "car": str(onset["drv"]), "driver": names.get(str(onset["drv"]), str(onset["drv"])),
           "cars_involved": sorted(cars), "onset_t": round(t0, 2), "msector": int(onset["msector"]),
           "lap": int(at["lap"].iloc[0]) if len(at) else None,
           "crash_location_m": round(crash_dist, 1), "official_top_flag": inc.top_flag,
           "max_delay_s": MAX_DELAY_S, "step_s": STEP_S, "racing_share": RACING_SHARE,
           "markers": markers, "curve": curve(pass_list), "passes": pass_list, "passes_until_s": round(horizon, 2),
           "passes_without_reference": sum(1 for p in pass_list if p["pct_of_own_normal"] is None
                                           and p["t_after_onset"] <= MAX_DELAY_S),
           # for the steward's cards (src.lab.cards): the onset car's speed against its own normal speed,
           # every detection and recommendation in the alert window, and the official messages of the incident
           "trace": {"t_after_onset": [rel(t) for t in win["t"]],
                     "speed_kmh": [finite(v) for v in win["speed"]],
                     "own_normal_kmh": [finite(v) for v in win["own_ref"]]},
           "detections": [{"t_after_onset": rel(d["t"]), "type": d["type"], "severity": d["severity"],
                           "drivers": list(d["drivers"]), "evidence": d["evidence"]} for d in dets
                          if in_window(d["t"]) and (cars & set(d["drivers"]) or match(d["msector"]))][:MAX_LISTED],
           "recommendations": [{"t_after_onset": rel(r["t"]), "flag": r["flag"], "confidence": r["confidence"],
                                "reason": r["reason"], "message": r["message"]} for r in recs
                               if in_window(r["t"]) and match(r["msector"])][:MAX_LISTED],
           "official_messages": [{"t_after_onset": rel(m["t"]), "flag": m["flag"], "message": m["message"]}
                                 for m in inc.messages]}
    rec["summary"] = summary_line(rec)
    rec["png"] = png_name(rec)
    return rec


def finite(v) -> float | None:
    return round(float(v), 1) if np.isfinite(v) else None


def png_name(inc: dict) -> str:
    return f"delay_cost_{inc['race']}_car{inc['car']}_{int(inc['onset_t'])}.png"


# ---- chart

def plot(inc: dict, path: Path) -> None:
    x, y = delays(inc["max_delay_s"], inc["step_s"]), inc["curve"]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.step(x, y, where="post", color=INK, lw=2, zorder=3)
    ax.fill_between(x, y, step="post", color=GRID, alpha=0.6, lw=0, zorder=2)
    m = inc["markers"]
    marks = []
    if m["our_first_alert"]:
        marks.append(("our first alert", m["our_first_alert"]["t_after_onset"], m["our_first_alert"]["type"]))
    if m["our_escalation"]:
        marks.append(("our escalation", m["our_escalation"]["t_after_onset"], m["our_escalation"]["flag"]))
    if m["official_yellow"] is not None:
        marks.append(("official yellow", m["official_yellow"], ""))
    if m["official_escalation"]:
        marks.append(("official escalation", m["official_escalation"]["t_after_onset"], m["official_escalation"]["flag"]))
    off_chart, level, last_x = [], 0, -1e9
    top = max(max(y), 1)
    for name, t, what in sorted(marks, key=lambda k: k[1]):
        label = f"{name} {what} {t:+.1f} s".replace("  ", " ")
        if t > inc["max_delay_s"] or t < 0:
            off_chart.append(label)
            continue
        color, ls = MARKER_STYLE[name]
        ax.axvline(t, color=color, ls=ls, lw=1.4, zorder=4)
        level = level + 1 if t - last_x < 0.09 * inc["max_delay_s"] else 0
        last_x = t
        ax.annotate(label, (t, top), xytext=(4, -4 - 12 * level), textcoords="offset points", ha="left", va="top",
                    fontsize=8, color=color, zorder=5,
                    bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85})
    ax.set_xlim(0, inc["max_delay_s"])
    ax.set_ylim(0, top * 1.25 + 0.5)
    ax.set_xlabel("Flag delay after the crash onset (seconds)", fontsize=9.5, color=INK2)
    ax.set_ylabel("Cars past the crash site at racing speed\n(80% or more of their own normal speed there)",
                  fontsize=8.5, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, labelsize=8.5, colors=INK2)
    rate = y[-1] / inc["max_delay_s"]
    fig.text(0.01, 0.985, f"{inc['race'].replace('_', ' ')}: {inc['driver']} (car {inc['car']}) crash, "
             f"the cost of every second of flag delay", fontsize=12.5, weight="bold", color=INK, va="top")
    fig.text(0.01, 0.94, f"{NOTE} Each second of delay here averaged {rate:.2f} cars "
             f"({y[-1]} cars in {inc['max_delay_s']:.0f} s)." + (f" Off the chart: {', '.join(off_chart)}." if off_chart else ""),
             fontsize=8.5, color=INK2, va="top", wrap=True)
    foot = f"Onset: {RULE}."
    if inc["race"] in CASE_RACES:
        foot += f" {OUT_OF_SAMPLE}"
    if inc["passes_without_reference"]:
        foot += f" {inc['passes_without_reference']} passes had no own reference speed and are not counted."
    fig.text(0.01, 0.012, foot, fontsize=7, color=INK2, va="bottom", wrap=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.24)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---- outputs

def write_outputs(incidents: list[dict], out_dir: Path, races: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"note": NOTE, "counterfactual": COUNTERFACTUAL, "onset_rule": RULE, "racing_share": RACING_SHARE,
           "max_delay_s": MAX_DELAY_S, "step_s": STEP_S, "delays_s": delays(), "races": races,
           "case_study_races": [r for r in races if r in CASE_RACES], "case_study_note": OUT_OF_SAMPLE,
           "incidents": incidents}
    (out_dir / "delay_cost.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    lines = ["# Delay-cost curves", "", f"{NOTE} {COUNTERFACTUAL}", "",
             f"Generated by `python -m src.lab.delay_cost` over {len(races)} races ({len(incidents)} incidents with an "
             f"identifiable onset). Onset rule: {RULE}.", "",
             "For each delay d (0 to 60 s, 0.5 s steps): cars that passed the spot where the onset car came to rest "
             f"between the onset and onset + d at {RACING_SHARE:.0%} or more of their own normal speed there "
             "(median of their previous 3 clean laps). Markers: our first alert, our escalation recommendation, "
             "the official yellow and the official escalation. 2021 Azerbaijan is a case study, out of sample twice: "
             + OUT_OF_SAMPLE, "", "## Summary, one line per incident", ""]
    lines += [f"- {inc['summary']}" for inc in incidents]
    lines += ["", "## Charts", ""]
    lines += [f"![{inc['race']} car {inc['car']}]({inc['png']})" for inc in incidents]
    (out_dir / "delay_cost.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Delay-cost curves per incident (counterfactual)")
    p.add_argument("races", nargs="*", help="race ids, default: every training race plus 2021_Azerbaijan")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 2))
    p.add_argument("--replot", action="store_true", help="redraw the PNGs from the saved JSON, no recomputation")
    a = p.parse_args()
    out_dir = Path(a.out)
    if a.replot:
        doc = json.loads((out_dir / "delay_cost.json").read_text(encoding="utf-8"))
        incidents, races = doc["incidents"], doc["races"]
    else:
        races = a.races or available_races() + CASE_RACES
        for rid in races:
            assert_not_holdout(rid=rid)
        with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
            results = list(ex.map(race_incidents, races))
        incidents = [inc for r in results for inc in r]
        write_outputs(incidents, out_dir, races)
    for inc in incidents:
        plot(inc, out_dir / inc["png"])
        print(inc["summary"])
    print(f"{len(incidents)} incidents, saved {out_dir / 'delay_cost.json'}, {out_dir / 'delay_cost.md'} and "
          f"{len(incidents)} PNGs")


if __name__ == "__main__":
    from src.lab.delay_cost import main as run    # run through the package import (pickling safety)
    run()

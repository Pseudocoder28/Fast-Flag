"""Latency from crash onset: race control vs our system, by official flag type and by
our alert type (training races only; the holdouts are never loaded here).

For each official incident (src.eval.incidents) the onset car is the car with the
earliest onset (src.eval.onset, fixed rule) in the 120 s before the incident's first
official message, in a matching marshal sector (any sector for track-wide-only
incidents). Then, for the first official message of each flag type in that incident:
- race control latency = official time - onset
- our latency = our first alert - onset, where our first alert is the first
  detection (any type) from 10 s before the onset to 180 s after it that involves an
  onset car or is in a matching sector
- lead = race control latency - our latency (positive = we were earlier)
Our detections come from the production detector settings with an ANOMALY model
trained on the other training races (leave-one-race-out). Events without an
identifiable onset car are listed with a reason, never dropped silently.

Run: python -m src.eval.latency_by_type
Writes docs/charts/latency_by_type.md, latency_by_type.csv, latency_by_type_no_onset.csv
and latency_by_type.png.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.detect.anomaly import attach_scores, train as train_anomaly
from src.detect.detectors import DetectorSuite
from src.detect.pipeline import load_config
from src.eval.incidents import INCIDENT_FLAGS, build_incidents, race_files, suspended_times
from src.eval.onset import FIELD_RACING, RULE, add_own_ratio, onsets
from src.ingest.holdout import assert_not_holdout
from src.ingest.sectors import sector_matches
from src.replay.engine import Engine, RaceData, available_races

CHARTS = Path("docs/charts")
ONSET_BEFORE_S = 120.0
ONSET_AFTER_S = 5.0
ALERT_BEFORE_S = 10.0
ALERT_AFTER_S = 180.0

# chart: reference palette slots 1 and 2 (documented as passing the colour-blindness checks),
# different marker shapes so identity is never colour alone; text uses ink tokens, not series colours
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("rc_latency_s", "Race control: first official message", "#2a78d6", "o"),
          ("our_latency_s", "Our system: first alert", "#eb6834", "s")]
FLAG_LABEL = {"YELLOW": "Yellow", "DOUBLE_YELLOW": "Double yellow", "VSC": "VSC", "SC": "Safety car", "RED": "Red flag"}
FLOOR_S = 0.25


def loro_suite(race: RaceData) -> DetectorSuite:
    """Production detector settings with an ANOMALY model trained on the other training races."""
    others = [r for r in available_races() if r != race.race]
    model = train_anomaly(others)
    attach_scores(race.frame, model)
    n = len(race.track.get("msectors", [])) or None
    return DetectorSuite(load_config(), anomaly_threshold=model.threshold, n_sectors=n)


def detections_loro(race: RaceData) -> list[dict]:
    eng = Engine(race, [loro_suite(race)])
    return [e["data"] for e in eng.advance(eng.t_end) if e["kind"] == "detection"]


def in_sectors(msector: int, inc, n: int) -> bool:
    return not inc.sectors or any(sector_matches(int(msector), s, n) for s in inc.sectors)


def no_onset_reason(frame: pd.DataFrame, inc, n: int) -> str:
    at = frame[(frame["t"] - inc.t).abs() < 0.13]
    if at.empty:
        return "no data at the official time"
    if int(at["lap"].iloc[0]) <= 2:
        return "lap 1 or 2: no clean reference laps yet"
    on_track = at[at["speed"].notna() & ~at["in_pit"].astype(bool)]
    near = on_track[on_track["msector"].apply(lambda m: in_sectors(m, inc, n))]
    if (near["own_ratio"] < 0.5).any() or (near["speed"] < 20).any():
        return "car already slow or stopped before the window (re-flag of an earlier stoppage)"
    if str(at["track_status"].iloc[0]) not in ("1", "2") or on_track["own_ratio"].median() < FIELD_RACING:
        return "field not racing (SC, VSC, restart or procession)"
    return "no car collapsed below 50% (debris, weather, or a car that went off and kept going)"


def race_events(rid: str, dets: list[dict] | None = None,
                allow_holdout: bool = False) -> tuple[list[dict], list[dict]]:
    """Onset-anchored latency events for one race. dets: precomputed detections (the
    A7 holdout run passes the frozen production detections); by default the
    detections come from an ANOMALY model trained on the other training races."""
    if not allow_holdout:
        assert_not_holdout(rid=rid)
    race = RaceData.load(rid)
    official, meta = race_files(rid)
    n = meta["n_msectors"]
    frame = add_own_ratio(race.frame, float(meta["lap_length_m"]))
    ons = onsets(frame, float(meta["lap_length_m"]))
    dets = sorted(dets if dets is not None else detections_loro(race), key=lambda d: d["t"])
    incidents, _ = build_incidents(official, meta, suspended_times(race.frame), n)
    events, missing = [], []
    for inc in incidents:
        cand = ons[(ons["t"] >= inc.t - ONSET_BEFORE_S) & (ons["t"] <= inc.t + ONSET_AFTER_S)]
        cand = cand[cand["msector"].apply(lambda m: in_sectors(m, inc, n))]
        firsts = {f: min(m["t"] for m in inc.messages if m["flag"] == f) for f in set(inc.flags)}
        if cand.empty:
            reason = no_onset_reason(frame, inc, n)
            missing += [{"race": rid, "t_official": t, "flag": f, "sectors": sorted(inc.sectors), "reason": reason}
                        for f, t in sorted(firsts.items(), key=lambda x: x[1])]
            continue
        onset = cand.iloc[0]
        cars = set(cand["drv"])
        after = frame[(frame["drv"] == onset["drv"]) & (frame["t"] > onset["t"]) & (frame["t"] <= onset["t"] + 20)]
        pitted = bool(after["in_pit"].any())         # a damaged car heading in, or a pit-entry slowdown
        alert = next((d for d in dets if onset["t"] - ALERT_BEFORE_S <= d["t"] <= onset["t"] + ALERT_AFTER_S
                      and (cars & set(d["drivers"]) or in_sectors(d["msector"], inc, n))), None)
        for f, t in firsts.items():
            rc = t - onset["t"]
            ours = alert["t"] - onset["t"] if alert else np.nan
            events.append({"race": rid, "flag": f, "t_onset": onset["t"], "onset_car": onset["drv"],
                           "cars": ",".join(sorted(cars)), "t_official": t, "rc_latency_s": round(rc, 2),
                           "our_alert_type": alert["type"] if alert else None,
                           "our_latency_s": round(ours, 2) if alert else np.nan,
                           "lead_s": round(rc - ours, 2) if alert else np.nan, "onset_car_pitted_20s": pitted})
    return events, missing


def summary(ev: pd.DataFrame, by: str, order: list[str]) -> pd.DataFrame:
    rows = []
    for key in order:
        g = ev[ev[by] == key]
        if g.empty:
            continue
        m = g[g["our_latency_s"].notna()]
        q = lambda s, p: float(np.percentile(s, p)) if len(s) else np.nan  # noqa: E731
        rows.append({by: key, "events": len(g), "matched": len(m),
                     "rc_median_s": q(g["rc_latency_s"], 50), "rc_p25_s": q(g["rc_latency_s"], 25),
                     "rc_p75_s": q(g["rc_latency_s"], 75), "ours_median_s": q(m["our_latency_s"], 50),
                     "ours_p25_s": q(m["our_latency_s"], 25), "ours_p75_s": q(m["our_latency_s"], 75),
                     "earlier_share": float((m["lead_s"] > 0).mean()) if len(m) else np.nan})
    return pd.DataFrame(rows)


def plot(ev: pd.DataFrame, by_flag: pd.DataFrame, n_races: int, n_missing: int, path: Path) -> None:
    """Headline chart: per flag type, race control latency next to ours (log time axis)."""
    flags = [f for f in INCIDENT_FLAGS if f in set(ev["flag"])]
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for row, f in enumerate(flags):
        y0 = len(flags) - 1 - row
        g = ev[ev["flag"] == f]
        for k, (col, _, color, marker) in enumerate(SERIES):
            vals = g[col].dropna().to_numpy()
            if not len(vals):
                continue
            y = y0 + (0.17 if k == 0 else -0.17)
            ax.scatter(np.clip(vals, FLOOR_S, None), y + rng.uniform(-0.05, 0.05, len(vals)), s=16, color=color,
                       alpha=0.35, marker=marker, linewidths=0, zorder=2)
            p25, med, p75 = np.percentile(vals, [25, 50, 75])
            ax.plot([max(p25, FLOOR_S), max(p75, FLOOR_S)], [y, y], color=color, lw=2, solid_capstyle="round", zorder=3)
            ax.scatter([max(med, FLOOR_S)], [y], s=90, color=color, marker=marker, edgecolors=SURFACE,
                       linewidths=2, zorder=4)
            ax.annotate(f"{med:.1f} s", (max(med, FLOOR_S), y), xytext=(0, 8 if k == 0 else -8),
                        textcoords="offset points", ha="center", va="bottom" if k == 0 else "top",
                        fontsize=8.5, color=INK2)
    stats = by_flag.set_index("flag")
    ax.set_yticks(range(len(flags)))
    ax.set_yticklabels([f"{FLAG_LABEL[f]}\n{int(stats.loc[f, 'events'])} events, ours earlier in "
                        f"{stats.loc[f, 'earlier_share']:.0%}" for f in reversed(flags)], fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 400)
    ticks = [0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.25 or less"] + [f"{t:g}" for t in ticks[1:]], fontsize=8.5, color=INK2)
    ax.minorticks_off()
    ax.set_xlabel("Seconds after the crash onset (log scale)", fontsize=9.5, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0)
    handles = [Line2D([0], [0], marker=m, color=c, lw=2, markersize=8, markeredgecolor=SURFACE, label=name)
               for _, name, c, m in SERIES]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9,
              labelcolor=INK)
    fig.text(0.01, 0.985, "From crash onset to first flag: race control vs our system", fontsize=13,
             weight="bold", color=INK, va="top")
    fig.text(0.01, 0.945, f"Replay of historical FastF1 data, {n_races} training races (2023 to 2026), "
             f"{len(ev)} flag events with an identifiable crash onset", fontsize=9.5, color=INK2, va="top")
    fig.text(0.01, 0.01, "Onset: speed below 50% of the car's own speed at that point on its previous 3 clean laps, "
             "for 1 s. Dots: events. Large marker: median. Line: middle half.\nValues at or below 0.25 s (race control "
             "flagging before the collapse, or our alert on the onset tick) are drawn at 0.25 s.\n"
             f"{n_missing} events without an identifiable onset car are listed in latency_by_type.md.",
             fontsize=7.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.235, right=0.98, top=0.84, bottom=0.21)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    races = available_races()
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(race_events, races))
    ev = pd.DataFrame([e for r in results for e in r[0]])
    missing = pd.DataFrame([m for r in results for m in r[1]])
    by_flag = summary(ev, "flag", list(INCIDENT_FLAGS))
    types = ["IMPACT", "STOPPED", "SPIN", "MULTI", "DROPOUT", "ANOMALY"]
    by_type = summary(ev.assign(our_alert_type=ev["our_alert_type"].fillna("none (missed)")),
                      "our_alert_type", types + ["none (missed)"])
    pd.set_option("display.width", 220)
    print(by_flag.round(2).to_string(index=False))
    print(by_type.round(2).to_string(index=False))
    print(f"events with an onset car: {len(ev)}; without: {len(missing)}")
    print(missing.groupby("reason").size().to_string() if len(missing) else "")
    CHARTS.mkdir(parents=True, exist_ok=True)
    ev.to_csv(CHARTS / "latency_by_type.csv", index=False, encoding="utf-8")
    missing.to_csv(CHARTS / "latency_by_type_no_onset.csv", index=False, encoding="utf-8")
    write_report(ev, missing, by_flag, by_type, len(races))
    plot(ev, by_flag, len(races), len(missing), CHARTS / "latency_by_type.png")


def write_report(ev: pd.DataFrame, missing: pd.DataFrame, by_flag: pd.DataFrame, by_type: pd.DataFrame,
                 n_races: int) -> None:
    reasons = missing.groupby(["reason", "flag"]).size().unstack(fill_value=0) if len(missing) else pd.DataFrame()
    (CHARTS / "latency_by_type.md").write_text(
        "# Latency from crash onset: race control vs our system (training races)\n\n"
        f"Replay of historical FastF1 data, {n_races} training races, no holdout race. Latencies are measured "
        "from the crash onset.\n\n"
        f"**Onset rule** (fixed, independent of the detectors): {RULE}.\n\n"
        f"The onset car of an official incident is the car with the earliest onset in the {ONSET_BEFORE_S:.0f} s "
        "before the incident's first official message, in a matching marshal sector. Race control latency = "
        "first official message of that flag type - onset. Our latency = our first alert (any detection from "
        f"{ALERT_BEFORE_S:.0f} s before the onset to {ALERT_AFTER_S:.0f} s after, involving an onset car or in a "
        "matching sector) - onset. Detections: production settings, ANOMALY model trained on the other races.\n\n"
        "**What this shows and what it does not.** This compares latency on crashes where a car clearly "
        "collapsed, which is also what our detectors are best at, so the matched share here is not a recall "
        "figure (recall per incident is in detect_eval.md and tune_results.md). An incident contributes one "
        "event per flag type it reached, so SC and red flag events are usually escalations of an incident "
        "that also had a yellow. Our detection thresholds were tuned on these training races; the ANOMALY "
        "model was not trained on the race it scores.\n\n"
        "## By official flag type\n\n" + by_flag.round(2).to_markdown(index=False)
        + "\n\n## By our alert type\n\n" + by_type.round(2).to_markdown(index=False)
        + f"\n\n## Events without an identifiable onset car ({len(missing)} events)\n\n"
        + (reasons.to_markdown() if len(reasons) else "none")
        + "\n\nEvery one is listed in latency_by_type_no_onset.csv.\n\n"
        + f"**Check:** in {int(ev.drop_duplicates(['race', 't_onset'])['onset_car_pitted_20s'].sum())} of "
          f"{len(ev.drop_duplicates(['race', 't_onset']))} incidents the onset car entered the pit lane within 20 s "
          "(a damaged car heading in, or a car braking for the pit entry before the pit-lane geometry covers it). "
          "They are kept and marked in latency_by_type.csv (onset_car_pitted_20s).\n\n"
          "![latency by flag type](latency_by_type.png)\n",
        encoding="utf-8")


if __name__ == "__main__":
    from src.eval.latency_by_type import main as run    # run through the package import (pickling safety)
    run()

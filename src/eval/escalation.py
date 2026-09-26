"""Escalation scorecard: our race control's VSC, SC and red flag recommendations
against the official ones, on the training races (the holdouts are never loaded
here; the A7 holdout run calls race_scorecard with the frozen models).

Our side: the production detector settings with an ANOMALY model trained on the
other training races (leave-one-race-out, as in src.eval.run), streamed tick by
tick through the race control engine (src.racecontrol.engine) in the order the
live server sends envelopes. The risk model only changes a recommendation's
confidence, never its flag or its time, so it is left out.

Official side: every official incident (src.eval.incidents) with a VSC, SC or red
flag message. Its escalation time is the first of those messages.

Per official escalation:
- earlier / later: our first VSC, SC or red recommendation from 60 s before the
  incident's first official message to 60 s after the official escalation, with a
  cause sector that matches the incident (the flagged sector, up to 2 downstream or
  1 upstream; any sector for a track-wide-only incident). Lead = official
  escalation time - ours, positive = we were earlier.
- already out: no such recommendation, but our own VSC, SC or red was already out
  at that moment. Kept out of the lead times, like "held" in the detection eval.
- missed: neither. The detail says whether a car collapsed near the incident
  (src.eval.onset) and what we recommended instead, or why no car collapsed
  (debris, weather, a re-flag): car telemetry cannot see those.

Per recommendation of ours that matched no official escalation:
- during an official neutralisation: the official track status was already SC,
  VSC or red. It supports what race control had done, so it is not counted as extra.
- official yellows only: it fits an official incident (window and sectors as in
  src.eval.run) that race control kept at yellow or double yellow
- escalated incident, outside the match window: it fits an escalated incident,
  but not within the match window of its official escalation
- no official flag: nothing official nearby
Extra escalations per race hour counts the last three, over race hours outside red
flag suspensions (as src.eval.run).

Run: python -m src.eval.escalation
Writes docs/charts/escalation.md, escalation.json, escalation_official.csv,
escalation_ours.csv and escalation.png.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.incidents import MATCH_BEFORE_S, build_incidents, race_files, suspended_times, was_suspended
from src.eval.latency_by_type import ONSET_AFTER_S, ONSET_BEFORE_S, in_sectors, loro_suite, no_onset_reason
from src.eval.onset import add_own_ratio, onsets
from src.eval.run import belongs
from src.ingest.features import NEUTRAL_STATUS
from src.ingest.holdout import assert_not_holdout
from src.racecontrol.engine import ON_TRACK_LAT_OFF_M, RaceControl
from src.replay.engine import Engine, Processor, RaceData, available_races

CHARTS = Path("docs/charts")
ESCALATIONS = ("VSC", "SC", "RED")
RANK = {f: i for i, f in enumerate(ESCALATIONS)}
SECTOR_FLAGS = ("YELLOW", "DOUBLE_YELLOW")
LATE_S = 60.0
STOPPED_KMH = 5.0          # stopped-car rows for the lateral offset check
MATCHED = ("earlier", "later")
OFFICIAL_COLUMNS = ["race", "t_incident", "t_official", "official_flag", "official_top", "sectors", "status", "lead_s",
                    "our_t", "our_flag", "our_top", "miss_kind", "onset_car", "onset_car_alerts", "detail"]
OURS_COLUMNS = ["race", "t", "flag", "msector", "reason", "category"]
EXTRA = ("official yellows only", "escalated incident, outside the match window", "no official flag")


# ---- our side: detections streamed through the race control engine

def replay(race: RaceData, suite: Processor) -> tuple[list[dict], list[dict], list[float]]:
    """Returns (detections, recommendations, times the engine reset its flag state)."""
    rc, eng = RaceControl(), Engine(race, [suite])
    dets, recs, resets = [], [], []
    for step in eng.steps(eng.t_end):
        for env in step.envelopes:
            if env["kind"] == "tick":
                out, did_reset = rc.on_tick(env["data"])
                if did_reset:
                    resets.append(float(env["data"]["t"]))
            elif env["kind"] == "detection":
                dets.append(env["data"])
                out = rc.on_detection(env["data"])
            else:
                continue
            recs.extend(out)
    return dets, recs, resets


def track_wide_changes(recs: list[dict], resets: list[float]) -> list[tuple[float, str, int | None]]:
    """(t, flag, cause sector) each time our track-wide flag changes: VSC, SC or RED out,
    CLEAR on TRACK CLEAR or on an engine reset. Stable sort keeps the emission order."""
    changes = [(t, "CLEAR", None) for t in resets]
    changes += [(r["t"], r["flag"], r["msector"]) for r in recs
                if r["flag"] in ESCALATIONS or (r["flag"] == "CLEAR" and r["message"] == "TRACK CLEAR")]
    return sorted(changes, key=lambda c: c[0])


def track_wide_at(changes: list[tuple[float, str, int | None]], t: float) -> tuple[str, float | None, int | None]:
    """Our track-wide flag at t: (flag, since, cause sector)."""
    since, flag, cause = None, "CLEAR", None
    for tc, f, ms in changes:
        if tc > t:
            break
        since, flag, cause = tc, f, ms
    return flag, since, cause


def official_neutral(status: pd.Series, susp: pd.Series, t: float) -> bool:
    """Official track status SC, VSC or red at t (as of the last tick), or suspended."""
    i = max(int(status.index.searchsorted(t, side="right")) - 1, 0)
    return str(status.iloc[i]) in NEUTRAL_STATUS or was_suspended(susp, t)


# ---- one race

def onset_car(inc, ons: pd.DataFrame, n: int) -> pd.Series | None:
    """The first car to collapse near the incident (same rule as src.eval.latency_by_type)."""
    cand = ons[(ons["t"] >= inc.t - ONSET_BEFORE_S) & (ons["t"] <= inc.t + ONSET_AFTER_S)]
    cand = cand[cand["msector"].apply(lambda m: in_sectors(m, inc, n))]
    return None if cand.empty else cand.iloc[0]


def car_alerts(dets: list[dict], car: pd.Series) -> str:
    """Our detection types for the onset car from 10 s before its onset to 60 s after."""
    t0 = float(car["t"])
    return " ".join(sorted({d["type"] for d in dets if t0 - 10 <= d["t"] <= t0 + 60 and car["drv"] in d["drivers"]}))


def miss_detail(inc, recs: list[dict], car: pd.Series | None, frame: pd.DataFrame, n: int, t_hi: float) -> dict:
    if car is None:
        return {"miss_kind": "no car collapse", "detail": no_onset_reason(frame, inc, n)}
    shown = [r["flag"] for r in recs if r["flag"] in SECTOR_FLAGS and inc.t - MATCH_BEFORE_S <= r["t"] <= t_hi
             and in_sectors(r["msector"], inc, n)]
    top = max(shown, key=SECTOR_FLAGS.index, default=None)
    what = (f"we showed {top.replace('_', ' ').lower()} but did not escalate" if top
            else "we made no recommendation")
    d = inc.t - float(car["t"])
    return {"miss_kind": "car collapse seen",
            "detail": f"car {car['drv']} collapsed {abs(d):.0f} s {'before' if d >= 0 else 'after'} the first "
                      f"official message; {what}"}


def production_suite(race: RaceData) -> Processor:
    """The frozen production detectors and ANOMALY model, as the live server runs them."""
    from src.detect.pipeline import detection_processors
    return detection_processors(race)[0]


def race_scorecard(rid: str, suite_for: Callable[[RaceData], Processor] = loro_suite,
                   allow_holdout: bool = False) -> dict:
    """Both sides of the scorecard for one race. suite_for builds the detectors for the
    loaded race: by default the production settings with an ANOMALY model trained on
    the other training races; the A7 holdout run passes production_suite."""
    if not allow_holdout:
        assert_not_holdout(rid=rid)
    race = RaceData.load(rid)
    official, meta = race_files(rid)
    n, length = meta["n_msectors"], float(meta["lap_length_m"])
    susp = suspended_times(race.frame)
    incidents, _ = build_incidents(official, meta, susp, n)
    dets, recs, resets = replay(race, suite_for(race))
    frame = add_own_ratio(race.frame, length)
    ons = onsets(frame, length)
    status = frame.drop_duplicates("t").set_index("t")["track_status"].sort_index()
    esc = [r for r in recs if r["flag"] in ESCALATIONS]
    changes = track_wide_changes(recs, resets)

    official_rows, matched_ids = [], set()
    for inc in incidents:
        msgs = [m for m in inc.messages if m["flag"] in ESCALATIONS]
        if not msgs:
            continue
        first = min(msgs, key=lambda m: m["t"])
        t_hi = first["t"] + LATE_S
        mine = [r for r in esc if inc.t - MATCH_BEFORE_S <= r["t"] <= t_hi and in_sectors(r["msector"], inc, n)]
        matched_ids |= {r["id"] for r in mine}
        car = onset_car(inc, ons, n)
        row = {"race": rid, "t_incident": inc.t, "t_official": first["t"], "official_flag": first["flag"],
               "official_top": max((m["flag"] for m in msgs), key=RANK.get),
               "sectors": " ".join(map(str, sorted(inc.sectors))) or "track-wide", "status": "",
               "lead_s": np.nan, "our_t": np.nan, "our_flag": None, "our_top": None, "miss_kind": None,
               "onset_car": None if car is None else car["drv"],
               "onset_car_alerts": "" if car is None else car_alerts(dets, car), "detail": ""}
        if mine:
            lead = first["t"] - mine[0]["t"]
            row.update(status="earlier" if lead > 0 else "later", lead_s=round(lead, 2), our_t=mine[0]["t"],
                       our_flag=mine[0]["flag"], our_top=max((r["flag"] for r in mine), key=RANK.get),
                       detail=mine[0]["reason"])
        else:
            flag, since, cause = track_wide_at(changes, first["t"])
            if flag in ESCALATIONS:
                row.update(status="already out", our_t=since, our_flag=flag,
                           detail=f"our {flag} out since {first['t'] - since:.0f} s before, cause sector {cause}")
            else:
                row.update(status="missed", **miss_detail(inc, recs, car, frame, n, t_hi))
        official_rows.append(row)

    ours_rows = []
    for r in esc:
        if r["id"] in matched_ids:
            category = "matched"
        elif official_neutral(status, susp, r["t"]):
            category = "during an official neutralisation"
        else:
            fit = [i for i in incidents if belongs(r, i, n)]
            if any(f in ESCALATIONS for i in fit for f in i.flags):
                category = "escalated incident, outside the match window"
            else:
                category = "official yellows only" if fit else "no official flag"
        ours_rows.append({"race": rid, "t": r["t"], "flag": r["flag"], "msector": r["msector"],
                          "reason": r["reason"], "category": category})

    dt = np.diff(susp.index.to_numpy(), append=susp.index[-1])
    hours = float(dt[~susp.to_numpy()].sum()) / 3600
    stopped = race.frame.loc[(race.frame["speed"] < STOPPED_KMH) & ~race.frame["in_pit"].astype(bool), "lat_off"]
    stopped = stopped.abs().dropna()
    return {"race": rid, "official": official_rows, "ours": ours_rows, "hours": hours, "resets": len(resets),
            "stopped_rows": len(stopped), "stopped_within_1m": int((stopped <= 1.0).sum()),
            "stopped_off_track": int((stopped > ON_TRACK_LAT_OFF_M).sum()),
            "stopped_off_track_cars": sorted(race.frame.loc[stopped.index[stopped > ON_TRACK_LAT_OFF_M], "drv"].unique())}


# ---- totals

def counts(s: pd.Series, order: tuple[str, ...] | list[str]) -> dict[str, int]:
    vc = s.value_counts()
    return {k: int(vc.get(k, 0)) for k in order}


def tables(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(official escalations, our escalations) over all races, with fixed columns even when empty."""
    return (pd.DataFrame([r for res in results for r in res["official"]], columns=OFFICIAL_COLUMNS),
            pd.DataFrame([r for res in results for r in res["ours"]], columns=OURS_COLUMNS))


def summarise(results: list[dict]) -> dict:
    off, ours = tables(results)
    hours = sum(res["hours"] for res in results)
    m = off[off["status"].isin(MATCHED)]
    lead = m["lead_s"].to_numpy(float)
    extra = ours[ours["category"].isin(EXTRA)]
    by_flag = {}
    for f in ESCALATIONS:
        g = off[off["official_flag"] == f]
        gl = g.loc[g["status"].isin(MATCHED), "lead_s"].to_numpy(float)
        by_flag[f] = {"official": len(g), **counts(g["status"], ["earlier", "later", "already out", "missed"]),
                      "median_lead_s": round(float(np.median(gl)), 1) if len(gl) else None}
    return {
        "races": len(results), "race_hours": round(hours, 2),
        "official_escalations": len(off),
        "status": counts(off["status"], ["earlier", "later", "already out", "missed"]),
        "matched": len(m),
        "median_lead_s": round(float(np.median(lead)), 1) if len(lead) else None,
        "lead_iqr_s": [round(float(q), 1) for q in np.percentile(lead, [25, 75])] if len(lead) else None,
        "same_first_flag": int((m["our_flag"] == m["official_flag"]).sum()),
        "flag_pairs": {f"{a}->{b}": int(k) for (a, b), k in
                       m.groupby(["official_flag", "our_flag"]).size().items()},
        "by_official_flag": by_flag,
        "missed_kind": counts(off.loc[off["status"] == "missed", "miss_kind"],
                              ["car collapse seen", "no car collapse"]),
        "our_escalations": len(ours),
        "our_by_category": counts(ours["category"], ["matched", "during an official neutralisation", *EXTRA]),
        "extra": len(extra),
        "extra_per_hour": round(len(extra) / hours, 2) if hours else None,
        "extra_by_flag": counts(extra["flag"], ESCALATIONS),
        "engine_resets": int(sum(res["resets"] for res in results)),
        "stopped_lateral_offset": stopped_offset(results),
        "impact_rule": impact_rule(off),
    }


def stopped_offset(results: list[dict]) -> dict:
    """Why we never recommend VSC: the race control engine calls a stopped car off track
    (VSC) when it is more than ON_TRACK_LAT_OFF_M from the racing line, but FastF1
    positions of stopped cars sit on the line, even for retired cars parked in run-off."""
    rows = sum(r["stopped_rows"] for r in results)
    return {"stopped_car_rows": rows,
            "share_within_1m": round(sum(r["stopped_within_1m"] for r in results) / rows, 4) if rows else None,
            "rows_beyond_threshold": sum(r["stopped_off_track"] for r in results),
            "episodes_beyond_threshold": [f"{r['race']} car {c}" for r in results for c in r["stopped_off_track_cars"]],
            "engine_off_track_threshold_m": ON_TRACK_LAT_OFF_M}


def impact_rule(off: pd.DataFrame) -> dict:
    """Candidate replacement (not in the engine): SC or red when our IMPACT or MULTI
    detection involved the onset car, VSC otherwise. Scored on the official escalations
    with an onset car, against the current outcome (always SC)."""
    has = off[off["onset_car"].notna()]
    impact = has["onset_car_alerts"].str.contains("IMPACT|MULTI")
    severe = has["official_flag"].isin(["SC", "RED"])
    return {"escalations_with_onset_car": len(has),
            "official_sc_or_red_with_impact": int((impact & severe).sum()), "official_sc_or_red": int(severe.sum()),
            "official_vsc_without_impact": int((~impact & ~severe).sum()), "official_vsc": int((~severe).sum()),
            "right_with_impact_rule": int((impact == severe).sum()),
            "right_with_current_rule": int(severe.sum())}


# ---- report and chart

FLAG_NAME = {"VSC": "VSC", "SC": "Safety car", "RED": "Red flag"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY_DARK, GRAY_LIGHT = "#2a78d6", "#eb6834", "#8a8883", "#d6d4cf"
SEGMENTS = [("earlier", "Earlier than race control", BLUE, SURFACE),
            ("later", "Later than race control", ORANGE, INK),
            ("car collapse seen", "Missed: a car collapse we could see", GRAY_DARK, SURFACE),
            ("no car collapse", "Missed: no car collapse (re-flag, debris, weather, lap 1)", GRAY_LIGHT, INK)]
IN_SAMPLE = ("In-sample: the detector settings were tuned and the race control rules were set on these races; "
             "the holdout run (A7) gives the out-of-sample version.")
COUNTERFACTUAL = ("Counterfactual: assumes race control acted on our recommendation at once. Race control also has "
                  "marshal reports and CCTV, and picks VSC or SC by recovery work we cannot see. Claim earlier than "
                  "the race control feed, never earlier than the marshals.")


def outcome(row: pd.Series) -> str:
    return row["miss_kind"] if row["status"] == "missed" else row["status"]


def plot(off: pd.DataFrame, summ: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.2), facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [3.1, 1.9], "hspace": 0.55})
    fig.subplots_adjust(left=0.2, right=0.96, top=0.83, bottom=0.27)
    rng = np.random.default_rng(0)
    m = off[off["status"].isin(MATCHED)]
    for i, f in enumerate(ESCALATIONS):
        g = m[m["official_flag"] == f]
        y = i + rng.uniform(-0.2, 0.2, len(g))
        ax1.scatter(g["lead_s"], y, s=52, c=[BLUE if v > 0 else ORANGE for v in g["lead_s"]],
                    edgecolors=SURFACE, linewidths=1.5, zorder=3)
        if len(g):
            med = float(g["lead_s"].median())
            ax1.plot([med, med], [i - 0.32, i + 0.32], color=INK, lw=2, zorder=4, solid_capstyle="round")
            label = f"median {med:.0f} s" if len(g) > 1 else f"{med:.0f} s"
            ax1.text(med, i - 0.38, label, ha="center", va="bottom", fontsize=9, color=INK)
    ax1.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax1.text(0.8, 2.52, "official escalation", ha="left", va="center", fontsize=8.5, color=INK2)
    ax1.set_ylim(2.65, -0.65)
    ax1.set_yticks(range(len(ESCALATIONS)))
    ax1.set_yticklabels([f"{FLAG_NAME[f]}\n{summ['by_official_flag'][f]['earlier'] + summ['by_official_flag'][f]['later']}"
                         f" of {summ['by_official_flag'][f]['official']} matched" for f in ESCALATIONS])
    ax1.set_xlabel("Seconds our VSC, SC or red recommendation came before the official one", color=INK2)
    ax1.set_title("Lead over the official escalation, per matched escalation", loc="left", fontsize=10.5,
                  color=INK, pad=8)
    lo = min(-10.0, float(m["lead_s"].min()) - 5) if len(m) else -10.0
    ax1.set_xlim(lo, max(65.0, float(m["lead_s"].max()) + 5) if len(m) else 65.0)

    left = np.zeros(len(ESCALATIONS))
    kinds = off.apply(outcome, axis=1)
    for key, label, colour, text_colour in SEGMENTS:
        vals = np.array([int(((off["official_flag"] == f) & (kinds == key)).sum()) for f in ESCALATIONS])
        ax2.barh(range(len(ESCALATIONS)), vals, left=left, height=0.62, color=colour, edgecolor=SURFACE,
                 linewidth=2, label=label)
        for i, v in enumerate(vals):
            if v:
                ax2.text(left[i] + v / 2, i, str(v), ha="center", va="center", fontsize=9, color=text_colour)
        left += vals
    ax2.set_ylim(len(ESCALATIONS) - 0.4, -0.6)
    ax2.set_yticks(range(len(ESCALATIONS)))
    ax2.set_yticklabels([FLAG_NAME[f] for f in ESCALATIONS])
    ax2.set_xlabel("Official escalations", color=INK2)
    ax2.set_title("Outcome per official escalation", loc="left", fontsize=10.5, color=INK, pad=8)
    fig.legend(handles=[Patch(facecolor=c, edgecolor=SURFACE, label=lbl) for _, lbl, c, _ in SEGMENTS],
               loc="upper left", bbox_to_anchor=(0.19, 0.2), ncol=2, frameon=False, fontsize=8.5)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        for lbl in ax.get_yticklabels():
            lbl.set_color(INK)

    c = summ["our_by_category"]
    fig.text(0.02, 0.975, f"Escalation scorecard: {summ['official_escalations']} official VSC, SC and red flags in "
             f"{summ['races']} training races", fontsize=13, color=INK, weight="bold", va="top")
    fig.text(0.02, 0.94, f"Our race control recommended a neutralisation for {summ['matched']}, "
             f"{summ['status']['earlier']} of them earlier (median {summ['median_lead_s']:.0f} s).\nIt also made "
             f"{summ['extra']} escalations race control did not ({summ['extra_per_hour']:.2f} per race hour).",
             fontsize=9.5, color=INK2, va="top", linespacing=1.4)
    fig.text(0.02, 0.015, f"Our {summ['our_escalations']} recommendations: {c['matched']} matched, "
             f"{c['during an official neutralisation']} during an official neutralisation, "
             f"{c['official yellows only']} where race control kept yellows only, {c['no official flag']} with no "
             f"official flag. Replay of historical FastF1 data.\n{IN_SAMPLE}\n"
             + COUNTERFACTUAL.replace(" Claim", "\nClaim"), fontsize=7.5, color=INK2, va="bottom", linespacing=1.4)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def pairs_text(pairs: dict[str, int]) -> str:
    out = []
    for key, k in pairs.items():
        theirs, ours = key.split("->")
        out.append(f"official {FLAG_NAME[theirs]} and ours {FLAG_NAME[ours]}: {k}")
    return "; ".join(out)


def pct(a: int, b: int) -> str:
    return f"{a / b:.0%}" if b else "n/a"


def write_report(off: pd.DataFrame, ours: pd.DataFrame, summ: dict, results: list[dict]) -> str:
    st, c, so, ir = summ["status"], summ["our_by_category"], summ["stopped_lateral_offset"], summ["impact_rule"]
    lines = [
        "# Escalation scorecard (training races)", "",
        f"Replay of historical FastF1 data: {summ['races']} training races, {summ['race_hours']:.1f} race hours. "
        "Our side is the race control engine (src/racecontrol) fed by the production detector settings with an "
        "ANOMALY model trained on the other training races. Definitions: src/eval/escalation.py.", "",
        IN_SAMPLE, "", COUNTERFACTUAL, "",
        f"## Official escalations: {summ['official_escalations']}", "",
        f"- We recommended a VSC, SC or red for {summ['matched']} of {summ['official_escalations']} "
        f"({pct(summ['matched'], summ['official_escalations'])}): {st['earlier']} earlier than race control, "
        f"{st['later']} later. Median lead {summ['median_lead_s']:.1f} s (middle half "
        f"{summ['lead_iqr_s'][0]:.1f} to {summ['lead_iqr_s'][1]:.1f} s).",
        f"- Already out (our flag was up before, no new recommendation): {st['already out']}. Missed: "
        f"{st['missed']}, of which {summ['missed_kind']['car collapse seen']} had a car collapse our onset rule "
        f"found and {summ['missed_kind']['no car collapse']} had none.",
        f"- Same first flag as race control: {summ['same_first_flag']} of {summ['matched']} "
        f"({pairs_text(summ['flag_pairs'])}).",
        "", "| official flag | escalations | earlier | later | missed | median lead (s) |", "|---|---|---|---|---|---|"]
    for f in ESCALATIONS:
        b = summ["by_official_flag"][f]
        med = f"{b['median_lead_s']:.1f}" if b["median_lead_s"] is not None else "n/a"
        lines.append(f"| {FLAG_NAME[f]} | {b['official']} | {b['earlier']} | {b['later']} | {b['missed']} | {med} |")
    lines += ["", "Missed official escalations:", "",
              "| race | official time (s) | flag | sectors | kind | detail |", "|---|---|---|---|---|---|"]
    for _, r in off[off["status"] == "missed"].iterrows():
        lines.append(f"| {r['race']} | {r['t_official']:.1f} | {r['official_flag']} | {r['sectors']} | "
                     f"{r['miss_kind']} | {r['detail']} |")
    lines += ["", f"## Our escalations: {summ['our_escalations']}", "",
              f"- Matched an official escalation: {c['matched']}. During an official neutralisation (not extra): "
              f"{c['during an official neutralisation']}.",
              f"- Extra, race control never escalated: {summ['extra']}, {summ['extra_per_hour']:.2f} per race hour "
              f"({c['official yellows only']} where race control kept yellows only, "
              f"{c['escalated incident, outside the match window']} near an escalated incident but outside its "
              f"match window, {c['no official flag']} with no official flag). By flag: "
              + ", ".join(f"{FLAG_NAME[k]} {v}" for k, v in summ["extra_by_flag"].items()) + ".",
              f"- Race control engine resets (tick jumps over 2 s): {summ['engine_resets']}.", "",
              "## Finding: our engine never recommends a VSC", "",
              f"The engine sends a VSC for a car stopped more than {so['engine_off_track_threshold_m']:.0f} m from "
              f"the racing line and an SC otherwise. FastF1 positions of stopped cars sit on the line, even for "
              f"retired cars parked in run-off: of {so['stopped_car_rows']:,} stopped-car rows (below "
              f"{STOPPED_KMH:.0f} km/h, outside the pit lane), {so['share_within_1m']:.2%} are within 1 m of it and "
              f"{so['rows_beyond_threshold']} are beyond {so['engine_off_track_threshold_m']:.0f} m (short episodes: "
              f"{', '.join(so['episodes_beyond_threshold']) or 'none'}). So every sustained stop becomes an SC, and "
              f"all {sum(v for k, v in summ['flag_pairs'].items() if k.startswith('VSC'))} matched official VSCs "
              "got an SC from us.", "",
              f"Candidate rule (not in the engine): SC when our IMPACT or MULTI detection involved the car, VSC when it "
              f"stopped without one. On the {ir['escalations_with_onset_car']} official escalations with an onset car, "
              f"{ir['official_sc_or_red_with_impact']} of {ir['official_sc_or_red']} SC or red had an impact and "
              f"{ir['official_vsc_without_impact']} of {ir['official_vsc']} VSC had none: it picks race control's "
              f"flag {ir['right_with_impact_rule']} times against {ir['right_with_current_rule']} now. Same "
              "in-sample caveat, and a small sample.", "",
              "## Per race", "",
              "| race | race hours | official escalations | earlier | later | missed | our escalations | extra |",
              "|---|---|---|---|---|---|---|---|"]
    for res in results:
        o, u = tables([res])
        lines.append(f"| {res['race']} | {res['hours']:.2f} | {len(o)} | {(o['status'] == 'earlier').sum()} | "
                     f"{(o['status'] == 'later').sum()} | {(o['status'] == 'missed').sum()} | {len(u)} | "
                     f"{u['category'].isin(EXTRA).sum()} |")
    lines += ["", "Chart: escalation.png. Every official escalation: escalation_official.csv. Every recommendation "
              "of ours: escalation_ours.csv.", ""]
    return "\n".join(lines)


def main() -> None:
    races = available_races()
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(race_scorecard, races))
    off, ours = tables(results)
    summ = summarise(results)
    print(json.dumps(summ, indent=2))
    CHARTS.mkdir(parents=True, exist_ok=True)
    off.to_csv(CHARTS / "escalation_official.csv", index=False, encoding="utf-8")
    ours.to_csv(CHARTS / "escalation_ours.csv", index=False, encoding="utf-8")
    (CHARTS / "escalation.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    (CHARTS / "escalation.md").write_text(write_report(off, ours, summ, results), encoding="utf-8")
    plot(off, summ, CHARTS / "escalation.png")
    print(f"wrote {CHARTS / 'escalation.md'} and the csv, json and png next to it")


if __name__ == "__main__":
    main()

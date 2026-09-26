"""Build docs/charts/NUMBERS.md: the single source of truth for every number in the pitch.

It only reads the result files in docs/charts/, so it can never disagree with them.
Rerun it after any evaluation changes; never edit NUMBERS.md by hand.

Run: python -m src.eval.numbers
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

CHARTS = Path("docs") / "charts"
OUT = CHARTS / "NUMBERS.md"
FLAG_NAMES = {"YELLOW": "yellow", "DOUBLE_YELLOW": "double yellow", "VSC": "VSC", "SC": "SC", "RED": "red"}


def read_md_table(path: Path) -> pd.DataFrame:
    """First pipe table in a markdown file (as written by pandas to_markdown)."""
    rows = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in rows]
    header, body = cells[0], [r for r in cells[2:] if len(r) == len(cells[0])]
    end = next((i for i, r in enumerate(body) if r == header), len(body))
    df = pd.DataFrame(body[:end], columns=header)
    return df.apply(pd.to_numeric, errors="ignore")


def git_stamp() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        # only changes that could affect the numbers count (NUMBERS.md itself does not)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "docs/charts", ":!docs/charts/NUMBERS.md"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + (" (with uncommitted changes in src/ or docs/charts/)" if dirty else "")
    except Exception:
        return "unknown"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def detection_loro() -> list[str]:
    t = read_md_table(CHARTS / "tune_results.md")
    out = ["## Detection, headline (leave-one-race-out, same false-alarm budget)", "",
           "Source: tune_results.md. For each training race the thresholds were chosen on the other races and "
           "scored on that race; the naive speed threshold was tuned the same way. Recall = official incidents "
           "matched (an alert within 60 s before to 10 s after the first official message, in a matching marshal "
           "sector, or a stopped car we were still flagging). Lead = official time minus our first alert.", ""]
    for b, g in t.groupby("budget_fa_h"):
        d, s = g[g["system"] == "detectors"].iloc[0], g[g["system"] == "baseline"].iloc[0]
        out.append(f"- Budget {b:g} false alarms per race hour: our detectors {pct(d['recall'])} of {int(d['incidents'])} "
                   f"incidents, median lead {d['median_lead_s']:.1f} s, earlier than race control in "
                   f"{pct(d['earlier_than_official'])}, {d['fa_h']:.1f} false alarms/h. Speed threshold: "
                   f"{pct(s['recall'])}, {s['median_lead_s']:.1f} s, {pct(s['earlier_than_official'])}, {s['fa_h']:.1f}/h.")
    return out + [""]


def detection_deployed() -> list[str]:
    ev = pd.read_csv(CHARTS / "detect_eval.csv", encoding="utf-8")
    al = pd.read_csv(CHARTS / "detect_alerts.csv", encoding="utf-8")
    al = al[al["system"] == "detectors"]
    out = ["## Detection as deployed (production settings plus ANOMALY alerts)", "",
           "Source: detect_eval.csv, detect_alerts.csv. Thresholds were selected on these same races, so these are "
           "not strictly out of sample (use the headline above for that); the ANOMALY model is leave-one-race-out.", ""]
    for system, label in (("detectors", "Our system"), ("baseline", "Untuned 50 km/h speed threshold")):
        g = ev[ev["system"] == system]
        inc, m, held = g["incidents"].sum(), g["matched"].sum(), g["held"].sum()
        hours, fa = g["hours"].sum(), g["false_alarms"].sum()
        out.append(f"- {label}: recall {pct(m / inc)} ({m} of {inc}, {held} by a still-held stopped-car alert), "
                   f"{fa} false alarms in {hours:.1f} race hours = {fa / hours:.1f}/h.")
    by_type = al.groupby("type")["false_alarm"].agg(["size", "sum"])
    out.append("- Our alerts by type (alerts / false alarms): " + ", ".join(
        f"{k} {int(v['size'])}/{int(v['sum'])}" for k, v in by_type.iterrows()) + ".")
    if "ANOMALY" in by_type.index:
        a = by_type.loc["ANOMALY"]
        out.append(f"- ANOMALY as an alert: {int(a['size'])} alerts, {int(a['sum'])} false. It roughly doubles the "
                   "false-alarm rate for about 2 extra incidents. Its measurable value is as a risk-model feature (below).")
    return out + [""]


def latency_from_onset() -> list[str]:
    ev = pd.read_csv(CHARTS / "latency_by_type.csv", encoding="utf-8")
    miss = pd.read_csv(CHARTS / "latency_by_type_no_onset.csv", encoding="utf-8")
    out = ["## Latency from crash onset: race control vs us", "",
           "Source: latency_by_type.csv / .md / .png. Onset = speed below 50% of the car's own speed at that point "
           "on its previous 3 clean laps, for 1 s (full rule in latency_by_type.md). Latency on crashes with a "
           "clear collapse, which is what our detectors are best at: not a recall figure.", ""]
    for f in FLAG_NAMES:
        g = ev[ev["flag"] == f]
        if g.empty:
            continue
        m = g[g["our_latency_s"].notna()]
        out.append(f"- {FLAG_NAMES[f]}: {len(g)} events, we caught {len(m)}. Race control median "
                   f"{g['rc_latency_s'].median():.1f} s after onset, us {m['our_latency_s'].median():.1f} s; "
                   f"we were earlier in {pct((m['lead_s'] > 0).mean())}.")
    reasons = "; ".join(f"{n} {r}" for r, n in miss.groupby("reason").size().sort_values(ascending=False).items())
    out.append(f"- {len(miss)} flag events had no identifiable onset car and are excluded, all listed: {reasons}.")
    return out + [""]


def risk() -> list[str]:
    t = pd.read_csv(CHARTS / "risk_eval.csv", encoding="utf-8")
    ew = pd.read_csv(CHARTS / "risk_early_warning.csv", encoding="utf-8")
    out = ["## Risk model (LightGBM, leave-one-race-out)", "",
           "Source: risk_eval.csv, risk_early_warning.csv, risk_importance.csv. PR-AUC over every racing tick. "
           "Precursor = the same without each incident's last 3 s, so the car already crashing does not count as "
           "predicting it: quote the precursor numbers.", ""]
    for h, g in t.groupby("horizon_s"):
        r = g.set_index("model")
        out.append(f"- {h} s: precursor PR-AUC {r.loc['lightgbm', 'pr_auc_precursor']:.4f} (without the ANOMALY feature "
                   f"{r.loc['lightgbm_without_anomaly', 'pr_auc_precursor']:.4f}; ANOMALY score alone "
                   f"{r.loc['anomaly_score', 'pr_auc_precursor']:.4f}; speed threshold "
                   f"{r.loc['speed_threshold', 'pr_auc_precursor']:.4f}; base rate {r.loc['lightgbm', 'base_rate']:.4f}). "
                   f"Headline PR-AUC {r.loc['lightgbm', 'pr_auc']:.3f}.")
    for _, e in ew[ew["model"] == "lightgbm"].iterrows():
        out.append(f"- High-risk line crossed by {100 * e['neg_tick_rate']:.1f}% of no-incident ticks: "
                   f"{pct(e['flagged_3s_before'])} of {int(e['car_incidents'])} car incidents flagged at least 3 s before "
                   f"detection (median {e['median_s_before_when_flagged']:.1f} s), {e['false_episodes_per_hour']:.0f} "
                   "false high-risk episodes per race hour.")
    out.append("- Say: a risk heat indicator, not an alarm. Some incidents have no precursor in the data.")
    return out + [""]


def pipeline_speed() -> list[str]:
    j = json.loads((CHARTS / "latency.json").read_text(encoding="utf-8"))
    tot = j["stages"]["total"]
    stages = ", ".join(f"{k} {v['mean_ms']:.2f} ms" for k, v in j["stages"].items() if k != "total")
    return ["## Pipeline speed", "", "Source: latency.json (python -m src.replay.benchmark).", "",
            f"- {j['ticks']} ticks of {j['race'].replace('_', ' ')} in {j['wall_seconds']} s = "
            f"{j['times_real_time']}x real time, stages: {', '.join(j['processors'])}.",
            f"- Per tick, from emission until its envelopes are sent: mean {tot['mean_ms']:.2f} ms, p95 "
            f"{tot['p95_ms']:.2f} ms, worst {tot['max_ms']:.1f} ms; {j['over_budget']} ticks over "
            f"{j['budget_ms']:.0f} ms. Mean per stage: {stages}.", ""]


def case_studies() -> list[str]:
    doc = json.loads((CHARTS / "case_studies.json").read_text(encoding="utf-8"))
    out = ["## Case studies", "", f"Source: case_studies.json and case_*.png. {doc['counterfactual']}", ""]
    for rid, r in doc["races"].items():
        out.append(f"{rid.replace('_', ' ')}: {r['out_of_sample']}")
        for c in r["crashes"]:
            tl = c["timeline"]
            rel = lambda t: f"{t - tl['onset']:+.1f} s" if t is not None else "n/a"  # noqa: E731
            our = tl["our_first_alert"]
            rec = tl["our_escalation_recommendation"]
            rec_txt = f"{rec['flag']} {rel(rec['t'])}" if isinstance(rec, dict) else str(rec).split(":")[0]
            out.append(f"- {c['driver']} (car {c['car']}): our first alert {rel(our['t']) if our else 'none'}"
                       f"{' (' + our['type'] + ')' if our else ''}, official yellow {rel(tl['official_yellow'])}, "
                       f"our escalation {rec_txt}, official {tl['official_escalation']['flag']} "
                       f"{rel(tl['official_escalation']['t'])} (times after onset).")
            for w in c["windows"]:
                cars = (w["note"].split(":")[0] if w["note"] else "pending") if w["cars_passing"] is None \
                    else f"{w['cars_passing']} cars"
                secs = "" if w["seconds"] is None else f", {w['seconds']:.1f} s"
                out.append(f"  - {w['name']}: {cars}{secs}")
    return out + [""]


def escalation() -> list[str]:
    from src.eval.escalation import impact_signal
    e = json.loads((CHARTS / "escalation.json").read_text(encoding="utf-8"))
    ours = pd.read_csv(CHARTS / "escalation_ours.csv", encoding="utf-8")
    ir = impact_signal(pd.read_csv(CHARTS / "escalation_official.csv", encoding="utf-8"))
    st, c = e["status"], e["our_by_category"]
    by = ", ".join(f"{FLAG_NAMES[f]} {v['official']}" for f, v in e["by_official_flag"].items())
    by_ours = ", ".join(f"{FLAG_NAMES[f]} {int((ours['flag'] == f).sum())}" for f in ("VSC", "SC", "RED"))
    return ["## Escalation scorecard: our VSC, SC and red recommendations vs race control", "",
            "Source: escalation.json, escalation.md and escalation.png (python -m src.eval.escalation). In-sample: the "
            "detector settings were tuned and the race control rules set on these races. Counterfactual: assumes race "
            "control acted on our recommendation at once.", "",
            f"- {e['official_escalations']} official escalations in {e['races']} training races ({by}). We recommended "
            f"a VSC, SC or red for {e['matched']} ({pct(e['matched'] / e['official_escalations'])}), {st['earlier']} of "
            f"them earlier than the race control feed; median lead {e['median_lead_s']:.1f} s (middle half "
            f"{e['lead_iqr_s'][0]:.1f} to {e['lead_iqr_s'][1]:.1f} s).",
            f"- Missed {st['missed']}: {e['missed_kind']['car collapse seen']} with a car collapse we could see, "
            f"{e['missed_kind']['no car collapse']} without one (re-flags of an earlier stoppage, debris, weather, lap 1).",
            f"- Escalations race control never made: {e['extra']} in {e['race_hours']:.1f} race hours = "
            f"{e['extra_per_hour']:.2f} per race hour ({c['official yellows only']} where race control kept yellows "
            f"only, {c['no official flag']} with no official flag).",
            f"- Flag choice: same first flag as race control in {e['same_first_flag']} of {e['matched']}; our "
            f"{len(ours)} recommendations by flag: {by_ours}. FastF1 puts stopped cars on the racing line, so the "
            f"lateral offset cannot tell VSC from SC; an impact can: \"SC after an impact, VSC otherwise\" picks race "
            f"control's flag {ir['right_with_impact_rule']} of {ir['escalations_with_onset_car']} times, \"always SC\" "
            f"{ir['right_if_always_sc']}.",
            ""]


def holdout() -> list[str]:
    runs = sorted(CHARTS.glob("holdout_*.json"))
    out = ["## Holdout (2026 Azerbaijan, 2026 Madrid)", ""]
    if not runs:
        return out + ["Not run yet: A7, once, after the freeze.", ""]
    for path in runs:
        r = json.loads(path.read_text(encoding="utf-8"))
        fz = r.get("freeze", {})
        out.append(f"{r['race'].replace('_', ' ')} (source: {path.name}): run {len(r.get('runs', []))} time(s), code and "
                   f"models frozen at commit {str(fz.get('commit', '?'))[:7]}; thresholds frozen, nothing fitted on it.")
        for name, d in r["detection"].items():
            out.append(f"- Detection, {name.replace('_', ' ')}: recall {pct(d['recall']) if d['recall'] is not None else 'n/a'} "
                       f"({d['matched']} of {d['incidents']}), {d['fa_per_hour']:.1f} false alarms/h" if d["fa_per_hour"] is not None
                       else f"- Detection, {name.replace('_', ' ')}: n/a")
        for row in r["latency_from_onset"]:
            out.append(f"- {FLAG_NAMES.get(row['flag'], row['flag'])}: race control {row['rc_median_s']:.1f} s after onset, "
                       f"us {row['ours_median_s']:.1f} s ({row['matched']} of {row['events']} events)")
        for h, v in r["risk"]["horizons"].items():
            p = v.get("lightgbm_pr_auc_precursor")
            out.append(f"- Risk {h} s: precursor PR-AUC {'n/a' if p is None else f'{p:.4f}'} (base rate {v['base_rate']:.4f})")
        if "escalation" in r:
            e = r["escalation"]
            lead = "n/a" if e["median_lead_s"] is None else f"{e['median_lead_s']:.1f} s"
            out.append(f"- Escalations: {e['official_escalations']} official, we recommended {e['matched']} "
                       f"({e['status']['earlier']} earlier, median lead {lead}), {e['extra']} extra "
                       f"({e['extra_per_hour']:.2f} per race hour)")
        for c in r["case_study"]["crashes"]:
            w3 = next((w for w in c["windows"] if w["name"].startswith("W3")), None)
            if w3 is not None:
                out.append(f"- {c['driver']} crash, W3 (our escalation to the official one): "
                           f"{w3['cars_passing'] if w3['cars_passing'] is not None else 'n/a'} cars")
        out.append("")
    return out


def build() -> str:
    head = ["# NUMBERS: the single source of truth for the pitch", "",
            f"Generated by `python -m src.eval.numbers` on {datetime.now():%a %d %b %Y %H:%M} from the files in "
            f"docs/charts/ at commit {git_stamp()}. Do not edit by hand: rerun the script after any evaluation change.", "",
            "## How to quote these", "",
            "- Every number comes from a replay of historical FastF1 data. Say so.",
            "- Say \"earlier than the race control feed\" (the official messages), never \"earlier than the marshals\".",
            "- Training data: 20 races from 2023 to 2026. Holdout races (2026 Azerbaijan, 2026 Madrid): no numbers "
            "until A7, after the code freeze.",
            "- Risk is a heat indicator, not an alarm. Case-study windows are counterfactuals.", ""]
    body = (detection_loro() + detection_deployed() + latency_from_onset() + risk() + pipeline_speed()
            + escalation() + case_studies())
    return "\n".join(head + body + holdout())


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

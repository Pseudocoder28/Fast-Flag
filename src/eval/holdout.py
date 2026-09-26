"""A7 holdout: freeze the code and models, then run the whole pipeline once on a holdout race.

Run:
  python -m src.eval.holdout freeze
  python -m src.eval.holdout run --race 2026_Azerbaijan
  python -m src.eval.holdout run --race 2026_Azerbaijan --rerun-reason "..."   (only if it already ran)
  python -m src.eval.holdout dry-run --race 2021_Azerbaijan                   (any non-holdout race)

What it enforces (PROJECT_BRIEF.md 4.4, CLAUDE.md rules 3 and 4):
- A holdout race is loaded only by `run`, and only after `freeze`.
- `run` refuses if anything under src/ differs from the frozen commit (committed or not), or if any
  model or settings file in data/models changed since the freeze.
- Every threshold is the frozen one: detector settings, the ANOMALY alert threshold and the risk
  "high risk" line. Nothing is fitted or tuned on the holdout.
- A race runs once. A later run needs --rerun-reason, which goes into the run log and the report.
  The run is logged before evaluating, so an interrupted run still counts.

Outputs: `run` writes docs/charts/holdout_<race>.md and .json plus case_<race>_car<N>.png;
`dry-run` writes the same files to data/dry_run/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.ingest.holdout import is_holdout_id
from src.replay.engine import HOLDOUT_DIR, Engine, RaceData, available_races

MODELS = Path("data") / "models"
FREEZE = MODELS / "freeze.json"
MODEL_FILES = ("anomaly.joblib", "detector_config.json", "risk_10.txt", "risk_20.txt", "risk_30.txt",
               "risk_meta.json")
CHARTS = Path("docs") / "charts"
DRY_DIR = Path("data") / "dry_run"
EARLY_S = 3.0


# ---- frozen state

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def model_hashes(models: Path = MODELS) -> dict[str, str]:
    return {name: sha256(models / name) for name in MODEL_FILES if (models / name).exists()}


def src_changes(since: str | None = None) -> list[str]:
    """Files under src/ that are uncommitted, or that differ from commit `since`."""
    dirty = [ln[3:].strip() for ln in git("status", "--porcelain", "--", "src").splitlines() if ln.strip()]
    committed = git("diff", "--name-only", since, "HEAD", "--", "src").splitlines() if since else []
    return sorted(set(dirty) | {c for c in committed if c})


def verify(manifest: dict, changes: list[str], hashes: dict[str, str]) -> list[str]:
    """Problems with the frozen state (an empty list means it is intact)."""
    problems = []
    if changes:
        problems.append(f"src/ differs from the frozen commit {manifest['commit'][:7]}: " + ", ".join(changes))
    for name, h in manifest["models"].items():
        if hashes.get(name) != h:
            problems.append(f"data/models/{name} changed since the freeze")
    return problems


def check_once(runs: list[dict], reason: str | None) -> str | None:
    """A holdout race runs once: a second run needs a written reason."""
    if runs and not reason:
        last = runs[-1]
        return (f"this race already ran on {last['at']} (frozen commit {last['commit'][:7]}). A holdout runs once; "
                "rerun only with --rerun-reason, which is recorded in the report.")
    return None


def runs_log(rid: str) -> Path:
    return HOLDOUT_DIR / f"{rid}_runs.json"


def freeze(reason: str | None = None) -> dict:
    changes = src_changes()
    if changes:
        raise SystemExit("commit the code under src/ before freezing: " + ", ".join(changes))
    done = sorted(p.name for p in HOLDOUT_DIR.glob("*_runs.json")) if HOLDOUT_DIR.exists() else []
    if done and not reason:
        raise SystemExit(f"a holdout already ran ({', '.join(done)}); re-freezing now needs --reason, "
                         "which is recorded in the manifest")
    from src.detect.pipeline import load_anomaly, load_config
    meta = json.loads((MODELS / "risk_meta.json").read_text(encoding="utf-8"))
    manifest = {"frozen_at": datetime.now().isoformat(timespec="seconds"), "commit": git("rev-parse", "HEAD"),
                "models": model_hashes(), "training_races": available_races(),
                "anomaly_threshold": load_anomaly().threshold, "risk_threshold_p30": meta.get("threshold_p30"),
                "detector_config": asdict(load_config()), "reason": reason}
    FREEZE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---- evaluation (the same code path for run and dry-run)

def summarise(r) -> dict:
    return {"incidents": r.incidents, "matched": r.matched, "held": r.held,
            "recall": r.matched / r.incidents if r.incidents else None,
            "recall_window": (r.matched - r.held) / r.incidents if r.incidents else None,
            "median_lead_s": float(np.median(r.lead_times)) if r.lead_times else None,
            "earlier_share": float(np.mean([x > 0 for x in r.lead_times])) if r.lead_times else None,
            "false_alarms": r.false_alarms, "race_hours": r.hours,
            "fa_per_hour": r.false_alarms / r.hours if r.hours else None}


def pr_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(average_precision_score(y, s)) if y.any() else None


def risk_section(race: RaceData, dets: list[dict]) -> dict:
    """Frozen production risk models on every racing tick; the high-risk line is the frozen one."""
    from src.predict.dataset import labelled_rows
    from src.predict.pipeline import load_models
    from src.predict.processor import correct
    from src.predict.risk import early_warning, incident_end
    boosters, meta = load_models()
    df, x, info = labelled_rows(race, dets)
    feats = meta["features"]
    pred = pd.DataFrame({"race": race.race, "drv": df["drv"].to_numpy(), "t": df["t"].to_numpy(),
                         "y10": df["y10"].to_numpy(), "y30": df["y30"].to_numpy(), "speed": df["speed"].to_numpy(),
                         "anomaly_score": x["anomaly_score"].to_numpy(),
                         "p10": correct(boosters[10].predict(x[feats]), meta["neg_sample"]),
                         "p30": correct(boosters[30].predict(x[feats]), meta["neg_sample"])})
    t_inc = incident_end(pred)
    precursor = ~((pred["t"] >= t_inc - EARLY_S) & (pred["t"] < t_inc)).to_numpy()
    out = {"cars_with_incident": info["cars_with_incident"], "ticks": len(pred), "horizons": {}}
    for h in (10, 30):
        y, p = pred[f"y{h}"].to_numpy(), pred[f"p{h}"].to_numpy()
        scores = {"lightgbm": p, "anomaly_score": pred["anomaly_score"].fillna(0).to_numpy(),
                  "speed_threshold": -pred["speed"].to_numpy()}
        out["horizons"][h] = {"base_rate": float(y.mean()), **{
            f"{name}_{kind}": pr_auc(y[m], s[m]) for name, s in scores.items()
            for kind, m in (("pr_auc", np.ones(len(y), bool)), ("pr_auc_precursor", precursor))}}
    if info["cars_with_incident"]:
        hours = len(pred) / 4 / 3600 / max(pred["drv"].nunique(), 1)
        out["early_warning"] = early_warning(pred, float(meta["threshold_p30"]), hours, col="p30")
    return out


def evaluate(rid: str, out_dir: Path, allow_holdout: bool, label: str = "", scope_note: str = "") -> dict:
    """Every holdout section for one race, plus its charts in out_dir. label names the race on
    the charts (e.g. "2026 Azerbaijan holdout"); scope_note says what kind of test it is."""
    from src.detect.detectors import NaiveThreshold
    from src.eval.case_study import crashes, plot as plot_crash
    from src.eval.escalation import (plot as plot_escalation, production_suite, race_scorecard,
                                     summarise as summarise_escalation, tables)
    from src.eval.incidents import INCIDENT_FLAGS
    from src.eval.latency_by_type import plot as plot_latency, race_events, summary
    from src.eval.run import advisories, rule_alerts, score
    from src.replay.pipeline import all_processors
    label = label or rid.replace("_", " ")
    race = RaceData.load(rid)
    eng = Engine(race, all_processors(race))               # frozen detectors, ANOMALY and risk
    dets = [e["data"] for e in eng.advance(eng.t_end) if e["kind"] == "detection"]
    base_eng = Engine(race, [NaiveThreshold()])
    base = [e["data"] for e in base_eng.advance(base_eng.t_end) if e["kind"] == "detection"]
    detection = {"rule_alerts": summarise(score(race, "detectors", rule_alerts(dets))),
                 "anomaly_advisories": summarise(score(race, "anomaly_advisory", advisories(dets))),
                 "naive_speed_threshold": summarise(score(race, "baseline", base))}
    events, missing = race_events(rid, dets=dets, allow_holdout=allow_holdout)
    ev = pd.DataFrame(events)
    by_flag = summary(ev, "flag", list(INCIDENT_FLAGS)) if len(ev) else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = []
    pd.DataFrame(missing).to_csv(out_dir / f"latency_{rid}_no_onset.csv", index=False, encoding="utf-8")
    if len(ev):
        plot_latency(ev, by_flag, 1, len(missing), out_dir / f"latency_{rid}.png",
                     subtitle=f"Replay of historical FastF1 data, {label}: {len(ev)} flag events with an "
                              f"identifiable crash onset", source=f"latency_{rid}_no_onset.csv")
        charts.append(f"latency_{rid}.png")
    found, rec_source = crashes(rid, allow_holdout=allow_holdout)
    for c in found:
        plot_crash(c, out_dir / f"case_{rid}_car{c['car']}.png")
        charts.append(f"case_{rid}_car{c['car']}.png")
    card = race_scorecard(rid, suite_for=production_suite, allow_holdout=allow_holdout)
    off, ours = tables([card])
    esc = summarise_escalation([card])
    off.to_csv(out_dir / f"escalation_{rid}_official.csv", index=False, encoding="utf-8")
    ours.to_csv(out_dir / f"escalation_{rid}_ours.csv", index=False, encoding="utf-8")
    plot_escalation(off, esc, out_dir / f"escalation_{rid}.png",
                    title=f"Escalation scorecard, {label}: {esc['official_escalations']} official VSC, SC and red flags",
                    scope_note=scope_note or label)
    charts.append(f"escalation_{rid}.png")
    return {"race": rid, "detection": detection, "latency_from_onset": by_flag.to_dict("records"),
            "latency_events": len(ev), "latency_no_onset": len(missing), "risk": risk_section(race, dets),
            "case_study": {"recommendation_source": rec_source, "crashes": found},
            "escalation": esc, "charts": charts}


# ---- report

def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def num(x: float | None, fmt: str = ".2f") -> str:
    return "n/a" if x is None else format(x, fmt)


def report_md(res: dict, header: list[str]) -> str:
    lines = header + ["", "## Detection", ""]
    for name, d in res["detection"].items():
        lines.append(f"- {name.replace('_', ' ')}: recall {pct(d['recall'])} ({d['matched']} of {d['incidents']} "
                     f"incidents, {d['held']} held), median lead {num(d['median_lead_s'], '.1f')} s, earlier than race "
                     f"control in {pct(d['earlier_share'])}, {d['false_alarms']} false alarms in "
                     f"{num(d['race_hours'], '.2f')} race hours = {num(d['fa_per_hour'], '.1f')}/h")
    lines += ["", f"## Latency from crash onset ({res['latency_events']} events with an onset car, "
                  f"{res['latency_no_onset']} without)", ""]
    for r in res["latency_from_onset"]:
        lines.append(f"- {r['flag']}: {r['events']} events, caught {r['matched']}, race control median "
                     f"{num(r['rc_median_s'], '.1f')} s, us {num(r['ours_median_s'], '.1f')} s, earlier in "
                     f"{pct(r['earlier_share'])}")
    rk = res["risk"]
    lines += ["", f"## Risk ({rk['cars_with_incident']} car incidents, {rk['ticks']} racing ticks)", ""]
    for h, v in rk["horizons"].items():
        lines.append(f"- {h} s: precursor PR-AUC {num(v['lightgbm_pr_auc_precursor'], '.4f')} (ANOMALY score "
                     f"{num(v['anomaly_score_pr_auc_precursor'], '.4f')}, speed threshold "
                     f"{num(v['speed_threshold_pr_auc_precursor'], '.4f')}, base rate {v['base_rate']:.4f}); "
                     f"headline {num(v['lightgbm_pr_auc'], '.3f')}")
    if "early_warning" in rk:
        e = rk["early_warning"]
        lines.append(f"- Frozen high-risk line ({e['threshold_p30']:.4f}): {pct(e['flagged_3s_before'])} of "
                     f"{e['car_incidents']} car incidents flagged at least {EARLY_S:.0f} s before detection, "
                     f"{e['false_episodes_per_hour']:.0f} false high-risk episodes per race hour")
    if "escalation" in res:
        e = res["escalation"]
        lines += ["", f"## Escalations (VSC, SC, red): {e['official_escalations']} official", "",
                  f"- We recommended a VSC, SC or red for {e['matched']}: {e['status']['earlier']} earlier, "
                  f"{e['status']['later']} later, median lead {num(e['median_lead_s'], '.1f')} s. Missed "
                  f"{e['status']['missed']}, already out {e['status']['already out']}.",
                  f"- Extra escalations race control never made: {e['extra']} = {num(e['extra_per_hour'], '.2f')} "
                  f"per race hour. Same first flag as race control: {e['same_first_flag']} of {e['matched']}."]
    if res.get("charts"):
        lines += ["", "## Charts", "", *[f"- {c}" for c in res["charts"]]]
    lines += ["", "## Crashes (timelines and exposure)", ""]
    for c in res["case_study"]["crashes"]:
        tl = c["timeline"]
        rel = lambda t: "n/a" if t is None else f"{t - tl['onset']:+.1f} s"  # noqa: E731
        rec = tl["our_escalation_recommendation"]
        rec_txt = f"{rec['flag']} {rel(rec['t'])}" if isinstance(rec, dict) else str(rec).split(":")[0]
        our = tl["our_first_alert"]
        lines.append(f"- {c['driver']} (car {c['car']}): our first alert {rel(our['t']) if our else 'none'}, official "
                     f"yellow {rel(tl['official_yellow'])}, our escalation {rec_txt}, official "
                     f"{tl['official_escalation']['flag']} {rel(tl['official_escalation']['t'])}")
        for w in c["windows"]:
            lines.append(f"  - {w['name']}: {w['cars_passing'] if w['cars_passing'] is not None else 'n/a'} cars")
    return "\n".join(lines) + "\n"


def write(res: dict, out_dir: Path, header: list[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"holdout_{res['race']}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    path = out_dir / f"holdout_{res['race']}.md"
    path.write_text(report_md(res, header), encoding="utf-8")
    return path


# ---- commands

def run(rid: str, reason: str | None) -> Path:
    if not is_holdout_id(rid):
        raise SystemExit(f"{rid} is not a holdout race: use dry-run")
    if not FREEZE.exists():
        raise SystemExit("freeze first: python -m src.eval.holdout freeze")
    manifest = json.loads(FREEZE.read_text(encoding="utf-8"))
    problems = verify(manifest, src_changes(manifest["commit"]), model_hashes())
    if problems:
        raise SystemExit("refusing: the frozen state changed:\n- " + "\n- ".join(problems))
    log = runs_log(rid)
    runs = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    refusal = check_once(runs, reason)
    if refusal:
        raise SystemExit(refusal)
    if not (HOLDOUT_DIR / f"{rid}_meta.json").exists():
        from src.ingest.build import build_race
        build_race(rid, holdout=True)
    runs.append({"at": datetime.now().isoformat(timespec="seconds"), "commit": manifest["commit"], "reason": reason})
    log.write_text(json.dumps(runs, indent=2), encoding="utf-8")        # logged before anything is seen
    race = rid.replace("_", " ")
    res = evaluate(rid, CHARTS, allow_holdout=True, label=f"{race} holdout",
                   scope_note=f"Holdout: code and models frozen at commit {manifest['commit'][:7]}; nothing was fitted "
                              "or tuned on this race.")
    res |= {"freeze": {k: manifest[k] for k in ("frozen_at", "commit", "anomaly_threshold", "risk_threshold_p30")},
            "runs": runs}
    header = [f"# Holdout: {rid}", "",
              f"Replay of historical FastF1 data. Run {len(runs)} of this race, at {runs[-1]['at']}, with code and "
              f"models frozen at commit {manifest['commit'][:7]} ({manifest['frozen_at']}). Every threshold is the "
              "frozen one; nothing was fitted on this race."]
    header += [f"- Earlier run {i + 1}: {r['at']}" + (f", rerun reason: {r['reason']}" if r["reason"] else "")
               for i, r in enumerate(runs[:-1])]
    if reason:
        header.append(f"- This run's reason: {reason}")
    return write(res, CHARTS, header)


def dry_run(rid: str) -> Path:
    if is_holdout_id(rid):
        raise SystemExit(f"{rid} is a holdout race: dry-run is for non-holdout races only")
    res = evaluate(rid, DRY_DIR, allow_holdout=False, label=f"{rid.replace('_', ' ')} dry run",
                   scope_note="Dry run on a race outside training and tuning, with the current models: not the holdout.")
    header = [f"# DRY RUN: {rid} (not a holdout)", "",
              "Replay of historical FastF1 data. Same pipeline and code path as the A7 holdout run, with the "
              "current models; written to data/dry_run/, never to docs/charts/."]
    return write(res, DRY_DIR, header)


def main() -> None:
    p = argparse.ArgumentParser(description="A7 holdout: freeze, one-shot run, dry run")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--reason", help="required to re-freeze after a holdout already ran")
    r = sub.add_parser("run")
    r.add_argument("--race", required=True)
    r.add_argument("--rerun-reason")
    d = sub.add_parser("dry-run")
    d.add_argument("--race", required=True)
    a = p.parse_args()
    if a.cmd == "freeze":
        m = freeze(a.reason)
        print(f"frozen at commit {m['commit'][:7]} with {len(m['models'])} model files -> {FREEZE}")
    elif a.cmd == "run":
        print(f"wrote {run(a.race, a.rerun_reason)}")
    else:
        print(f"wrote {dry_run(a.race)}")


if __name__ == "__main__":
    from src.eval.holdout import main as entry    # run through the package import
    entry()

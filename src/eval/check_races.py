"""Check races: a one-shot, out-of-sample test of the race control rules and detector
settings that were tuned on the 20 training races.

The check races are the next 20 in data/race_ranking.csv after the training 20, skipping
the holdout, built with python -m src.ingest.build --case into data/case_studies. Nothing
was fitted or tuned on them: the detectors and the ANOMALY model are the production ones,
trained on the training races (as in the A7 holdout run), and the scorecard is
src.eval.escalation. 2021 Azerbaijan is not one of them: the late-race red was checked on
it and the demo uses it.

Like the holdout, it runs once, on committed code: each run is logged (time, commit,
model hashes) in docs/charts/escalation_check_runs.json before anything is scored, and a
second run needs --rerun-reason, which the report shows. A rule changed after these
results were seen makes them in-sample: say so.

Run: python -m src.eval.check_races
     python -m src.eval.check_races --rerun-reason "why"
Writes docs/charts/escalation_check.md, .json and .png, escalation_check_official.csv and
escalation_check_ours.csv.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

from src.eval.escalation import CHARTS, plot, production_suite, race_scorecard, summarise, tables, write_report
from src.eval.holdout import git, model_hashes, src_changes
from src.ingest.holdout import assert_not_holdout
from src.replay.engine import CASE_DIR, available_races

CHECK_RACES = ("2023_Las_Vegas", "2023_Singapore", "2026_Monaco", "2024_Azerbaijan", "2026_Austrian", "2023_Dutch",
               "2024_Monaco", "2024_Australian", "2024_Chinese", "2026_Chinese", "2023_Japanese", "2026_Barcelona",
               "2025_São_Paulo", "2023_Azerbaijan", "2024_Abu_Dhabi", "2023_Canadian", "2023_São_Paulo",
               "2025_Las_Vegas", "2025_Saudi_Arabian", "2025_Emilia_Romagna")
NAME = "escalation_check"
RUNS = CHARTS / f"{NAME}_runs.json"


def problems(races: tuple[str, ...] = CHECK_RACES) -> list[str]:
    """Why the check cannot run now (an empty list means it can)."""
    out = []
    training = set(available_races())
    for rid in races:
        assert_not_holdout(rid=rid)
        if rid in training:
            out.append(f"{rid} is a training race")
        elif not (CASE_DIR / f"{rid}_meta.json").exists():
            out.append(f"{rid} is not built: python -m src.ingest.build {rid} --case")
    changes = src_changes()
    if changes:
        out.append("commit the code under src/ first: " + ", ".join(changes))
    return out


def refusal(runs: list[dict], reason: str | None) -> str | None:
    """The check runs once: a second run needs a written reason."""
    if runs and not reason:
        return (f"the check races already ran on {runs[-1]['at']} (commit {runs[-1]['commit'][:7]}). They run once; "
                "rerun only with --rerun-reason, which the report shows.")
    return None


def head(summ: dict, runs: list[dict]) -> list[str]:
    last = runs[-1]
    lines = ["# Escalation scorecard (check races)", "",
             f"Replay of historical FastF1 data: {summ['races']} check races, {summ['race_hours']:.1f} race hours, the "
             "next 20 in data/race_ranking.csv after the training races (the holdout skipped). Our side is the race "
             "control engine (src/racecontrol) fed by the production detectors and ANOMALY model, trained on the "
             "training races. Definitions: src/eval/escalation.py.", "",
             scope_note(runs)]
    for i, r in enumerate(runs[:-1]):
        why = f", rerun reason: {r['reason']}" if r["reason"] else ""
        lines.append(f"- Earlier run {i + 1}: {r['at']}, commit {r['commit'][:7]}{why}")
    if last["reason"]:
        lines.append(f"- This run's reason: {last['reason']}")
    return lines


def scope_note(runs: list[dict]) -> str:
    last = runs[-1]
    return (f"Out of sample: run {len(runs)}, at {last['at'][:16].replace('T', ' ')}, on commit {last['commit'][:7]}. "
            "No rule or model was fitted or tuned on these races.")


def main() -> None:
    p = argparse.ArgumentParser(description="One-shot escalation scorecard on the check races")
    p.add_argument("--rerun-reason", help="required for a second run; shown in the report")
    a = p.parse_args()
    found = problems()
    if found:
        raise SystemExit("refusing:\n- " + "\n- ".join(found))
    runs = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else []
    stop = refusal(runs, a.rerun_reason)
    if stop:
        raise SystemExit(stop)
    runs.append({"at": datetime.now().isoformat(timespec="seconds"), "commit": git("rev-parse", "HEAD"),
                 "reason": a.rerun_reason, "models": model_hashes()})
    CHARTS.mkdir(parents=True, exist_ok=True)
    RUNS.write_text(json.dumps(runs, indent=2), encoding="utf-8")          # logged before anything is scored
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(partial(race_scorecard, suite_for=production_suite), CHECK_RACES))
    off, ours = tables(results)
    summ = summarise(results) | {"check_races": list(CHECK_RACES), "runs": runs}
    off.to_csv(CHARTS / f"{NAME}_official.csv", index=False, encoding="utf-8")
    ours.to_csv(CHARTS / f"{NAME}_ours.csv", index=False, encoding="utf-8")
    (CHARTS / f"{NAME}.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    (CHARTS / f"{NAME}.md").write_text(write_report(off, ours, summ, results, head(summ, runs), NAME), encoding="utf-8")
    plot(off, summ, CHARTS / f"{NAME}.png",
         title=f"Check races: {summ['official_escalations']} official VSC, SC and red flags in {summ['races']} races "
               "never used for tuning",
         scope_note=scope_note(runs))
    print(f"wrote {CHARTS / (NAME + '.md')}: matched {summ['matched']} of {summ['official_escalations']}, "
          f"extras {summ['extra_per_hour']} per race hour")


if __name__ == "__main__":
    main()

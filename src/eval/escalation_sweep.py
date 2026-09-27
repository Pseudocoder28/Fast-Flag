"""Sensitivity of the escalation scorecard to one race control engine setting, on the
training races (the holdouts are never loaded here). Every value reruns the whole
scorecard (src.eval.escalation) with the setting overridden inside the worker
processes only; src/racecontrol/engine.py itself is never edited. In-sample by
construction: the A7 holdout run tested the engine frozen before the 26 Sept race control
changes, and those changes have no out-of-sample test.

Run: python -m src.eval.escalation_sweep SC_STOPPED_HOLD_S 3 5 8 10
     python -m src.eval.escalation_sweep SC_STOPPED_HOLD_S 3 5 --out /tmp/sweep
Writes <out>/escalation_sweep_<SETTING>.csv and .md (default out: docs/charts).
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from src.eval.escalation import CHARTS, ESCALATIONS, RED_UNCALLED, race_scorecard, summarise
from src.replay.engine import available_races


def scorecard_with(job: tuple[str, float, str]) -> dict:
    """One race's scorecard with one engine setting overridden in this worker process."""
    import src.racecontrol.engine as engine
    name, value, rid = job
    if not hasattr(engine, name):
        raise AttributeError(f"src.racecontrol.engine has no setting {name}")
    setattr(engine, name, value)
    return race_scorecard(rid)


def row(name: str, value: float, summ: dict) -> dict:
    iqr = summ["lead_iqr_s"] or [None, None]
    c = summ["our_by_category"]
    return {name: value, "official": summ["official_escalations"], "matched": summ["matched"],
            "earlier": summ["status"]["earlier"], "later": summ["status"]["later"], "missed": summ["status"]["missed"],
            "median_lead_s": summ["median_lead_s"], "lead_p25_s": iqr[0], "lead_p75_s": iqr[1],
            "same_first_flag": summ["same_first_flag"],
            **{f"our_{f.lower()}": v for f, v in summ["our_by_flag"].items() if f in ESCALATIONS},
            "red_both": summ["red_check"]["both"], "red_race_control_only": summ["red_check"]["race_control_only"],
            "red_ours_only": summ["red_check"]["ours_only"], "red_median_lead_s": summ["red_check"]["median_lead_s"],
            "our_reds": summ["our_reds"]["total"], "our_reds_uncalled": summ["our_reds"]["without_race_control_red"],
            "extra": summ["extra"], "extra_per_hour": summ["extra_per_hour"],
            "extra_yellows_only": c["official yellows only"], "extra_no_official_flag": c["no official flag"],
            "extra_outside_window": c["escalated incident, outside the match window"],
            "extra_red_uncalled": c[RED_UNCALLED], "track_clears_under_neutral": summ["track_clears_under_neutral"]}


def report(df: pd.DataFrame, name: str, hours: float) -> str:
    lines = [f"# Escalation scorecard vs {name} (training races)", "",
             f"Replay of historical FastF1 data, {len(available_races())} training races, {hours:.1f} race hours. "
             f"Each row reruns python -m src.eval.escalation with src.racecontrol.engine.{name} set to that value "
             "(in the worker processes only). In-sample. The A7 holdout run tested the engine frozen before the 26 Sept "
             "race control changes; those changes were checked on replays of the holdout, so they have no out-of-sample "
             "test.",
             "", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join("" if pd.isna(v) else f"{v:g}" if isinstance(v, float) else str(v) for v in r) + " |"
              for r in df.itertuples(index=False)]
    return "\n".join(lines + [""])


def main() -> None:
    p = argparse.ArgumentParser(description="Escalation scorecard for several values of one race control setting")
    p.add_argument("setting", help="a module-level constant of src.racecontrol.engine, e.g. SC_STOPPED_HOLD_S")
    p.add_argument("values", nargs="+", type=float)
    p.add_argument("--out", type=Path, default=CHARTS)
    a = p.parse_args()
    races = available_races()
    rows, hours = [], 0.0
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        for v in a.values:
            results = list(ex.map(scorecard_with, [(a.setting, v, r) for r in races]))
            summ = summarise(results)
            hours = summ["race_hours"]
            rows.append(row(a.setting, v, summ))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / f"escalation_sweep_{a.setting}.csv", index=False, encoding="utf-8")
    (a.out / f"escalation_sweep_{a.setting}.md").write_text(report(df, a.setting, hours), encoding="utf-8")
    print(f"wrote {a.out / f'escalation_sweep_{a.setting}.md'}")


if __name__ == "__main__":
    main()

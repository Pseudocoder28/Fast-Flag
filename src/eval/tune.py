"""Tune detector thresholds with leave-one-race-out on training races only (A4).

Every candidate setting is run on every training race once. Then, for each race,
the setting is chosen on the OTHER races (best matched incidents with false alarms
per race hour within a budget) and scored on the held-out race. The same selection
is done for the naive speed-threshold baseline, so both are compared at the same
false-alarm budget. The holdout race is never loaded.

Run: python -m src.eval.tune                   (budgets 2, 3 and 5 false alarms per hour)
     python -m src.eval.tune --budget 3
Writes docs/charts/tune_results.md and data/models/detector_config.json (setting
chosen on all training races, used by the server).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.detect.detectors import Config, DetectorSuite, NaiveThreshold
from src.eval.run import score
from src.ingest.holdout import assert_not_holdout
from src.replay.engine import Engine, RaceData, available_races

CHARTS = Path("docs/charts")
CONFIG_PATH = Path("data/models/detector_config.json")
BUDGETS = (2.0, 3.0, 5.0)


def detector_grid() -> list[dict]:
    grid = []
    for stop, impact in itertools.product((0.3, 0.4), (60.0, 80.0)):
        base = {"stop_ratio": stop, "impact_drop_kmh": impact}
        grid.append({**base, "slowdown_lap_ratio": None})
        for lap, rel, sus in itertools.product((0.35, 0.5), (0.4, 0.55), (1.0, 2.0)):
            grid.append({**base, "slowdown_lap_ratio": lap, "slowdown_rel_ratio": rel, "slowdown_sustain_s": sus})
    return grid


BASELINE_KMH = (20.0, 30.0, 40.0, 50.0)


def run_one(rid: str) -> list[dict]:
    """Every candidate on one race: per-incident matches, lead times, false alarms."""
    assert_not_holdout(rid=rid)
    race = RaceData.load(rid)
    n = len(race.track.get("msectors", [])) or None
    out = []
    cands = [("detectors", i, DetectorSuite(replace(Config(), **p), n_sectors=n)) for i, p in enumerate(detector_grid())]
    cands += [("baseline", i, NaiveThreshold(kmh)) for i, kmh in enumerate(BASELINE_KMH)]
    for system, idx, proc in cands:
        eng = Engine(race, [proc])
        dets = [e["data"] for e in eng.advance(eng.t_end) if e["kind"] == "detection"]
        r = score(race, system, dets)
        out.append({"race": rid, "system": system, "cand": idx, "incidents": r.incidents, "matched": r.matched,
                    "held": r.held, "leads": r.lead_times, "fa": r.false_alarms, "hours": r.hours})
    return out


def choose(df: pd.DataFrame, budget: float) -> int:
    """Candidate with the most matched incidents within the false-alarm budget
    (ties: fewer false alarms). Falls back to the fewest false alarms."""
    agg = df.groupby("cand").agg(matched=("matched", "sum"), fa=("fa", "sum"), hours=("hours", "sum"))
    agg["fa_h"] = agg["fa"] / agg["hours"]
    ok = agg[agg["fa_h"] <= budget]
    if not len(ok):
        return int(agg["fa_h"].idxmin())
    return int(ok.sort_values(["matched", "fa_h"], ascending=[False, True]).index[0])


def loro(df: pd.DataFrame, system: str, budget: float) -> dict:
    d = df[df["system"] == system]
    picked = []
    for rid in d["race"].unique():
        cand = choose(d[d["race"] != rid], budget)
        picked.append(d[(d["race"] == rid) & (d["cand"] == cand)].iloc[0])
    p = pd.DataFrame(picked)
    leads = [x for ls in p["leads"] for x in ls]
    return {"system": system, "budget_fa_h": budget, "incidents": int(p["incidents"].sum()),
            "matched": int(p["matched"].sum()), "recall": p["matched"].sum() / p["incidents"].sum(),
            "recall_window": (p["matched"].sum() - p["held"].sum()) / p["incidents"].sum(),
            "median_lead_s": float(np.median(leads)) if leads else np.nan,
            "earlier_than_official": float(np.mean([x > 0 for x in leads])) if leads else np.nan,
            "fa_h": p["fa"].sum() / p["hours"].sum()}


def main() -> None:
    p = argparse.ArgumentParser(description="Leave-one-race-out threshold tuning")
    p.add_argument("--budget", type=float, action="append", help="false alarms per race hour (repeatable)")
    a = p.parse_args()
    budgets = tuple(a.budget) if a.budget else BUDGETS
    races = available_races()
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        df = pd.DataFrame([row for rows in ex.map(run_one, races) for row in rows])
    res = pd.DataFrame([loro(df, s, b) for b in budgets for s in ("detectors", "baseline")])
    pd.set_option("display.width", 200)
    print("leave-one-race-out (setting chosen on the other races, scored on the held-out race):")
    print(res.round(3).to_string(index=False))
    grid = detector_grid()
    final = {b: grid[choose(df[df["system"] == "detectors"], b)] for b in budgets}
    base_final = {b: BASELINE_KMH[choose(df[df["system"] == "baseline"], b)] for b in budgets}
    for b in budgets:
        print(f"budget {b}/h: detector setting on all races {final[b]}, baseline {base_final[b]:.0f} km/h")
    default_budget = 3.0 if 3.0 in budgets else budgets[0]
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = replace(Config(), **final[default_budget])
    CONFIG_PATH.write_text(json.dumps({"budget_fa_h": default_budget, "config": asdict(cfg)}, indent=2))
    CHARTS.mkdir(parents=True, exist_ok=True)
    (CHARTS / "tune_results.md").write_text(
        "# Detector tuning, leave-one-race-out (training races only)\n\nReplay of historical FastF1 data. "
        "For each race the setting is chosen on the other training races and scored on that race; the naive "
        "speed-threshold baseline gets the same procedure and budget.\n\n" + res.round(3).to_markdown(index=False)
        + "\n\nSettings chosen on all training races:\n\n"
        + "\n".join(f"- budget {b}/h: detectors {final[b]}, baseline {base_final[b]:.0f} km/h" for b in budgets) + "\n")
    print(f"saved {CONFIG_PATH} (budget {default_budget}/h)")


if __name__ == "__main__":
    main()

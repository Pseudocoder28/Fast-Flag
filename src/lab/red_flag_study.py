"""Red flag study: which race control rules predict race control's own red flags, on the
training races (tuning) and 2021 Azerbaijan (a check it was never tuned on). Never a holdout.

Detections: production settings, ANOMALY model trained on the other training races (as the
escalation scorecard), computed once per race. Every variant then replays the same ticks and
detections through src.racecontrol.engine with some module constants overridden in the
worker process; engine.py is never edited here.

Scoring, per race: an official red is caught when we sent a RED from 300 s before it to
60 s after it. One of our reds is false when no official red is in that window around it
and race control's own track status was not red.

Run: python -m src.lab.red_flag_study          (about 3 minutes; writes docs/lab/red_flag.md)
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUT = Path("docs/lab/red_flag.md")
CHECK_RACE = "2021_Azerbaijan"
BEFORE_S, AFTER_S = 300.0, 60.0
DEFAULTS = {"RED_BY_TIME": False, "RED_FROM_MULTI": False, "LATE_RED_LAPS": 4, "LATE_RED_MIN_LAPS": 2}

VARIANTS: list[tuple[str, dict]] = [
    ("Old rules: red by time at a crash site, multi-car red", {"RED_BY_TIME": True, "RED_FROM_MULTI": True, "_laps": False}),
    ("Late-race red, up to 3 laps left", {"LATE_RED_LAPS": 3}),
    ("Late-race red, up to 4 laps left (chosen)", {"LATE_RED_LAPS": 4}),
    ("Late-race red, up to 5 laps left", {"LATE_RED_LAPS": 5}),
    ("Late-race red, up to 6 laps left", {"LATE_RED_LAPS": 6}),
    ("Late-race red (4) plus red by time", {"LATE_RED_LAPS": 4, "RED_BY_TIME": True}),
    ("Late-race red (4) plus multi-car red", {"LATE_RED_LAPS": 4, "RED_FROM_MULTI": True}),
]


def detections(rid: str) -> tuple[str, list[tuple[int, dict]]]:
    """Every detection with the index of the tick after which the live server sends it."""
    from src.eval.latency_by_type import loro_suite
    from src.ingest.holdout import assert_not_holdout
    from src.replay.engine import Engine, RaceData
    assert_not_holdout(rid=rid)
    race = RaceData.load(rid)
    eng = Engine(race, [loro_suite(race)])
    out = []
    for k, step in enumerate(eng.steps(eng.t_end)):
        out += [(k, env["data"]) for env in step.envelopes if env["kind"] == "detection"]
    return rid, out


def replay(job: tuple[str, list[tuple[int, dict]], dict]) -> dict:
    """One race through one variant: our red episodes against race control's reds."""
    rid, dets, patch = job
    import src.racecontrol.engine as E
    from src.eval.incidents import race_files
    from src.replay.engine import RaceData
    for name, value in {**DEFAULTS, **patch}.items():     # workers are reused: reset every setting first
        if not name.startswith("_"):
            setattr(E, name, value)
    race = RaceData.load(rid)
    official, meta = race_files(rid)
    rc = E.RaceControl(race_laps=E.race_laps_from_track(race.track) if patch.get("_laps", True) else None)
    by_k: dict[int, list[dict]] = {}
    for k, d in dets:
        by_k.setdefault(k, []).append(d)
    recs = []
    for k in range(len(race.times)):
        recs += rc.on_tick(race.tick(k))[0]
        for d in by_k.get(k, []):
            recs += rc.on_detection(d)
    episodes, level = [], "CLEAR"
    for r in recs:
        if r["flag"] == "CLEAR" and r["message"] == "TRACK CLEAR":
            level = "CLEAR"
        elif r["flag"] in ("VSC", "SC", "RED"):
            if r["flag"] == "RED" and level != "RED":
                episodes.append({"t": r["t"], "reason": r["reason"]})
            level = r["flag"]
    t0, t1 = meta["t_start"], meta["t_end"]
    reds = [m["t"] for m in official if m["flag"] == "RED" and t0 <= m["t"] <= t1]
    status = race.frame.drop_duplicates("t").set_index("t")["track_status"].astype(str)
    red_status = set(status[status == "5"].index)
    near = lambda t, o: o - BEFORE_S <= t <= o + AFTER_S  # noqa: E731
    hits = [{"t": o, "lead_s": round(o - next(e["t"] for e in episodes if near(e["t"], o)), 1)}
            for o in reds if any(near(e["t"], o) for e in episodes)]
    misses = [o for o in reds if not any(near(e["t"], o) for e in episodes)]
    false = [e for e in episodes if not any(near(e["t"], o) for o in reds)
             and not any(abs(e["t"] - t) < 0.3 for t in red_status)]
    return {"race": rid, "hits": hits, "misses": misses, "false": false, "hours": (t1 - t0) / 3600}


def summary(rows: list[dict]) -> dict:
    hits = sum(len(r["hits"]) for r in rows)
    total = hits + sum(len(r["misses"]) for r in rows)
    false = sum(len(r["false"]) for r in rows)
    hours = sum(r["hours"] for r in rows)
    return {"hits": hits, "total": total, "false": false, "per_h": false / hours if hours else 0.0,
            "leads": sorted(h["lead_s"] for r in rows for h in r["hits"])}


def main() -> None:
    from src.replay.engine import available_races
    races = available_races() + [CHECK_RACE]
    with ProcessPoolExecutor(max_workers=min(3, os.cpu_count() or 2)) as ex:
        dets = dict(ex.map(detections, races))
        results = {label: list(ex.map(replay, [(rid, dets[rid], patch) for rid in races]))
                   for label, patch in VARIANTS}
    lines = ["# Red flag study: which rules predict race control's red flags", "",
             "Replay of historical FastF1 data. Generated by `python -m src.lab.red_flag_study`. Tuned on the 20 "
             f"training races; {CHECK_RACE.replace('_', ' ')} is a check it was never tuned on; no holdout race is "
             f"loaded. A red is caught when we sent a RED from {BEFORE_S:.0f} s before race control's to "
             f"{AFTER_S:.0f} s after it; a red of ours is false when race control showed no red in that window.", "",
             "| Rules | Training: reds caught | Training: false reds | 2021 Azerbaijan: caught | 2021 Azerbaijan: false |",
             "|---|---:|---:|---:|---:|"]
    for label, _ in VARIANTS:
        rows = results[label]
        tr = summary([r for r in rows if r["race"] != CHECK_RACE])
        ck = summary([r for r in rows if r["race"] == CHECK_RACE])
        lines.append(f"| {label} | {tr['hits']} of {tr['total']} | {tr['false']} ({tr['per_h']:.2f} per race hour) "
                     f"| {ck['hits']} of {ck['total']} | {ck['false']} |")
        print(lines[-1])
    chosen = results["Late-race red, up to 4 laps left (chosen)"]
    old = results["Old rules: red by time at a crash site, multi-car red"]
    lines += ["", "## Every red, chosen rules", ""]
    for r in chosen:
        lines += [f"- Caught: {r['race']}, race control's red at {h['t']:.1f} s, ours {h['lead_s']:.0f} s earlier"
                  for h in r["hits"]]
        lines += [f"- Missed: {r['race']}, race control's red at {o:.1f} s" for o in r["misses"]]
        lines += [f"- False: {r['race']}, ours at {e['t']:.1f} s ({e['reason']})" for e in r["false"]]
    lines += ["", "## False reds under the old rules", ""]
    for r in old:
        lines += [f"- {r['race']}, ours at {e['t']:.1f} s ({e['reason']})" for e in r["false"]]
    lines += ["", "## What this means", "",
              "- Race control's reds in this data follow either a late neutralisation (a restart beats finishing "
              "behind the Safety Car) or damage car data cannot see (barrier, gravel, debris, rain). Only the first is "
              "visible here, so our red comes only late in the race: our VSC or SC out for a live incident with 2 to "
              "4 laps left, from a race length computed from the lap length before the race.",
              "- The old red by time at a crash site caught two mid-race reds (2023 Australia, Albon; 2023 Mexico City, "
              "Magnussen) but sent many more reds race control never called. A long recovery now only adds an "
              "advisory to the crash sector's flag.",
              "- Five reds in 20 races is very little data. Treat these rules as a transparent starting point, not a "
              "measured red flag accuracy: race control also has CCTV, marshal reports and medical information."]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    from src.lab.red_flag_study import main as run    # run through the package import (pickling safety)
    run()

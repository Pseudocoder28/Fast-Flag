"""Case study: crash timelines and exposure (who drove past a crash before the flags).

A crash is an official incident that escalated to VSC, SC or red and has an
identifiable onset car (same rules as src.eval.latency_by_type). Per crash:
- timeline: onset, our first alert, official yellow or double yellow, our first
  VSC/SC/RED recommendation, official VSC/SC/red
- exposure: every car passing the crash location (where the crashed car came to
  rest) with session time, speed, and speed as a % of that car's own median speed
  at that point on its previous 3 clean laps, in three windows:
    W1 onset -> our first alert (unavoidable: nobody could know yet)
    W2 our first alert -> official yellow
    W3 our escalation recommendation -> official escalation (the headline number)
  plus, for context only, official yellow -> official escalation.

Recommendations come from the race control engine (src/racecontrol, Ishaan), called
through the interface in docs/session_ishaan.md: RaceControl().on_tick / on_detection /
on_risk, each returning a list of recs. Until that engine is importable, our
recommendation and W3 are reported as pending, with the reason.

Counterfactual: assumes race control acted on our recommendation instantly, with no
model of driver reactions. Out of sample twice for 2021: the models never saw this
race, and 2021 cars ran under different technical rules from our 2023 to 2026
training data. Holdout races get this analysis only in A7, after the freeze.

Run: python -m src.eval.case_study                  (2021_Azerbaijan)
Writes docs/charts/case_studies.json and docs/charts/case_<race>_car<N>.png.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval.incidents import build_incidents, race_files, suspended_times  # noqa: E402
from src.eval.onset import RULE, add_own_ratio, onsets  # noqa: E402
from src.ingest.holdout import is_holdout_id  # noqa: E402
from src.ingest.reference import circ  # noqa: E402
from src.ingest.sectors import sector_matches  # noqa: E402
from src.replay.engine import Engine, RaceData, race_dir  # noqa: E402

CHARTS = Path("docs/charts")
OUT = CHARTS / "case_studies.json"
ESCALATIONS = ("VSC", "SC", "RED")
SECTOR_FLAGS = ("YELLOW", "DOUBLE_YELLOW")
ONSET_BEFORE_S, ONSET_AFTER_S = 120.0, 5.0
ALERT_BEFORE_S, ALERT_AFTER_S = 10.0, 180.0
REST_KMH, REST_WITHIN_S = 5.0, 30.0
COUNTERFACTUAL = ("Counterfactual: assumes race control acted on our recommendation instantly; "
                  "no model of how drivers react.")
OUT_OF_SAMPLE = ("Out of sample twice: our models never saw this race, and 2021 cars ran under different "
                 "technical rules from our 2023 to 2026 training data.")
SURFACE, INK, INK2, GRID, WASH = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
BLUE, ORANGE = "#2a78d6", "#eb6834"


# ---- pipeline: detections, risk, and (when available) race control recommendations

def run_pipeline(race: RaceData) -> tuple[list[dict], list[dict] | None, str]:
    from src.replay.pipeline import all_processors
    eng = Engine(race, all_processors(race))
    envs = eng.advance(eng.t_end)
    dets = sorted((e["data"] for e in envs if e["kind"] == "detection"), key=lambda d: d["t"])
    recs, source = recommendations(envs)
    return dets, recs, source


def recommendations(envelopes: list[dict]) -> tuple[list[dict] | None, str]:
    try:
        from src.racecontrol.engine import RaceControl
    except Exception as e:  # not on this branch yet
        return None, (f"pending: the race control engine (src/racecontrol, Ishaan) is not importable here "
                      f"({type(e).__name__}); rerun once it is on main")
    rc = RaceControl()
    recs = []
    for env in envelopes:
        handler = {"tick": rc.on_tick, "detection": rc.on_detection, "risk": rc.on_risk}.get(env["kind"])
        if handler:
            out = handler(env["data"])
            if isinstance(out, tuple):      # on_tick on b-work returns (recs, did_reset), not just recs
                out = out[0]
            recs.extend(out or [])
    return sorted(recs, key=lambda r: r["t"]), "src.racecontrol.engine.RaceControl"


# ---- crash location and exposure

def rest_position(frame: pd.DataFrame, drv: str, onset: pd.Series) -> float:
    after = frame[(frame["drv"] == drv) & (frame["t"] >= onset["t"]) & (frame["t"] <= onset["t"] + REST_WITHIN_S)]
    stopped = after[(after["speed"] < REST_KMH) & after["dist"].notna()]
    return float(stopped["dist"].iloc[0]) if len(stopped) else float(onset["dist"])


def passes(frame: pd.DataFrame, exclude: set[str], crash_dist: float, t0: float, t1: float, length: float,
           names: dict[str, str]) -> list[dict]:
    """Cars crossing the crash location between t0 and t1 (interpolated between ticks)."""
    out = []
    w = frame[(frame["t"] >= t0 - 1) & (frame["t"] <= t1 + 1) & ~frame["drv"].isin(exclude)]
    for drv, car in w.groupby("drv"):
        car = car.sort_values("t")
        ok = (car["dist"].notna() & car["speed"].notna() & ~car["in_pit"].astype(bool)).to_numpy()
        t, d, v, ref = (car[c].to_numpy(float) for c in ("t", "dist", "speed", "own_ref"))
        for k in range(len(car) - 1):
            if not (ok[k] and ok[k + 1]):
                continue
            step = circ(d[k + 1] - d[k], length)
            to_crash = circ(crash_dist - d[k], length)
            if not (0 < step < 150 and 0 <= to_crash < step):
                continue
            frac = to_crash / step
            tp = t[k] + frac * (t[k + 1] - t[k])
            if not (t0 <= tp < t1):
                continue
            speed = v[k] + frac * (v[k + 1] - v[k])
            own = ref[k + 1] if np.isfinite(ref[k + 1]) else ref[k]
            out.append({"car": str(drv), "driver": names.get(str(drv), str(drv)), "t": round(float(tp), 2),
                        "speed_kmh": round(float(speed), 1),
                        "own_normal_kmh": round(float(own), 1) if np.isfinite(own) else None,
                        "pct_of_own_normal": round(float(100 * speed / own), 1) if np.isfinite(own) and own > 0 else None})
    return sorted(out, key=lambda p: p["t"])


def window(name: str, start: float | None, end: float | None, frame, exclude, crash_dist, length, names,
           note: str = "") -> dict:
    if start is None or end is None:
        return {"name": name, "start": start, "end": end, "seconds": None, "cars_passing": None, "passes": [],
                "note": note or "pending"}
    if end <= start:
        return {"name": name, "start": start, "end": end, "seconds": round(end - start, 2), "cars_passing": 0,
                "passes": [], "note": "empty: the later event came first"}
    p = passes(frame, exclude, crash_dist, start, end, length, names)
    return {"name": name, "start": round(start, 2), "end": round(end, 2), "seconds": round(end - start, 2),
            "cars_passing": len(p), "passes": p, "note": note}


# ---- crashes

def driver_names(race: RaceData) -> dict[str, str]:
    try:
        import fastf1
        fastf1.set_log_level(logging.ERROR)
        fastf1.Cache.enable_cache(str(Path("data") / "fastf1_cache"))
        s = fastf1.get_session(int(race.meta["year"]), int(race.meta["round"]), "R")
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        return {str(r.DriverNumber): str(r.Abbreviation) for r in s.results.itertuples()}
    except Exception:
        return {}


def crashes(rid: str) -> tuple[list[dict], str]:
    if is_holdout_id(rid):
        raise SystemExit(f"{rid} is a holdout race: it gets this analysis only in A7, after the freeze")
    race = RaceData.load(rid)
    official, meta = race_files(rid, race_dir(rid))
    n, length = meta["n_msectors"], float(meta["lap_length_m"])
    frame = add_own_ratio(race.frame, length)
    ons = onsets(frame, length)
    dets, recs, rec_source = run_pipeline(race)
    names = driver_names(race)
    incidents, _ = build_incidents(official, meta, suspended_times(race.frame), n)
    out = []
    for inc in incidents:
        esc = [m for m in inc.messages if m["flag"] in ESCALATIONS]
        if not esc:
            continue
        match = (lambda m: not inc.sectors or any(sector_matches(int(m), s, n) for s in inc.sectors))
        cand = ons[(ons["t"] >= inc.t - ONSET_BEFORE_S) & (ons["t"] <= inc.t + ONSET_AFTER_S)]
        cand = cand[cand["msector"].apply(match)]
        if cand.empty:
            continue
        onset = cand.iloc[0]
        cars = set(cand["drv"])
        crash_dist = rest_position(frame, onset["drv"], onset)
        alert = next((d for d in dets if onset["t"] - ALERT_BEFORE_S <= d["t"] <= onset["t"] + ALERT_AFTER_S
                      and (cars & set(d["drivers"]) or match(d["msector"]))), None)
        yellow = min((m["t"] for m in inc.messages if m["flag"] in SECTOR_FLAGS), default=None)
        first_esc = min(esc, key=lambda m: m["t"])
        our_esc = None
        if recs is not None:
            our_esc = next((r for r in recs if r["flag"] in ESCALATIONS and r["t"] >= onset["t"] - ALERT_BEFORE_S
                            and (not r.get("msector") or match(r["msector"]))), None)
        a_t = alert["t"] if alert else None
        common = (frame, cars, crash_dist, length, names)
        out.append({
            "race": rid, "car": onset["drv"], "driver": names.get(onset["drv"], onset["drv"]),
            "cars_involved": sorted(cars), "crash_location_m": round(crash_dist, 1), "msector": int(onset["msector"]),
            "timeline": {
                "onset": round(float(onset["t"]), 2),
                "our_first_alert": {"t": a_t, "type": alert["type"], "evidence": alert["evidence"]} if alert else None,
                "official_yellow": yellow,
                "our_escalation_recommendation": ({"t": our_esc["t"], "flag": our_esc["flag"]} if our_esc
                                                  else (rec_source if recs is None else None)),
                "official_escalation": {"t": first_esc["t"], "flag": first_esc["flag"]},
                "all_official_messages": [{"t": m["t"], "flag": m["flag"], "message": m["message"]}
                                          for m in inc.messages],
            },
            "windows": [
                window("W1 onset -> our first alert (unavoidable)", float(onset["t"]), a_t, *common),
                window("W2 our first alert -> official yellow", a_t, yellow, *common),
                window("W3 our escalation recommendation -> official escalation (headline)",
                       our_esc["t"] if our_esc else None, first_esc["t"], *common,
                       note="" if our_esc else rec_source),
                window("context: official yellow -> official escalation (not a counterfactual)",
                       yellow, first_esc["t"], *common),
            ],
        })
    return out, rec_source


# ---- chart

def plot(c: dict, path: Path) -> None:
    """Timeline: shaded windows, event lines, and every car passing the crash site
    (highlighted and named in W1 to W3, grey for the after-yellow context)."""
    tl, t0 = c["timeline"], c["timeline"]["onset"]
    rel = lambda t: t - t0  # noqa: E731
    rec = tl["our_escalation_recommendation"]
    events = [("onset", tl["onset"]), ("our first alert", tl["our_first_alert"]["t"] if tl["our_first_alert"] else None),
              ("official yellow", tl["official_yellow"]),
              (f"official {tl['official_escalation']['flag']}", tl["official_escalation"]["t"])]
    if isinstance(rec, dict):
        events.append((f"our {rec['flag']} recommendation", rec["t"]))
    events = sorted([(n, t) for n, t in events if t is not None], key=lambda e: e[1])
    x_max = max(rel(t) for _, t in events) + 8
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    washes = {"W1": (WASH, 1.0), "W2": (ORANGE, 0.12), "W3": (BLUE, 0.12), "context:": ("#f6f5f2", 1.0)}
    narrow_count = 0                                   # narrow windows get their labels stacked, left of the window
    for w in c["windows"]:
        if not w["seconds"] or w["seconds"] <= 0:
            continue
        key = w["name"].split()[0]
        color, alpha = washes[key]
        a, b = rel(w["start"]), rel(w["end"])
        ax.axvspan(a, b, color=color, alpha=alpha, lw=0, zorder=0)
        label = f"after the yellow: {w['cars_passing']} cars" if key == "context:" else f"{key}: {w['cars_passing']} cars"
        narrow = b - a < 0.08 * (x_max + 5)
        y = 123 - 9 * narrow_count if narrow else 123
        narrow_count += narrow
        ax.text(a - 0.4 if narrow else (a + b) / 2, y, label, ha="right" if narrow else "center", va="bottom",
                fontsize=8.5, color=INK2)
    ax.axhline(100, color=GRID, lw=1, zorder=1)
    ax.text(x_max, 101, "normal speed at this point", ha="right", va="bottom", fontsize=7.5, color=INK2)
    level, last_x = 0, -1e9
    for label, t in events:
        x = rel(t)
        level = level + 1 if x - last_x < 0.07 * (x_max + 5) else 0
        last_x = x
        ax.axvline(x, color=INK2, lw=0.8, zorder=1)
        right = x > 0.8 * x_max
        ax.annotate(f"{label} {x:+.1f} s", (x, 0), xycoords=("data", "axes fraction"),
                    xytext=(-3 if right else 3, -14 - 12 * level), textcoords="offset points",
                    ha="right" if right else "left", va="top", fontsize=7.5, color=INK)
    story = [p for w in c["windows"] if not w["name"].startswith("context") for p in w["passes"]]
    context = [p for w in c["windows"] if w["name"].startswith("context") for p in w["passes"]]
    for p in context:
        if p["pct_of_own_normal"] is not None:
            ax.scatter(rel(p["t"]), p["pct_of_own_normal"], s=36, color="#b8b6b0", edgecolors=SURFACE,
                       linewidths=1.5, zorder=2)
    label_w = 0.075 * (x_max + 5)                      # rough width of a "HAM 91%" label in data units
    last = {1: [-1e9, 0], -1: [-1e9, 0]}              # per side (above = 1, below = -1): last x, level
    for k, p in enumerate(sorted(story, key=lambda p: p["t"])):
        if p["pct_of_own_normal"] is None:
            continue
        x, y = rel(p["t"]), p["pct_of_own_normal"]
        side = 1 if k % 2 == 0 else -1
        lx, lv = last[side]
        level = lv + 1 if x - lx < label_w else 0
        last[side] = [x, level]
        ax.scatter(x, y, s=64, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"{p['driver']} {y:.0f}%", (x, y), xytext=(0, side * (9 + 11 * level)), textcoords="offset points",
                    ha="center", va="bottom" if side > 0 else "top", fontsize=7.5, color=INK,
                    arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8} if level else None)
    ax.set_xlim(-5, x_max)
    ax.set_ylim(-4, 132)
    ax.set_ylabel("Speed passing the crash site\n(% of the car's own normal speed there)", fontsize=8.5, color=INK2)
    ax.set_xlabel("Seconds after the crash onset", fontsize=9, color=INK2, labelpad=44)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, labelsize=8, colors=INK2)
    fig.text(0.01, 0.985, f"{c['race'].replace('_', ' ')}: {c['driver']} (car {c['car']}) crash, who drove past "
             "before the flags", fontsize=12.5, weight="bold", color=INK, va="top")
    pending = "" if isinstance(rec, dict) else " Our escalation recommendation and W3 are pending the race control engine."
    fig.text(0.01, 0.945, "Replay of historical FastF1 data. Each dot is a car passing the spot where the crashed car "
             "came to rest: blue and named inside W1 to W3, grey after the official yellow." + pending,
             fontsize=8.5, color=INK2, va="top", wrap=True)
    fig.text(0.01, 0.012, f"{COUNTERFACTUAL}\n{OUT_OF_SAMPLE}\nW1: onset to our first alert (unavoidable). "
             "W2: our first alert to the official yellow. W3: our escalation recommendation to the official "
             "escalation.", fontsize=7.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.27)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Crash timelines and exposure for one race")
    p.add_argument("race", nargs="?", default="2021_Azerbaijan")
    a = p.parse_args()
    found, rec_source = crashes(a.race)
    CHARTS.mkdir(parents=True, exist_ok=True)
    doc = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    doc |= {"note": "Replay of historical FastF1 data.", "counterfactual": COUNTERFACTUAL, "onset_rule": RULE}
    doc.setdefault("races", {})[a.race] = {"out_of_sample": OUT_OF_SAMPLE, "recommendation_source": rec_source,
                                          "crashes": found}
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    for c in found:
        path = CHARTS / f"case_{c['race']}_car{c['car']}.png"
        plot(c, path)
        tl = c["timeline"]
        print(f"{c['driver']} (car {c['car']}) onset {tl['onset']:.1f}, our alert "
              f"{tl['our_first_alert']['t'] if tl['our_first_alert'] else None}, official yellow {tl['official_yellow']}, "
              f"official {tl['official_escalation']['flag']} {tl['official_escalation']['t']} -> {path}")
        for w in c["windows"]:
            print(f"   {w['name']}: {w['seconds']} s, cars passing {w['cars_passing']}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    from src.eval.case_study import main as run    # run through the package import
    run()

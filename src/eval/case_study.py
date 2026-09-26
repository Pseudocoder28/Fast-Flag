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
from src.eval.latency_by_type import onset_candidates, onset_cars, pick_onset  # noqa: E402
from src.eval.onset import RULE, add_own_ratio, onsets  # noqa: E402
from src.ingest.holdout import is_holdout_id  # noqa: E402
from src.ingest.reference import circ  # noqa: E402
from src.ingest.sectors import sector_matches  # noqa: E402
from src.replay.engine import Engine, RaceData, race_dir  # noqa: E402

CHARTS = Path("docs/charts")
OUT = CHARTS / "case_studies.json"
ESCALATIONS = ("VSC", "SC", "RED")
SECTOR_FLAGS = ("YELLOW", "DOUBLE_YELLOW")
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


def crashes(rid: str, allow_holdout: bool = False) -> tuple[list[dict], str]:
    """allow_holdout is only passed by the A7 holdout run (src.eval.holdout), after the freeze."""
    if is_holdout_id(rid) and not allow_holdout:
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
        cand = onset_candidates(inc, ons, n)
        if cand.empty:
            continue
        onset = pick_onset(cand, inc.t)
        cars = onset_cars(cand, onset)
        crash_dist = rest_position(frame, onset["drv"], onset)
        alert = next((d for d in dets if onset["t"] - ALERT_BEFORE_S <= d["t"] <= onset["t"] + ALERT_AFTER_S
                      and (cars & set(d["drivers"]) or match(d["msector"]))), None)
        yellow = min((m["t"] for m in inc.messages if m["flag"] in SECTOR_FLAGS), default=None)
        first_esc = min(esc, key=lambda m: m["t"])
        our_esc = None
        if recs is not None:
            our_esc = next((r for r in recs if r["flag"] in ESCALATIONS and r["t"] >= onset["t"] - ALERT_BEFORE_S
                            and (not r.get("msector") or match(r["msector"]))), None)
        no_esc = (rec_source if recs is None
                  else "none: our race control engine made no VSC, SC or red recommendation for this crash")
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
                                                  else no_esc),
                "official_escalation": {"t": first_esc["t"], "flag": first_esc["flag"]},
                "all_official_messages": [{"t": m["t"], "flag": m["flag"], "message": m["message"]}
                                          for m in inc.messages],
            },
            "windows": [
                window("W1 onset -> our first alert (unavoidable)", float(onset["t"]), a_t, *common),
                window("W2 our first alert -> official yellow", a_t, yellow, *common),
                window("W3 our escalation recommendation -> official escalation (headline)",
                       our_esc["t"] if our_esc else None, first_esc["t"], *common,
                       note="" if our_esc else no_esc),
                window("context: official yellow -> official escalation (not a counterfactual)",
                       yellow, first_esc["t"], *common),
            ],
        })
    return out, rec_source


# ---- chart

def place_labels(points: list[tuple[float, float, str]], xu: float, yu: float,
                 obstacles: list[tuple[float, float]] = ()) -> list[tuple[float, float, str, str]]:
    """Greedy, collision-free label spots for dots. xu, yu: data units per inch. Tries right,
    left, above, below and the diagonals; skips a label that fits nowhere. obstacles: other
    dots that labels must not cover. Returns (x, y, text, ha)."""
    dots = [(x, y) for x, y, _ in points] + list(obstacles)
    boxes = [(x - 0.06 * xu, y - 0.06 * yu, x + 0.06 * xu, y + 0.06 * yu) for x, y in dots]
    out = []
    for x, y, text in points:
        w, h = (0.07 * len(text) + 0.06) * xu, 0.12 * yu
        for dx, dy, ha in ((0.08 * xu, 0, "left"), (-0.08 * xu, 0, "right"), (0, 0.15 * yu, "center"),
                           (0, -0.15 * yu, "center"), (0.08 * xu, 0.13 * yu, "left"), (0.08 * xu, -0.13 * yu, "left"),
                           (-0.08 * xu, 0.13 * yu, "right"), (-0.08 * xu, -0.13 * yu, "right")):
            x0 = x + dx if ha == "left" else x + dx - w if ha == "right" else x - w / 2
            box = (x0, y + dy - h / 2, x0 + w, y + dy + h / 2)
            if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in boxes):
                boxes.append(box)
                out.append((x + dx, y + dy, text, ha))
                break
    return out


WINDOW_STYLE = {"W1": ("#b8b6b0", "W1 onset to our first alert"), "W2": (ORANGE, "W2 our first alert to official yellow"),
                "W3": (BLUE, "W3 our escalation to official escalation"),
                "context:": ("#d6d4ce", "context: official yellow to official escalation")}


def plot(c: dict, path: Path) -> None:
    """Timeline chart. Top strip: one bar per window (they can overlap in time). Below:
    every car passing the crash site, once, at its speed as % of its own normal speed
    there; blue and named if it passed inside W1 to W3, grey if only after the yellow."""
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
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    # timeline strip
    for row, w in enumerate(c["windows"]):
        key = w["name"].split()[0]
        color, label = WINDOW_STYLE[key]
        y = 138 - 8 * row
        cars = "n/a" if w["cars_passing"] is None else f"{w['cars_passing']} cars"
        if w["seconds"] and w["seconds"] > 0:
            a, b = rel(w["start"]), rel(w["end"])
            ax.plot([a, b], [y, y], color=color, lw=6, solid_capstyle="butt", zorder=2)
            ax.text(a, y + 1.6, f"{label}: {cars} ({b - a:.1f} s)", va="bottom", ha="left", fontsize=7.5, color=INK)
        else:
            ax.text(-4.5, y, f"{label}: {cars} ({w['note'].split(':')[0] if w['note'] else 'empty'})", va="center",
                    fontsize=7.5, color=INK2)
    ax.axhline(106, color=GRID, lw=1, zorder=1)
    ax.axhline(100, color=GRID, lw=1, zorder=1)
    ax.text(x_max, 101, "normal speed at this point", ha="right", va="bottom", fontsize=7.5, color=INK2)
    level, last_x, max_level = 0, -1e9, 0
    for label, t in events:
        x = rel(t)
        level = level + 1 if x - last_x < 0.07 * (x_max + 5) else 0
        last_x, max_level = x, max(max_level, level)
        ax.plot([x, x], [-4, 106], color=INK2, lw=0.8, zorder=1)
        right = x > 0.8 * x_max
        ax.annotate(f"{label} {x:+.1f} s", (x, 0), xycoords=("data", "axes fraction"),
                    xytext=(-3 if right else 3, -14 - 12 * level), textcoords="offset points",
                    ha="right" if right else "left", va="top", fontsize=7.5, color=INK)
    story_w = [w for w in c["windows"] if not w["name"].startswith("context")]
    story = {(p["car"], p["t"]): p for w in story_w for p in w["passes"]}          # each pass once
    context = {(p["car"], p["t"]): p for w in c["windows"] if w["name"].startswith("context") for p in w["passes"]}
    grey = [(rel(p["t"]), p["pct_of_own_normal"]) for key, p in context.items()
            if key not in story and p["pct_of_own_normal"] is not None]
    for x, y in grey:
        ax.scatter(x, y, s=36, color="#b8b6b0", edgecolors=SURFACE, linewidths=1.5, zorder=3)
    pts = [(rel(p["t"]), p["pct_of_own_normal"], p["driver"]) for p in sorted(story.values(), key=lambda p: p["t"])
           if p["pct_of_own_normal"] is not None]
    for x, y, _ in pts:
        ax.scatter(x, y, s=64, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=4)
    xu = (x_max + 5) / (11 * (0.98 - 0.08))            # data units per inch, from the figure layout below
    yu = (143 + 4) / (6.4 * (0.87 - 0.27))
    for x, y, text, ha in place_labels(pts, xu, yu, grey):
        ax.text(x, y, text, ha=ha, va="center", fontsize=7.5, color=INK, zorder=5)
    ax.set_xlim(-5, x_max)
    ax.set_ylim(-4, 143)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Speed passing the crash site\n(% of the car's own normal speed there)", fontsize=8.5, color=INK2)
    ax.set_xlabel("Seconds after the crash onset", fontsize=9, color=INK2, labelpad=18 + 12 * max_level)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, labelsize=8, colors=INK2)
    fig.text(0.01, 0.985, f"{c['race'].replace('_', ' ')}: {c['driver']} (car {c['car']}) crash, who drove past "
             "before the flags", fontsize=12.5, weight="bold", color=INK, va="top")
    if isinstance(rec, dict):
        pending = ""
    elif str(rec).startswith("pending"):
        pending = " Our escalation recommendation and W3 are pending the race control engine."
    else:
        pending = " Our race control engine made no VSC, SC or red recommendation for this crash, so W3 is empty."
    fig.text(0.01, 0.945, "Replay of historical FastF1 data. Each dot is a car passing the spot where the crashed car "
             "came to rest (once): blue and named if it passed inside W1 to W3, grey if only after the official "
             "yellow." + pending, fontsize=8.5, color=INK2, va="top", wrap=True)
    fig.text(0.01, 0.012, f"{COUNTERFACTUAL}\n{OUT_OF_SAMPLE}\nWindows can overlap: a car passing between our "
             "escalation and the official yellow counts in W2 and W3. Every car, time and speed is in case_studies.json.",
             fontsize=7.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.27)
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

"""Steward's cards: one self-contained HTML card (inline SVG, no CDN, system fonts) and
one PNG per incident, rendered from docs/lab/delay_cost.json (run src.lab.delay_cost first).

Each card, sized for a 16:9 slide: header (race, lap, sector, car), a timeline strip
(onset, our first alert, our recommendation, official yellow, official escalation), the
car's speed trace against its own normal speed, the detector evidence, our recommendation
and its reason, the exposure numbers, a delay-cost mini chart and the honesty label.

Run: python -m src.lab.cards                       (docs/lab/delay_cost.json -> docs/lab/cards/)
     python -m src.lab.cards --only 2021_Azerbaijan
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from src.eval.latency_by_type import ALERT_BEFORE_S  # noqa: E402
from src.lab.delay_cost import alert_cars_note, n_cars  # noqa: E402

IN_PATH = Path("docs/lab/delay_cost.json")
OUT_DIR = Path("docs/lab/cards")
W, H = 1280, 720                      # card size in px, 16:9
DPI = 100                             # PNG at exactly W x H px
SURFACE, INK, INK2, INK3, GRID, WASH = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df", "#f0efec"
BLUE, ORANGE, GOLD, GREEN = "#2a78d6", "#eb6834", "#b8960c", "#2e8b57"
FLAG_COLOR = {"YELLOW": "#ffd400", "DOUBLE_YELLOW": "#ffb000", "VSC": "#00a3e0", "SC": "#ff6b00", "RED": "#e10600"}
EVENT_STYLE = {"onset": (INK, "solid"), "our first alert": (ORANGE, "solid"), "our recommendation": (BLUE, "solid"),
               "official yellow": (GOLD, "dashed"), "official escalation": (INK, "dashed")}
TRACE_BEFORE_S, TRACE_AFTER_S = 15.0, 45.0
MAX_EVIDENCE = 4


# ---- card model: what every renderer shows

def events(inc: dict) -> list[dict]:
    m = inc["markers"]
    out = [{"name": "onset", "t": 0.0, "what": ""}]
    if m["our_first_alert"]:
        a = m["our_first_alert"]
        out.append({"name": "our first alert", "t": a["t_after_onset"], "what": a["type"] + alert_cars_note(a)})
    if m["our_escalation"]:
        out.append({"name": "our recommendation", "t": m["our_escalation"]["t_after_onset"], "what": m["our_escalation"]["flag"]})
    elif first_flag_rec(inc):
        r = first_flag_rec(inc)
        out.append({"name": "our recommendation", "t": r["t_after_onset"], "what": r["flag"]})
    if m["official_yellow"] is not None:
        out.append({"name": "official yellow", "t": m["official_yellow"], "what": ""})
    if m["official_escalation"]:
        out.append({"name": "official escalation", "t": m["official_escalation"]["t_after_onset"],
                    "what": m["official_escalation"]["flag"]})
    return sorted(out, key=lambda e: e["t"])


def recommendation(inc: dict) -> dict | None:
    recs = inc.get("recommendations") or []
    m = inc["markers"]
    if m["our_escalation"]:
        esc = next((r for r in recs if r["flag"] == m["our_escalation"]["flag"]
                    and abs(r["t_after_onset"] - m["our_escalation"]["t_after_onset"]) < 0.01), None)
        return esc or {"t_after_onset": m["our_escalation"]["t_after_onset"], "flag": m["our_escalation"]["flag"],
                       "confidence": None, "reason": m["our_escalation"].get("reason", "")}
    return first_flag_rec(inc)


def first_flag_rec(inc: dict) -> dict | None:
    """Our first YELLOW or DOUBLE_YELLOW for this crash, never a CLEAR and never before the
    alert window (a CLEAR from an earlier incident in the sector is not our call for this one)."""
    return next((r for r in inc.get("recommendations") or [] if r["flag"] in ("YELLOW", "DOUBLE_YELLOW")
                 and r["t_after_onset"] >= -ALERT_BEFORE_S), None)


def exposure_lines(inc: dict) -> list[str]:
    m, n60 = inc["markers"], inc["curve"][-1]
    lines = [f"Each second of delay averaged {n60 / inc['max_delay_s']:.2f} cars ({n_cars(n60)} in {inc['max_delay_s']:.0f} s)"]
    if m["our_escalation"]:
        lines.append(f"By our {m['our_escalation']['flag']} ({m['our_escalation']['t_after_onset']:+.1f} s): "
                     f"{n_cars(m['our_escalation']['cars_by_then'])}")
    if m["official_yellow"] is not None:
        lines.append(f"By the official yellow ({m['official_yellow']:+.1f} s): {n_cars(m['cars_by_official_yellow'])}")
    if m["official_escalation"]:
        o = m["official_escalation"]
        lines.append(f"By the official {o['flag']} ({o['t_after_onset']:+.1f} s): {n_cars(o['cars_by_then'])}")
    else:
        lines.append("No official escalation for this incident")
    return lines


def evidence_lines(inc: dict) -> list[str]:
    dets = inc.get("detections") or []
    if not dets and inc["markers"]["our_first_alert"]:
        a = inc["markers"]["our_first_alert"]
        dets = [{"t_after_onset": a["t_after_onset"], "type": a["type"], "severity": None, "evidence": a["evidence"],
                 "drivers": a.get("drivers", [])}]
    crash_cars = set(inc.get("cars_involved") or [inc["car"]])
    out = []
    for d in dets[:MAX_EVIDENCE]:
        sev = f" sev {d['severity']:.2f}" if d.get("severity") is not None else ""
        drivers = d.get("drivers") or []
        other = f" (car {', '.join(drivers)}, same sector)" if drivers and not crash_cars & set(drivers) else ""
        out.append(f"{d['t_after_onset']:+.1f} s {d['type']}{other}{sev}: {d['evidence']}")
    return out or ["no detection within the alert window"]


def official_lines(inc: dict) -> list[str]:
    msgs = inc.get("official_messages") or []
    out = [f"{m['t_after_onset']:+.1f} s {m['message']}" for m in msgs[:MAX_EVIDENCE]]
    if len(msgs) > MAX_EVIDENCE:
        out.append(f"and {len(msgs) - MAX_EVIDENCE} more")
    return out or ["no official message for this incident"]


def header(inc: dict) -> tuple[str, str]:
    lap = f"lap {inc['lap']}" if inc.get("lap") is not None else "lap ?"
    title = f"{inc['race'].replace('_', ' ')}: {inc['driver']} (car {inc['car']})"
    sub = f"{lap} | marshal sector {inc['msector']} | onset t = {inc['onset_t']:.1f} s | official top flag {inc['official_top_flag']}"
    return title, sub


def honesty(inc: dict, doc: dict) -> str:
    note = doc.get("note", "Replay of historical FastF1 data. Counterfactual: no model of driver reactions.")
    if inc["race"] in doc.get("case_study_races", []):
        note += " " + doc.get("case_study_note", "")
    return note


def card_name(inc: dict) -> str:
    return f"card_{inc['race']}_car{inc['car']}_{int(inc['onset_t'])}"


# ---- SVG helpers

def scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    return a + (b - a) * (v - lo) / (hi - lo) if hi > lo else a


def label_of(e: dict) -> str:
    return f"{e['name']} {e['what']} {e['t']:+.1f} s".replace("  ", " ")


def label_levels(ev: list[dict], t_max: float, min_gap: float) -> list[tuple[int, int]]:
    """(side, level) per event: sides alternate, and a label closer than min_gap (in
    seconds of the strip) to the previous label on its side is pushed one level out."""
    out, last = [], {1: [-1e9, 0], -1: [-1e9, 0]}
    for k, e in enumerate(ev):
        side = 1 if k % 2 == 0 else -1
        lx, lv = last[side]
        level = lv + 1 if e["t"] - lx < min_gap else 0
        last[side] = [e["t"], level]
        out.append((side, level))
    return out


def svg_timeline(inc: dict, w: int, h: int) -> str:
    ev = events(inc)
    t_max = max(10.0, max(e["t"] for e in ev) * 1.08)
    x0, x1, y = 20, w - 20, h // 2 + 2
    lines = [f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{GRID}" stroke-width="4"/>']
    labels = []
    for e, (side, level) in zip(ev, label_levels(ev, t_max, 0.18 * t_max)):
        x = scale(e["t"], 0, t_max, x0, x1)
        color, _ = EVENT_STYLE[e["name"]]
        ty = y - 12 - 14 * level if side > 0 else y + 20 + 14 * level
        anchor = "start" if x < x0 + 0.12 * (x1 - x0) else ("end" if x > x1 - 0.12 * (x1 - x0) else "middle")
        if level:        # the leader stops short of its own glyphs; labels are drawn on top with a halo
            lines.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{ty + 3 if side > 0 else ty - 12}" '
                         f'stroke="{color}" stroke-width="1"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
        labels.append(f'<text x="{x:.1f}" y="{ty}" font-size="12" fill="{color}" text-anchor="{anchor}" '
                      f'stroke="{SURFACE}" stroke-width="4" paint-order="stroke">{html.escape(label_of(e))}</text>')
    parts = lines + labels
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="system-ui, sans-serif">{"".join(parts)}</svg>'


def svg_trace(inc: dict, w: int, h: int) -> str:
    tr = inc.get("trace") or {"t_after_onset": [], "speed_kmh": [], "own_normal_kmh": []}
    ts, sp, own = tr["t_after_onset"], tr["speed_kmh"], tr["own_normal_kmh"]
    left, right, top, bottom = 44, w - 12, 14, h - 26
    if not ts:
        return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}"><text x="{w / 2}" y="{h / 2}" text-anchor="middle" '
                f'font-size="13" fill="{INK3}">no speed trace</text></svg>')
    v_max = max([v for v in sp + own if v is not None] + [50.0]) * 1.08
    t_lo, t_hi = min(ts), max(ts)
    sx = lambda t: scale(t, t_lo, t_hi, left, right)  # noqa: E731
    sy = lambda v: scale(v, 0, v_max, bottom, top)  # noqa: E731
    parts = [f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{GRID}"/>',
             f'<line x1="{sx(0):.1f}" y1="{top + 16}" x2="{sx(0):.1f}" y2="{bottom}" stroke="{INK}" stroke-dasharray="3 3"/>']
    for v in (100, 200, 300):
        if v < v_max:
            parts.append(f'<line x1="{left}" y1="{sy(v):.1f}" x2="{right}" y2="{sy(v):.1f}" stroke="{GRID}"/>')
            parts.append(f'<text x="{left - 4}" y="{sy(v) + 4:.1f}" font-size="10" fill="{INK3}" text-anchor="end">{v}</text>')
    for t in range(int(t_lo // 10) * 10, int(t_hi) + 1, 10):
        if t_lo <= t <= t_hi:
            parts.append(f'<text x="{sx(t):.1f}" y="{bottom + 14}" font-size="10" fill="{INK3}" text-anchor="middle">{"0 onset" if t == 0 else f"{t:+d}"}</text>')
    for series, color, width, dash in ((own, INK3, 1.5, ' stroke-dasharray="4 3"'), (sp, BLUE, 2.2, "")):
        pts = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(ts, series) if v is not None)
        if pts:
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{dash}/>')
    parts.append(f'<text x="{right}" y="{top + 10}" font-size="10" fill="{INK3}" text-anchor="end">seconds after onset</text>')
    parts.append(f'<text x="{left + 6}" y="{top + 10}" font-size="11" fill="{BLUE}">speed km/h</text>')
    parts.append(f'<text x="{left + 90}" y="{top + 10}" font-size="11" fill="{INK3}">own normal (previous 3 clean laps)</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="system-ui, sans-serif">{"".join(parts)}</svg>'


def svg_mini_curve(inc: dict, w: int, h: int) -> str:
    y = inc["curve"]
    d_max, step = inc["max_delay_s"], inc["step_s"]
    left, right, top, bottom = 30, w - 10, 12, h - 22
    y_max = max(max(y), 1) * 1.15
    sx = lambda d: scale(d, 0, d_max, left, right)  # noqa: E731
    sy = lambda v: scale(v, 0, y_max, bottom, top)  # noqa: E731
    pts = []
    for k, v in enumerate(y):
        d = k * step
        if k:
            pts.append(f"{sx(d):.1f},{sy(y[k - 1]):.1f}")
        pts.append(f"{sx(d):.1f},{sy(v):.1f}")
    parts = [f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{GRID}"/>',
             f'<polyline points="{" ".join(pts)}" fill="none" stroke="{INK}" stroke-width="2"/>',
             f'<text x="{left - 4}" y="{sy(max(y)) + 4:.1f}" font-size="10" fill="{INK3}" text-anchor="end">{max(y)}</text>',
             f'<text x="{right}" y="{bottom + 13}" font-size="10" fill="{INK3}" text-anchor="end">{d_max:.0f} s delay</text>',
             f'<text x="{left}" y="{bottom + 13}" font-size="10" fill="{INK3}">0</text>']
    for e in events(inc):
        if e["name"] == "onset" or not 0 <= e["t"] <= d_max:
            continue
        color, dash = EVENT_STYLE[e["name"]]
        da = ' stroke-dasharray="3 3"' if dash == "dashed" else ""
        parts.append(f'<line x1="{sx(e["t"]):.1f}" y1="{top}" x2="{sx(e["t"]):.1f}" y2="{bottom}" stroke="{color}" stroke-width="1.5"{da}/>')
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="system-ui, sans-serif">{"".join(parts)}</svg>'


# ---- HTML card

def html_card(inc: dict, doc: dict) -> str:
    title, sub = header(inc)
    rec = recommendation(inc)
    flag = rec["flag"] if rec else None
    color = FLAG_COLOR.get(flag or "", GRID)
    conf = f" ({rec['confidence']:.2f})" if rec and rec.get("confidence") is not None else ""
    rec_html = (f'<span class="flag" style="background:{color}">{html.escape(flag)}</span>'
                f'<span class="rec-t">{rec["t_after_onset"]:+.1f} s{conf}</span><div class="reason">{html.escape(rec.get("reason", ""))}</div>'
                if rec else '<div class="reason">no recommendation from our race control engine for this incident</div>')
    evidence = "".join(f"<li>{html.escape(line)}</li>" for line in evidence_lines(inc))
    feed = "".join(f"<li>{html.escape(line)}</li>" for line in official_lines(inc))
    exposure = "".join(f"<li>{html.escape(line)}</li>" for line in exposure_lines(inc))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ margin: 0; background: #dcdcd8; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: {INK}; }}
.card {{ width: {W}px; height: {H}px; margin: 24px auto; background: {SURFACE}; box-shadow: 0 8px 30px rgba(0,0,0,0.25); position: relative; overflow: hidden; }}
.head {{ background: {INK}; color: {SURFACE}; padding: 18px 28px; display: flex; justify-content: space-between; align-items: baseline; }}
.head h1 {{ margin: 0; font-size: 26px; font-weight: 800; letter-spacing: 0.01em; }}
.head .sub {{ font-size: 13px; color: #c9c8c0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.head .mark {{ font-weight: 900; letter-spacing: 0.14em; font-size: 14px; }}
.head .mark b {{ background: {FLAG_COLOR["YELLOW"]}; color: #000; padding: 2px 6px; margin-left: 4px; }}
.strip {{ padding: 6px 20px 0; }}
.grid {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 0 24px; padding: 0 28px; }}
h2 {{ font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: {INK2}; margin: 12px 0 4px; }}
ul {{ margin: 0; padding-left: 16px; font-size: 13px; line-height: 1.45; }}
.rec {{ font-size: 14px; }}
.flag {{ display: inline-block; padding: 3px 10px; font-weight: 900; letter-spacing: 0.06em; color: #000; border-radius: 3px; }}
.rec-t {{ margin-left: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: {INK2}; }}
.reason {{ margin-top: 4px; font-size: 13px; color: {INK}; }}
.foot {{ position: absolute; left: 28px; right: 28px; bottom: 12px; font-size: 11px; color: {INK2}; border-top: 1px solid {GRID}; padding-top: 6px; }}
</style></head><body>
<div class="card">
  <div class="head"><div><h1>{html.escape(title)}</h1><div class="sub">{html.escape(sub)}</div></div><div class="mark">FAST<b>FLAG</b></div></div>
  <div class="strip">{svg_timeline(inc, W - 40, 92)}</div>
  <div class="grid">
    <div>
      <h2>Speed trace, car {html.escape(inc["car"])}</h2>
      {svg_trace(inc, 700, 210)}
      <h2>Detector evidence</h2>
      <ul>{evidence}</ul>
      <h2>Official race control feed</h2>
      <ul>{feed}</ul>
    </div>
    <div>
      <h2>Our recommendation</h2>
      <div class="rec">{rec_html}</div>
      <h2>Exposure (cars past the crash site at racing speed)</h2>
      <ul>{exposure}</ul>
      <h2>Cost of delay</h2>
      {svg_mini_curve(inc, 480, 120)}
    </div>
  </div>
  <div class="foot">{html.escape(honesty(inc, doc))} Onset rule: {html.escape(doc.get("onset_rule", ""))}</div>
</div>
</body></html>
"""


# ---- PNG card (matplotlib, same content)

def png_card(inc: dict, doc: dict, path: Path) -> None:
    title, sub = header(inc)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(4, 2, figure=fig, left=0.035, right=0.975, top=0.87, bottom=0.11, hspace=0.7, wspace=0.16,
                  height_ratios=[0.95, 2.2, 1.6, 1.3], width_ratios=[1.35, 1])
    fig.patches.append(plt.Rectangle((0, 0.885), 1, 0.115, transform=fig.transFigure, color=INK, zorder=0))
    fig.text(0.022, 0.955, title, fontsize=16, weight="bold", color=SURFACE, va="center")
    fig.text(0.022, 0.91, sub, fontsize=8.5, color="#c9c8c0", va="center", family="monospace")
    fig.text(0.975, 0.945, "FAST FLAG", fontsize=10, weight="bold", color=FLAG_COLOR["YELLOW"], ha="right", va="center")

    ax = fig.add_subplot(gs[0, :])          # timeline strip
    ev = events(inc)
    t_max = max(10.0, max(e["t"] for e in ev) * 1.08)
    ax.hlines(0, 0, t_max, color=GRID, lw=3)
    for e, (side, level) in zip(ev, label_levels(ev, t_max, 0.18 * t_max)):
        color, _ = EVENT_STYLE[e["name"]]
        ax.scatter([e["t"]], [0], s=60, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ha = "left" if e["t"] < 0.12 * t_max else ("right" if e["t"] > 0.88 * t_max else "center")
        # level-0 labels sit above the raised labels' leaders, on a surface-coloured box
        ax.annotate(label_of(e), (e["t"], 0), xytext=(0, side * (8 + 10 * level)), textcoords="offset points",
                    ha=ha, va="bottom" if side > 0 else "top", fontsize=7, color=color, zorder=6 - level,
                    bbox={"boxstyle": "square,pad=0.1", "fc": SURFACE, "ec": "none"},
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6} if level else None)
    ax.set_xlim(-t_max * 0.02, t_max)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])          # speed trace
    tr = inc.get("trace") or {"t_after_onset": [], "speed_kmh": [], "own_normal_kmh": []}
    if tr["t_after_onset"]:
        ax.plot(tr["t_after_onset"], [v if v is not None else float("nan") for v in tr["own_normal_kmh"]],
                color=INK3, lw=1.2, ls="--", label="own normal (previous 3 clean laps)")
        ax.plot(tr["t_after_onset"], [v if v is not None else float("nan") for v in tr["speed_kmh"]],
                color=BLUE, lw=1.8, label=f"car {inc['car']} speed")
        ax.axvline(0, color=INK, lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    else:
        ax.text(0.5, 0.5, "no speed trace", transform=ax.transAxes, ha="center", color=INK3)
    ax.set_title("SPEED TRACE (km/h) VS SECONDS AFTER ONSET", loc="left", fontsize=7.5, color=INK2)
    style_axes(ax)

    ax = fig.add_subplot(gs[2:, 0])         # evidence and the official feed
    ax.axis("off")
    ax.text(0, 1, "DETECTOR EVIDENCE", fontsize=7.5, color=INK2, va="top", transform=ax.transAxes)
    ax.text(0, 0.88, "\n".join(wrap(line, 95) for line in evidence_lines(inc)), fontsize=7.5, color=INK, va="top",
            transform=ax.transAxes, linespacing=1.4)
    ax.text(0, 0.42, "OFFICIAL RACE CONTROL FEED", fontsize=7.5, color=INK2, va="top", transform=ax.transAxes)
    ax.text(0, 0.30, "\n".join(official_lines(inc)), fontsize=7.5, color=INK, va="top", transform=ax.transAxes,
            linespacing=1.4)

    ax = fig.add_subplot(gs[1, 1])          # recommendation + exposure
    ax.axis("off")
    rec = recommendation(inc)
    ax.text(0, 1, "OUR RECOMMENDATION", fontsize=7.5, color=INK2, va="top", transform=ax.transAxes)
    if rec:
        conf = f" ({rec['confidence']:.2f})" if rec.get("confidence") is not None else ""
        ax.text(0, 0.86, f" {rec['flag']} ", fontsize=10, weight="bold", color="#000", va="top", transform=ax.transAxes,
                bbox={"boxstyle": "round,pad=0.25", "fc": FLAG_COLOR.get(rec["flag"], GRID), "ec": "none"})
        ax.text(0, 0.70, f"{rec['t_after_onset']:+.1f} s after onset{conf}", fontsize=8, color=INK2, va="top",
                transform=ax.transAxes, family="monospace")      # its own line: a long flag name never covers it
        ax.text(0, 0.57, wrap(rec.get("reason", ""), 70), fontsize=7.5, color=INK, va="top", transform=ax.transAxes)
    else:
        ax.text(0, 0.86, "no recommendation from our race control engine", fontsize=7.5, color=INK, va="top",
                transform=ax.transAxes)
    ax.text(0, 0.42, "EXPOSURE (CARS PAST THE CRASH SITE AT RACING SPEED)", fontsize=7.5, color=INK2, va="top",
            transform=ax.transAxes)
    ax.text(0, 0.30, "\n".join(exposure_lines(inc)), fontsize=7.5, color=INK, va="top", transform=ax.transAxes,
            linespacing=1.4)

    ax = fig.add_subplot(gs[2:, 1])         # mini delay-cost curve
    x = [k * inc["step_s"] for k in range(len(inc["curve"]))]
    ax.step(x, inc["curve"], where="post", color=INK, lw=1.6)
    ax.fill_between(x, inc["curve"], step="post", color=GRID, alpha=0.6, lw=0)
    for e in events(inc):
        if e["name"] != "onset" and 0 <= e["t"] <= inc["max_delay_s"]:
            color, dash = EVENT_STYLE[e["name"]]
            ax.axvline(e["t"], color=color, lw=1.2, ls="--" if dash == "dashed" else "-")
    ax.set_xlim(0, inc["max_delay_s"])
    ax.set_ylim(0, max(max(inc["curve"]), 1) + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("COST OF DELAY: CARS PAST AT RACING SPEED VS FLAG DELAY (s)", loc="left", fontsize=7.5, color=INK2)
    style_axes(ax)

    fig.text(0.035, 0.03, wrap(f"{honesty(inc, doc)} Onset rule: {doc.get('onset_rule', '')}", 230), fontsize=6.2,
             color=INK2, va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, labelsize=7, colors=INK2)


def wrap(text: str, width: int) -> str:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "\n".join(lines + [cur]) if cur else "\n".join(lines)


# ---- index and main

def write_index(cards: list[tuple[dict, str]], out_dir: Path, doc: dict) -> None:
    rows = "".join(f'<li><a href="{name}.html">{html.escape(inc["race"].replace("_", " "))}: {html.escape(inc["driver"])} '
                   f'(car {html.escape(inc["car"])}) at {inc["onset_t"]:.1f} s</a> <a href="{name}.png">png</a></li>'
                   for inc, name in cards)
    (out_dir / "index.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Fast Flag steward cards</title>'
        f'<style>body{{font-family:system-ui,sans-serif;margin:32px;color:{INK}}}li{{margin:4px 0}}</style></head>'
        f'<body><h1>Steward cards</h1><p>{html.escape(doc.get("note", ""))}</p><ul>{rows}</ul></body></html>\n',
        encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Steward cards from docs/lab/delay_cost.json")
    p.add_argument("--in", dest="in_path", default=str(IN_PATH))
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--only", default=None, help="only incidents of this race id")
    a = p.parse_args()
    doc = json.loads(Path(a.in_path).read_text(encoding="utf-8"))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for inc in doc["incidents"]:
        if a.only and inc["race"] != a.only:
            continue
        name = card_name(inc)
        (out_dir / f"{name}.html").write_text(html_card(inc, doc), encoding="utf-8")
        png_card(inc, doc, out_dir / f"{name}.png")
        cards.append((inc, name))
        print(f"{name}: {inc['summary']}")
    write_index(cards, out_dir, doc)
    print(f"{len(cards)} cards in {out_dir} (index.html)")


if __name__ == "__main__":
    main()

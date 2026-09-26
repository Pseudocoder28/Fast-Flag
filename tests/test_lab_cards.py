"""Steward's cards (src/lab/cards.py) rendered from a synthetic incident, no race data needed.
Run: pytest tests/test_lab_cards.py"""

from __future__ import annotations

import json
from pathlib import Path

from src.lab.cards import (
    card_name,
    events,
    evidence_lines,
    exposure_lines,
    html_card,
    label_levels,
    official_lines,
    png_card,
    recommendation,
    svg_mini_curve,
    svg_timeline,
    svg_trace,
)
from src.lab.delay_cost import curve

DOC = {"note": "Replay of historical FastF1 data. Counterfactual: no model of driver reactions.",
       "onset_rule": "speed below 50% of the car's own reference for 1 s", "case_study_races": ["2021_Azerbaijan"],
       "case_study_note": "Out of sample twice."}


def incident(**over) -> dict:
    passes = [{"car": "16", "driver": "LEC", "t_after_onset": 6.0, "speed_kmh": 317.0, "own_normal_kmh": 314.0,
               "pct_of_own_normal": 101.0},
              {"car": "5", "driver": "VET", "t_after_onset": 7.3, "speed_kmh": 311.4, "own_normal_kmh": 321.2,
               "pct_of_own_normal": 96.9}]
    inc = {"race": "2021_Azerbaijan", "car": "18", "driver": "STR", "cars_involved": ["18"], "onset_t": 5274.25,
           "msector": 20, "lap": 31, "crash_location_m": 5288.1, "official_top_flag": "SC", "max_delay_s": 60.0,
           "step_s": 0.5, "racing_share": 0.8,
           "markers": {"our_first_alert": {"t_after_onset": 1.0, "type": "IMPACT", "evidence": "lost 216 km/h in 1 s",
                                           "cars_by_then": 0},
                       "our_escalation": {"t_after_onset": 5.0, "flag": "SC", "reason": "car 18 stopped on track for 3.0 s",
                                          "cars_by_then": 0},
                       "official_yellow": 3.74, "cars_by_official_yellow": 0,
                       "official_escalation": {"t_after_onset": 37.74, "flag": "SC", "cars_by_then": 2}},
           "curve": curve(passes), "passes": passes, "passes_until_s": 60.0, "passes_without_reference": 0,
           "trace": {"t_after_onset": [-1.0, -0.5, 0.0, 0.5, 1.0], "speed_kmh": [300.0, 290.0, 120.0, 20.0, 0.0],
                     "own_normal_kmh": [305.0, 306.0, 307.0, None, 309.0]},
           "detections": [{"t_after_onset": 1.0, "type": "IMPACT", "severity": 1.0, "drivers": ["18"],
                           "evidence": "lost 216 km/h in 1 s (271 to 0 km/h), harder than braking <b>"},
                          {"t_after_onset": 2.0, "type": "STOPPED", "severity": 1.0, "drivers": ["18"],
                           "evidence": "speed 0 km/h vs ref 313 km/h, on the racing line"}],
           "recommendations": [{"t_after_onset": 1.0, "flag": "YELLOW", "confidence": 0.95, "reason": "car 18 impact signature",
                                "message": "YELLOW IN TRACK SECTOR 20"},
                               {"t_after_onset": 5.0, "flag": "SC", "confidence": 0.95, "reason": "car 18 stopped on track for 3.0 s",
                                "message": "SAFETY CAR DEPLOYED"}],
           "official_messages": [{"t_after_onset": 3.74, "flag": "DOUBLE_YELLOW", "message": "DOUBLE YELLOW IN TRACK SECTOR 21"},
                                 {"t_after_onset": 37.74, "flag": "SC", "message": "SAFETY CAR DEPLOYED"}],
           "summary": "s", "png": "x.png"}
    inc.update(over)
    return inc


def test_events_are_time_ordered_and_complete() -> None:
    names = [(e["name"], e["t"]) for e in events(incident())]
    assert names == [("onset", 0.0), ("our first alert", 1.0), ("official yellow", 3.74), ("our recommendation", 5.0),
                     ("official escalation", 37.74)]


def test_events_fall_back_to_the_first_rec_without_an_escalation() -> None:
    inc = incident()
    inc["markers"]["our_escalation"] = None
    ev = {e["name"]: e for e in events(inc)}
    assert ev["our recommendation"]["what"] == "YELLOW" and ev["our recommendation"]["t"] == 1.0
    assert recommendation(inc)["flag"] == "YELLOW"


def test_recommendation_picks_the_escalation_rec_with_its_confidence() -> None:
    r = recommendation(incident())
    assert r["flag"] == "SC" and r["confidence"] == 0.95 and "stopped on track" in r["reason"]


def test_text_sections() -> None:
    inc = incident()
    assert exposure_lines(inc)[0] == "Each second of delay averaged 0.03 cars (2 cars in 60 s)"
    assert "By the official SC (+37.7 s): 2 cars" in exposure_lines(inc)
    assert "By our SC (+5.0 s): 0 cars" in exposure_lines(inc)
    assert evidence_lines(inc)[0].startswith("+1.0 s IMPACT sev 1.00: lost 216 km/h")
    assert official_lines(inc) == ["+3.7 s DOUBLE YELLOW IN TRACK SECTOR 21", "+37.7 s SAFETY CAR DEPLOYED"]
    inc["markers"]["official_escalation"] = None
    assert "No official escalation for this incident" in exposure_lines(inc)


def test_label_levels_push_close_labels_out() -> None:
    ev = [{"t": 0.0}, {"t": 1.0}, {"t": 3.7}, {"t": 5.0}, {"t": 37.7}]
    levels = label_levels(ev, 40.0, 7.2)
    assert levels[0] == (1, 0) and levels[1] == (-1, 0)
    assert levels[2] == (1, 1), "3.7 is within 7.2 s of 0.0 on the upper side"
    assert levels[3] == (-1, 1), "5.0 is within 7.2 s of 1.0 on the lower side"
    assert levels[4] == (1, 0)


def test_svgs_are_inline_and_escape_text() -> None:
    inc = incident()
    for svg in (svg_timeline(inc, 1240, 92), svg_trace(inc, 700, 210), svg_mini_curve(inc, 480, 120)):
        assert svg.startswith("<svg") and svg.endswith("</svg>") and "http" not in svg
    assert "no speed trace" in svg_trace(incident(trace={"t_after_onset": [], "speed_kmh": [], "own_normal_kmh": []}), 700, 210)


def test_html_card_is_self_contained_and_honest() -> None:
    page = html_card(incident(), DOC)
    for needle in ("2021 Azerbaijan: STR (car 18)", "lap 31", "marshal sector 20", "SAFETY CAR DEPLOYED",
                   "Each second of delay averaged", "Replay of historical FastF1 data", "Out of sample twice",
                   "FAST<b>FLAG</b>", "&lt;b&gt;"):
        assert needle in page, needle
    assert "<b>" not in page.split("harder than braking")[1][:20], "stream text must be escaped"
    assert "src=\"http" not in page and "<link" not in page and "<script" not in page


def test_png_card_writes_a_file(tmp_path: Path) -> None:
    out = tmp_path / "card.png"
    png_card(incident(), DOC, out)
    assert out.exists() and out.stat().st_size > 10_000
    assert card_name(incident()) == "card_2021_Azerbaijan_car18_5274"


def test_cards_render_from_a_json_round_trip(tmp_path: Path) -> None:
    inc = json.loads(json.dumps(incident()))
    assert "Cost of delay" in html_card(inc, DOC)


def test_a_stale_clear_is_never_our_recommendation() -> None:
    # 2023 Monaco STR: the only rec in the window was the CLEAR of an earlier incident, 4.75 s before the onset
    inc = incident(recommendations=[{"t_after_onset": -4.75, "flag": "CLEAR", "confidence": 0.95,
                                     "reason": "no flagged car remains in sector", "message": "CLEAR IN TRACK SECTOR 5"}])
    inc["markers"]["our_escalation"] = None
    assert recommendation(inc) is None
    assert "our recommendation" not in [e["name"] for e in events(inc)]
    assert "no recommendation from our race control engine" in html_card(inc, DOC)


def test_an_alert_on_another_car_names_that_car() -> None:
    inc = incident(detections=[{"t_after_onset": 28.25, "type": "ANOMALY", "severity": 0.37, "drivers": ["81"],
                                "evidence": "anomaly score 0.76 (threshold 0.75)"}])
    inc["markers"]["our_first_alert"].update(type="ANOMALY", drivers=["81"], on_onset_car=False)
    assert evidence_lines(inc)[0].startswith("+28.2 s ANOMALY (car 81, same sector) sev 0.37")
    assert any(e["what"] == "ANOMALY (car 81, same sector)" for e in events(inc))

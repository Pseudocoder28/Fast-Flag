"""Delay-cost curve (src/lab/delay_cost.py): the curve maths, the markers and the summary
line on synthetic passes, without loading any race data. Run: pytest tests/test_lab_delay_cost.py"""

from __future__ import annotations

import pytest

from src.ingest.holdout import HoldoutError
from src.lab.delay_cost import (
    CASE_RACES,
    MAX_DELAY_S,
    RACING_SHARE,
    STEP_S,
    cars_by,
    curve,
    delays,
    png_name,
    race_incidents,
    summary_line,
)


def p(car: str, t: float, pct: float | None, speed: float = 250.0) -> dict:
    return {"car": car, "driver": car, "t_after_onset": t, "speed_kmh": speed, "own_normal_kmh": 260.0,
            "pct_of_own_normal": pct}


PASSES = [p("16", 1.0, 101.0), p("5", 3.5, 96.9), p("22", 8.0, 79.9), p("4", 12.25, 90.3), p("77", 12.5, 80.0),
          p("3", 20.0, None), p("44", 59.9, 88.0), p("11", 60.5, 95.0)]


def test_delays_grid() -> None:
    d = delays()
    assert d[0] == 0.0 and d[-1] == MAX_DELAY_S and len(d) == int(MAX_DELAY_S / STEP_S) + 1
    assert d[1] - d[0] == STEP_S


def test_curve_counts_only_racing_speed_passes_within_the_delay() -> None:
    c = curve(PASSES)
    d = delays()
    assert len(c) == len(d)
    assert c[0] == 0                                   # nothing passes at delay 0
    assert c[d.index(1.0)] == 1                        # car 16 at exactly 1.0 s counts at d = 1.0
    assert c[d.index(3.0)] == 1 and c[d.index(3.5)] == 2
    assert c[d.index(8.0)] == 2                        # car 22 at 79.9% is below racing speed
    assert c[d.index(12.0)] == 2 and c[d.index(12.5)] == 4   # car 4 (90.3%) and car 77 (exactly 80%) count
    assert c[d.index(20.0)] == 4                       # car 3 has no reference: not counted
    assert c[-1] == 5                                  # car 44 at 59.9 s counts, car 11 at 60.5 s is outside
    assert c == sorted(c), "the curve never goes down"


def test_curve_share_and_window_are_parameters() -> None:
    assert curve(PASSES, share=0.5)[-1] == 6
    assert curve(PASSES, max_delay_s=10.0, step_s=1.0) == [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]


def test_cars_by() -> None:
    assert cars_by(PASSES, None) is None
    assert cars_by(PASSES, 0.5) == 0
    assert cars_by(PASSES, 12.5) == 4
    assert cars_by(PASSES, 100.0) == 6


def incident(**markers) -> dict:
    m = {"our_first_alert": None, "our_escalation": None, "official_yellow": None, "cars_by_official_yellow": None,
         "official_escalation": None}
    m.update(markers)
    return {"race": "2021_Azerbaijan", "car": "18", "driver": "STR", "onset_t": 5274.25, "max_delay_s": MAX_DELAY_S,
            "step_s": STEP_S, "racing_share": RACING_SHARE, "markers": m, "curve": curve(PASSES), "passes": PASSES}


def test_summary_line_reports_rate_and_markers() -> None:
    inc = incident(our_first_alert={"t_after_onset": 1.0, "type": "IMPACT", "evidence": "x", "cars_by_then": 1},
                   our_escalation={"t_after_onset": 5.0, "flag": "SC", "reason": "r", "cars_by_then": 2},
                   official_yellow=3.7, cars_by_official_yellow=2,
                   official_escalation={"t_after_onset": 37.7, "flag": "SC", "cars_by_then": 4})
    s = summary_line(inc)
    assert s.startswith("2021 Azerbaijan, STR (car 18) at 5274.2 s: each second of delay here averaged 0.08 cars "
                        "(5 cars passed at racing speed in the 60 s after onset)")
    assert "our first alert +1.0 s" in s and "our SC +5.0 s (2 cars by then)" in s
    assert "official yellow +3.7 s (2 cars by then)" in s and "official SC +37.7 s (4 cars by then)" in s


def test_summary_line_is_honest_when_nothing_matched() -> None:
    s = summary_line(incident())
    assert "no official escalation" in s and "our" not in s and "official yellow" not in s


def test_png_name_is_unique_per_onset() -> None:
    assert png_name(incident()) == "delay_cost_2021_Azerbaijan_car18_5274.png"


def test_holdout_races_are_refused_before_any_data_is_read() -> None:
    for rid in ("2026_Azerbaijan", "2026_Spanish"):
        with pytest.raises(HoldoutError):
            race_incidents(rid)
    assert not any(r.startswith("2026") for r in CASE_RACES)

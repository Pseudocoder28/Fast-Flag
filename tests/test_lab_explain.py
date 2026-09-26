"""Plain-English card explanations (src/lab/explain.py): every case says what happened,
honestly, with the card's own numbers. Run: pytest tests/test_lab_explain.py"""

from __future__ import annotations

import copy

from src.lab.explain import GLOSSARY, HOW_TO_READ, story
from tests.test_lab_cards import DOC, incident

from src.lab.cards import html_card


def text(inc: dict) -> str:
    return " ".join(story(inc))


def test_stroll_story_uses_the_card_numbers() -> None:
    s = text(incident())
    assert "STR (car 18) suddenly lost speed on lap 31" in s
    assert "first alert 1.0 second after" in s and "3.7 seconds after the crash began" in s
    assert "our alert was 2.7 seconds earlier than the race control feed" in s
    assert "we recommended the Safety Car 5.0 seconds" in s and "called the Safety Car 37.7 seconds" in s
    assert "gap of 32.7 seconds" in s and "2 cars drove past the crash at racing speed" in s


def test_race_control_first_is_said_plainly() -> None:
    inc = incident()
    inc["markers"]["official_yellow"] = -0.8
    inc["markers"]["official_escalation"].update(t_after_onset=3.3, flag="VSC")
    s = text(inc)
    assert "0.8 seconds before the moment we mark as the start" in s
    assert "race control's feed was 1.8 seconds earlier than our alert" in s
    assert "Race control was 1.7 seconds earlier than our recommendation" in s
    assert "earlier than the race control feed" not in s


def test_every_missing_piece_is_said_not_hidden() -> None:
    inc = incident()
    m = inc["markers"]
    m["our_first_alert"] = None
    m["official_yellow"] = None
    m["our_escalation"] = None
    m["official_escalation"] = None
    s = text(inc)
    assert "Our system raised no alert" in s and "did not show a yellow flag" in s
    assert "Neither we nor race control called" in s


def test_alert_on_another_car_is_named() -> None:
    inc = incident()
    inc["markers"]["our_first_alert"].update(drivers=["81"], on_onset_car=False)
    assert "on car 81 in the same part of the track, not on STR's car itself" in text(inc)


def test_no_cars_and_one_car() -> None:
    inc = incident(curve=[0] * 121)
    assert "No car drove past the crash spot at racing speed" in text(inc)
    one = copy.deepcopy(incident(curve=[0] * 120 + [1]))
    assert "1 car did that" in text(one)


def test_no_jargon_without_a_definition() -> None:
    words = " ".join(t for t, _ in GLOSSARY)
    for term in ("Yellow flag", "Double yellow flag", "Virtual Safety Car", "Safety Car", "Red flag",
                 "Race control", "Marshal sector", "Racing speed"):
        assert term in words
    everything = text(incident()) + " ".join(d for _, d in HOW_TO_READ)
    for jargon in ("IsolationForest", "LightGBM", "onset", "telemetry", "ANOMALY", "msector", "PR-AUC"):
        assert jargon not in everything, jargon
    assert "\u2014" not in everything + " ".join(d for _, d in GLOSSARY), "no em dashes"


def test_card_page_carries_the_explanation_below_the_card() -> None:
    page = html_card(incident(), DOC)
    assert page.index('class="card"') < page.index('class="explain"')
    assert "WHAT AM I LOOKING AT?" in page and "WORDS USED ON THIS CARD" in page and "Honest limits" in page

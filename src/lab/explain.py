"""Plain-English explanations for the steward cards, for readers who have never seen
F1 telemetry: what each chart shows, what happened in this crash, when we alerted
against when race control did, and what the car count means. Built from the numbers
in docs/lab/delay_cost.json only, so every sentence matches its card.

Run: python -m src.lab.explain 2021_Azerbaijan 18    (prints the text for one card)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FLAG_NAME = {"YELLOW": "a yellow flag", "DOUBLE_YELLOW": "a double yellow flag", "VSC": "a Virtual Safety Car",
             "SC": "the Safety Car", "RED": "a red flag"}
FLAG_SHORT = {"YELLOW": "yellow flag", "DOUBLE_YELLOW": "double yellow flag", "VSC": "Virtual Safety Car",
              "SC": "Safety Car", "RED": "red flag"}
SIGNAL = {"IMPACT": "the car lost a huge amount of speed in about a second, the way a crash does",
          "STOPPED": "the car had stopped where cars normally drive fast",
          "SPIN": "the car left its normal path or slowed down sharply",
          "DROPOUT": "the car's data stopped arriving while it was moving",
          "MULTI": "two or more cars were in trouble in the same place",
          "ANOMALY": "the car's driving looked unusual compared with normal racing"}

GLOSSARY = [
    ("Yellow flag", "danger ahead: slow down, no overtaking."),
    ("Double yellow flag", "serious danger: slow down a lot and be ready to stop."),
    ("Virtual Safety Car (VSC)", "every driver must slow to a set, lower speed, everywhere on the track."),
    ("Safety Car (SC)", "a road car leads the whole field slowly so marshals can clear the track safely."),
    ("Red flag", "the race is stopped."),
    ("Race control", "the officials who decide the flags. Their messages are the official feed we compare against."),
    ("Marshal sector", "one of about 20 numbered stretches of the track, each watched by marshals (trackside safety staff)."),
    ("Racing speed", "at least 80% of that car's usual speed at that exact spot, so the driver has not slowed down."),
]

HOW_TO_READ = [
    ("The timeline strip (top)",
     "Each dot is a moment, in seconds after the crash began. Orange: our first alert. Blue: our flag "
     "recommendation. Gold: race control's first yellow flag. Black: race control's bigger response (a "
     "Virtual Safety Car, the Safety Car or a red flag)."),
    ("Speed trace (left)",
     "Left to right: seconds before (negative) and after (positive) the crash began. Bottom to top: speed "
     "in km/h. Blue line: this car's speed. Grey dashed line: its usual speed at the same spot, from its "
     "previous 3 clean laps (laps with no incident and no pit stop). When the blue line drops far below the "
     "grey one, that's the crash."),
    ("Cost of delay (bottom right)",
     "Left to right: how long a flag waits, in seconds after the crash began. Bottom to top: how many cars "
     "had driven past the crash spot at racing speed by then. The line can only go up. The vertical lines "
     "use the same colours as the timeline. The further right race control's line sits, the more cars had "
     "already gone past when it arrived."),
    ("Detector evidence (left, below the trace)",
     "What the measurements showed that made our system react, for example \"lost 216 km/h in 1 s\". A "
     "detector is a check that watches every car's speed and position many times a second."),
]


def secs(x: float) -> str:
    return f"{abs(x):.1f} second" + ("" if round(abs(x), 1) == 1.0 else "s")


def cars(n: int) -> str:
    return f"{n} car" + ("" if n == 1 else "s")


def speeds(inc: dict) -> tuple[float, float] | None:
    """The car's speed just before the crash began and its lowest speed in the 10 s after."""
    tr = inc.get("trace") or {}
    pts = [(t, v) for t, v in zip(tr.get("t_after_onset", []), tr.get("speed_kmh", [])) if v is not None]
    before = [v for t, v in pts if -3.0 <= t <= 0.0]
    after = [v for t, v in pts if 0.0 <= t <= 10.0]
    return (max(before), min(after)) if before and after else None


def what_happened(inc: dict) -> str:
    sp = speeds(inc)
    lap = f" on lap {inc['lap']}" if inc.get("lap") is not None else ""
    text = f"{inc['driver']} (car {inc['car']}) suddenly lost speed{lap}"
    if sp:
        text += f": from about {sp[0]:.0f} km/h to {sp[1]:.0f} km/h"
    return (text + ". We call that moment the start of the crash, and every time on this card is counted from "
            "it: it is 0 on the timeline and on both charts.")


def alerts(inc: dict) -> str:
    m = inc["markers"]
    a = m["our_first_alert"]
    y = m["official_yellow"]
    if a:
        text = f"Our system raised its first alert {secs(a['t_after_onset'])} after the crash began"
        if a.get("on_onset_car", True):
            text += f", because {SIGNAL.get(a['type'], 'something looked wrong')}."
        else:
            text += (f", on car {', '.join(a.get('drivers', []))} in the same part of the track, not on "
                     f"{inc['driver']}'s car itself.")
    else:
        text = "Our system raised no alert for this crash."
    if y is None:
        return text + " Race control did not show a yellow flag for it."
    when = (f"{secs(y)} after the crash began" if y >= 0.05 else
            f"{secs(y)} before the moment we mark as the start" if y <= -0.05 else "at the same moment the crash began")
    text += f" Race control's first yellow flag came {when}."
    if a:
        d = y - a["t_after_onset"]
        if d >= 0.05:
            text += f" So our alert was {secs(d)} earlier than the race control feed."
        elif d <= -0.05:
            text += f" So race control's feed was {secs(d)} earlier than our alert."
        else:
            text += " Both came at the same moment."
    return text


def escalation(inc: dict) -> str:
    m = inc["markers"]
    ours, off = m["our_escalation"], m["official_escalation"]
    if ours and off:
        text = (f"For the bigger response, we recommended {FLAG_NAME[ours['flag']]} {secs(ours['t_after_onset'])} "
                f"after the crash began. Race control called {FLAG_NAME[off['flag']]} "
                f"{secs(off['t_after_onset'])} after it.")
        gap = off["t_after_onset"] - ours["t_after_onset"]
        if gap >= 0.05:
            between = (off.get("cars_by_then") or 0) - (ours.get("cars_by_then") or 0)
            text += (f" That is a gap of {secs(gap)}. During it, {cars(between)} drove past the crash at racing "
                     "speed, before the official call slowed everyone down.")
        elif gap <= -0.05:
            text += f" Race control was {secs(gap)} earlier than our recommendation here."
        return text
    if ours:
        return (f"We recommended {FLAG_NAME[ours['flag']]} {secs(ours['t_after_onset'])} after the crash began. "
                "Race control handled this crash with yellow flags only, with no Virtual Safety Car, Safety Car or "
                "red flag.")
    if off:
        return (f"Race control called {FLAG_NAME[off['flag']]} {secs(off['t_after_onset'])} after the crash began. "
                "Our system did not recommend a Virtual Safety Car, Safety Car or red flag for this crash.")
    return ("Neither we nor race control called a Virtual Safety Car, Safety Car or red flag: this crash was "
            "handled with yellow flags only.")


def car_count(inc: dict) -> str:
    n = inc["curve"][-1]
    window = f"{inc['max_delay_s']:.0f} seconds"
    if n == 0:
        return (f"No car drove past the crash spot at racing speed in the first {window}. The fewer cars pass a "
                "crash at full speed, the safer it is for the marshals and for the drivers.")
    return (f"Why the car count matters: every car that passes a crash at full speed goes past wreckage, "
            f"debris and marshals before its driver has been told to slow down. In the first {window} after this "
            f"crash, {cars(n)} did that, about one every {inc['max_delay_s'] / n:.0f} seconds. Every second a "
            "flag waits adds to that number.")


def honesty_note() -> str:
    return ("Honest limits: this is a replay of historical race data, not a live race. The counts show how the "
            "cars actually drove. We don't claim every driver would have slowed down sooner, only how many were "
            "exposed while the flag was not out. We compare against the race control feed: marshals at the "
            "trackside may already have been waving flags.")


def story(inc: dict) -> list[str]:
    """What happened in this crash, in five short paragraphs."""
    return [what_happened(inc), alerts(inc), escalation(inc), car_count(inc), honesty_note()]


def main() -> None:
    race, car = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2021_Azerbaijan", "18")
    doc = json.loads(Path("docs/lab/delay_cost.json").read_text(encoding="utf-8"))
    for inc in doc["incidents"]:
        if inc["race"] == race and inc["car"] == car:
            print("\n\n".join(story(inc)))
            print()


if __name__ == "__main__":
    main()

"""Holdout races. Training, tuning and the scan must never load these.

Run `python -m src.ingest.holdout` to print the list.
"""

from __future__ import annotations

# (season, FastF1 Location). Location is stable across FastF1 naming quirks:
# in 2026 the Madrid round is called "Spanish Grand Prix" and Barcelona is
# "Barcelona Grand Prix", so event names alone are ambiguous.
HOLDOUT: list[tuple[int, str]] = [
    (2026, "Baku"),    # primary: 2026 Azerbaijan GP, 26 Sept 2026
    (2026, "Madrid"),  # secondary: 2026 Madrid round (Spanish GP)
]

HOLDOUT_RACE_IDS: set[str] = {"2026_Azerbaijan", "2026_Spanish"}


class HoldoutError(RuntimeError):
    """Raised when training or tuning code tries to touch a holdout race."""


def is_holdout(year: int, location: str) -> bool:
    return (int(year), str(location).strip()) in HOLDOUT


def race_id(year: int, event_name: str) -> str:
    """Race id used across the project, e.g. 2024_Singapore."""
    short = event_name.replace("Grand Prix", "").strip().replace(" ", "_")
    return f"{year}_{short}"


def is_holdout_id(rid: str) -> bool:
    return rid in HOLDOUT_RACE_IDS


def assert_not_holdout(year: int | None = None, location: str | None = None,
                       rid: str | None = None) -> None:
    """Call this at the top of any training/tuning loader."""
    if rid is not None and is_holdout_id(rid):
        raise HoldoutError(f"refusing to load holdout race {rid}")
    if year is not None and location is not None and is_holdout(year, location):
        raise HoldoutError(f"refusing to load holdout race {year} {location}")


if __name__ == "__main__":
    for y, loc in HOLDOUT:
        print(y, loc)
    print(sorted(HOLDOUT_RACE_IDS))

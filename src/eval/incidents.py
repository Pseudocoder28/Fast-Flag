"""Official incidents and matching (Section 6.7).

Official events: race control messages with flag YELLOW, DOUBLE_YELLOW, SC, VSC
or RED between lights out and the chequered flag, outside a red flag
suspension (with the field in the pit lane, flags are for marshals and recovery
vehicles, not for cars). BLUE, track limits, DRS and investigations are never
included (src.ingest.official drops them).

Incidents: flags in a sector stay active until race control clears that sector
(or the whole track). A new sector flag joins an incident when its sector is
still active, or when an active sector within 2 sectors was flagged in the last
120 s. SC, VSC and RED join the incident whose sector flags are still shown,
else the incident flagged in the last 120 s, else they start a track-wide one.

Already flagged: race control often re-flags a sector while a car that stopped
earlier is recovered. If our system is still holding an alert for a stopped car
in a matching sector at that moment, the incident counts as matched ("held"),
but it is kept out of the lead-time numbers.

Matching (per incident): our first alert whose car is in a matching marshal
sector (the flagged sector, up to 2 sectors downstream of it, or 1 upstream,
because race control also flags the sectors approaching an incident) within
[t_official - 60 s, t_official + 10 s]. Track-wide-only incidents accept any
sector. Lead time = t_official - t_ours (positive = we were earlier).

Run: python -m src.eval.incidents 2024_Canadian   (prints the incidents)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.ingest.sectors import sector_matches, sector_offset

FEATURES = Path("data/features")
INCIDENT_FLAGS = ("YELLOW", "DOUBLE_YELLOW", "VSC", "SC", "RED")
SEVERITY = {f: i for i, f in enumerate(INCIDENT_FLAGS)}
JOIN_S = 120.0
JOIN_SECTORS = 2
MATCH_BEFORE_S = 60.0
MATCH_AFTER_S = 10.0


@dataclass(eq=False)          # compared by identity, so incidents can live in sets
class Incident:
    t: float                               # first official message
    sectors: set[int] = field(default_factory=set)
    flags: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    t_last: float = 0.0

    @property
    def top_flag(self) -> str:
        return max(self.flags, key=SEVERITY.get)

    @property
    def track_wide_only(self) -> bool:
        return not self.sectors


def race_files(rid: str, root: Path | None = None) -> tuple[list[dict], dict]:
    """Official events and metadata of a built race, from wherever it was built
    (data/features, data/case_studies or data/holdout)."""
    if root is None:
        from src.replay.engine import race_dir
        root = race_dir(rid)
    official = json.loads((root / f"{rid}_official.json").read_text(encoding="utf-8"))
    meta = json.loads((root / f"{rid}_meta.json").read_text(encoding="utf-8"))
    return official, meta


def suspended_times(frame: pd.DataFrame) -> pd.Series:
    """suspended flag per tick time, from the built race frame."""
    return frame.drop_duplicates("t").set_index("t")["suspended"].sort_index()


def was_suspended(susp: pd.Series, t: float) -> bool:
    i = susp.index.searchsorted(t, side="right") - 1
    return bool(susp.iloc[max(i, 0)])


def build_incidents(official: list[dict], meta: dict, susp: pd.Series,
                    n_sectors: int) -> tuple[list[Incident], int]:
    """Group official messages into incidents. Returns (incidents, messages skipped
    because the race was suspended)."""
    incidents: list[Incident] = []
    active: dict[int, Incident] = {}          # sector -> incident while its flag is shown
    skipped = 0
    for e in sorted(official, key=lambda e: e["t"]):
        if not meta["t_start"] <= e["t"] <= meta["t_end"]:
            continue
        if e["flag"] == "CLEAR":
            if e["msector"] is None:
                active.clear()
            else:
                active.pop(e["msector"], None)
            continue
        if e["flag"] not in INCIDENT_FLAGS:
            continue
        if was_suspended(susp, e["t"]):
            skipped += 1
            continue
        inc = find_incident(e, active, incidents, n_sectors)
        if inc is None:
            inc = Incident(t=e["t"])
            incidents.append(inc)
        inc.flags.append(e["flag"])
        inc.messages.append(e)
        inc.t_last = e["t"]
        if e["msector"] is not None:
            inc.sectors.add(e["msector"])
            active[e["msector"]] = inc
    return incidents, skipped


def find_incident(e: dict, active: dict[int, Incident], incidents: list[Incident],
                  n: int) -> Incident | None:
    s = e["msector"]
    if s is not None and s in active:
        return active[s]
    recent = [i for i in incidents if e["t"] - i.t_last <= JOIN_S]
    if s is None:
        # SC / VSC / RED belong to the incident whose sector flags are still shown
        # (e.g. the red flag for a crash that was flagged 3 minutes earlier)
        if active:
            return max(set(active.values()), key=lambda i: i.t_last)
        return recent[-1] if recent else None
    for i in reversed(recent):
        if any(abs(sector_offset(s, a, n)) <= JOIN_SECTORS for a, inc in active.items() if inc is i):
            return i
    return None


def alert_matches(inc: Incident, t: float, msector: int, n: int) -> bool:
    if not inc.t - MATCH_BEFORE_S <= t <= inc.t + MATCH_AFTER_S:
        return False
    return inc.track_wide_only or any(sector_matches(msector, s, n) for s in inc.sectors)


def main() -> None:
    from src.replay.engine import RaceData
    rid = sys.argv[1]
    race = RaceData.load(rid)
    official, meta = race_files(rid)
    incs, skipped = build_incidents(official, meta, suspended_times(race.frame), meta["n_msectors"])
    for i in incs:
        print(f"{i.t:8.1f}  {i.top_flag:<14} sectors {sorted(i.sectors)}  {len(i.messages)} messages")
    print(f"{len(incs)} incidents, {skipped} messages skipped (suspended)")


if __name__ == "__main__":
    main()

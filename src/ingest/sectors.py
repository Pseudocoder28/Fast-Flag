"""Marshal sector geometry shared by the ingest sanity check and eval matching.

Race control flags the incident sector and the sectors approaching it (e.g. a car
stopped in sector 18 gets double yellows in 16 and 17), so a car matches a
flagged sector when it is in that sector, up to 2 sectors downstream of it, or
1 upstream. Sector numbers wrap at the start line.
"""

from __future__ import annotations


def sector_offset(car: int, flagged: int, n: int) -> int:
    """Circular offset from the flagged marshal sector to the car's sector."""
    return (car - flagged + n // 2) % n - n // 2


def sector_matches(car: int, flagged: int, n: int) -> bool:
    return -1 <= sector_offset(car, flagged, n) <= 2

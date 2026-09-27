"""Cars out of the race at the replay time, from the replay's own data (GET /cars).

A car is out of the race when, as of the replay time, it has sent no data for OUT_AFTER_S
(or never sent any), or it has not driven (MOVE_KMH or faster) for OUT_AFTER_S while the
field was moving. The grid and a standing restart never count; a red flag queue only counts
from the restart for cars in the pit lane, while a car stopped out on track through a red
flag is a wreck and goes out after OUT_AFTER_S. The pit wall and the overlay take such cars
off the track and list them, and both ask the server, so they always agree, also right after
a seek. A car is back in the race only when it drives again: a wreck carried off by the
recovery truck or pushed into the garage stays out. Strictly causal: only rows with t up to
the replay time are read.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

MOVE_KMH = 20.0          # a car below this speed is not driving (a recovery truck or marshals move it at 0) ...
OUT_AFTER_S = 60.0       # ... and out of the race after this long (or this long with no data)
STOPPAGE_MIN_S = 60.0    # a field stop this long is a stoppage (red flag, grid): race control flags in it
                         # are for marshals and recovery vehicles, no car is on track


def car_order(drv: str) -> tuple:
    return (not drv.isdigit(), int(drv) if drv.isdigit() else 0, drv)


@dataclass
class Track:
    t: np.ndarray        # times this car had live data, ascending
    speed: np.ndarray
    x: np.ndarray
    y: np.ndarray
    in_pit: np.ndarray
    msector: np.ndarray


class Cars:
    def __init__(self, rows: pd.DataFrame, stop_times: np.ndarray, t_start: float,
                 track_stop_times: np.ndarray | None = None, entries: list[str] | None = None) -> None:
        """rows: one row per (t, drv) with live data (t, drv, speed, x, y, in_pit, msector);
        stop_times: ascending tick times at which the whole field was stopped (red flag, grid);
        track_stop_times: the ones that also hold a car standing out on track (the field on
        track and slow: the grid, a standing restart; not a red flag queue in the pit lane),
        stop_times when not given; entries: every car of the race, also one with no data."""
        self.t_start = t_start
        self.stop_times = np.asarray(stop_times, dtype=float)
        self.track_stop_times = self.stop_times if track_stop_times is None else np.asarray(track_stop_times, float)
        self.tracks: dict[str, Track] = {}
        for drv, g in rows.sort_values("t").groupby("drv", sort=False):
            self.tracks[str(drv)] = Track(g["t"].to_numpy(float), g["speed"].to_numpy(float), g["x"].to_numpy(float),
                                          g["y"].to_numpy(float), g["in_pit"].to_numpy(bool), g["msector"].to_numpy())
        empty = np.array([], dtype=float)
        for drv in entries or []:
            self.tracks.setdefault(str(drv), Track(empty, empty, empty, empty, np.array([], bool), np.array([], int)))
        self.ids = sorted(self.tracks, key=car_order)
        self.data_t = np.unique(rows["t"].to_numpy(float))   # times any car had live data

    @classmethod
    def from_race(cls, race) -> "Cars":
        """From a src.replay.engine.RaceData: live rows as the ticks send them, and the
        field stopped when the race is suspended (red flag, most of the field in the pit
        lane) or most cars on track are below 30 km/h (the grid, a standing restart). A car
        out on track only counts the second kind while most of the field is on track too."""
        f = race.frame
        rows = f.loc[race._live, ["t", "drv", "speed", "x", "y", "in_pit", "msector"]]
        per_t = f.drop_duplicates("t").set_index("t")
        none = pd.Series(False, index=per_t.index)
        susp = per_t["suspended"].astype(bool) if "suspended" in per_t else none
        slow = per_t["field_slow"].fillna(0.0) >= 0.5 if "field_slow" in per_t else none
        in_pit = per_t["field_in_pit"].fillna(0.0) >= 0.5 if "field_in_pit" in per_t else none
        times = per_t.index.to_numpy(float)
        return cls(rows, times[(susp | slow).to_numpy()], float(race.times[0]),
                   track_stop_times=times[(slow & ~in_pit & ~susp).to_numpy()], entries=getattr(race, "cars", None))

    @classmethod
    def from_ticks(cls, ticks: list[dict]) -> "Cars":
        """From tick envelopes' data (the mock server's fixtures)."""
        rows, stops, track_stops = [], [], []
        for tk in ticks:
            cars = tk.get("cars", [])
            for c in cars:
                rows.append((float(tk["t"]), str(c["drv"]), float(c["speed"]), float(c["x"]), float(c["y"]),
                             bool(c["in_pit"]), int(c["msector"])))
            on = [c for c in cars if not c["in_pit"]]
            in_pit = bool(cars) and sum(c["in_pit"] for c in cars) >= len(cars) / 2
            slow = bool(on) and sum(float(c["speed"]) < 30 for c in on) >= len(on) / 2
            if in_pit or slow:
                stops.append(float(tk["t"]))
            if slow and not in_pit:
                track_stops.append(float(tk["t"]))
        frame = pd.DataFrame(rows, columns=["t", "drv", "speed", "x", "y", "in_pit", "msector"])
        t0 = float(ticks[0]["t"]) if ticks else 0.0
        return cls(frame, np.array(sorted(stops)), t0, track_stop_times=np.array(sorted(track_stops)))

    def still_while_moving(self, since: float, t: float, stops: np.ndarray | None = None) -> float:
        """The longest stretch between since and t with the field moving (no field stop in
        it): once a car has stood still that long it stays out, also through a later red
        flag or grid, until it drives again."""
        stops = self.stop_times if stops is None else stops
        a, b = np.searchsorted(stops, [since, t], side="left")
        cuts = np.concatenate(([since], stops[a:b], [t]))
        return float(np.max(np.diff(cuts))) if len(cuts) > 1 else 0.0

    def at(self, t: float) -> dict:
        """The cars out of the race at replay time t, oldest first, and every car of the race."""
        j = int(np.searchsorted(self.data_t, t, side="right")) - 1
        now = float(self.data_t[j]) if j >= 0 else t   # after the data ends (the race is over) nobody goes silent
        i = bisect.bisect_right(self.stop_times, t) - 1
        field_stop_t = float(self.stop_times[i]) if i >= 0 else None
        out = []
        for drv in self.ids:
            tr = self.tracks[drv]
            k = int(np.searchsorted(tr.t, t, side="right")) - 1
            if k < 0:
                if now - self.t_start >= OUT_AFTER_S:
                    out.append({"drv": drv, "why": "no data", "since": None, "msector": None, "x": None, "y": None})
                continue
            seen = float(tr.t[k])
            where = {"msector": int(tr.msector[k]), "x": round(float(tr.x[k]), 1), "y": round(float(tr.y[k]), 1)}
            if now - seen >= OUT_AFTER_S:
                out.append({"drv": drv, "why": "no data", "since": seen, **where})
                continue
            drove = np.flatnonzero(tr.speed[:k + 1] >= MOVE_KMH)
            still_since = float(tr.t[drove[-1] + 1]) if len(drove) and drove[-1] < k else float(tr.t[0])
            if len(drove) and drove[-1] == k:
                continue                                        # driving now
            pit = bool(tr.in_pit[k])
            if self.still_while_moving(still_since, t, self.stop_times if pit else self.track_stop_times) >= OUT_AFTER_S:
                why = "in the pit lane" if pit else "stopped"
                out.append({"drv": drv, "why": why, "since": still_since, **where})
        out.sort(key=lambda o: (o["since"] is None, o["since"] or 0.0))
        return {"t": round(float(t), 2), "field_stop_t": field_stop_t, "out": out, "cars": self.ids,
                "stoppages": self.stoppages(t)}

    def stoppages(self, t: float) -> list[list[float]]:
        """[start, end] of every field stop of STOPPAGE_MIN_S or more up to t (the last one may
        still be going on: it ends at t)."""
        s = self.stop_times[self.stop_times <= t]
        if not len(s):
            return []
        cut = np.flatnonzero(np.diff(s) > 1.0)
        starts, ends = np.concatenate(([s[0]], s[cut + 1])), np.concatenate((s[cut], [s[-1]]))
        return [[round(float(a), 2), round(float(b), 2)] for a, b in zip(starts, ends) if b - a >= STOPPAGE_MIN_S]

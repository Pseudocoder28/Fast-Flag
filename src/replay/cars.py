"""Cars out of the race at the replay time, from the replay's own data (GET /cars).

A car is out of the race when, as of the replay time, it has sent no data for OUT_AFTER_S,
or it has stayed within STILL_MOVE_M of one spot for OUT_AFTER_S while the field was
moving (the grid, and a red flag queue in the pit lane, only count from the restart). The
pit wall and the overlay take such cars off the track and list them, and both ask the
server, so they always agree, also right after a seek. A car that moves again, or whose
data comes back somewhere else, is back in the race. Strictly causal: only rows with t up
to the replay time are read.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

STILL_MOVE_M = 3.0       # a car within this distance of one spot is standing still ...
OUT_AFTER_S = 60.0       # ... and out of the race after this long (or this long with no data)


def car_order(drv: str) -> tuple:
    return (not drv.isdigit(), int(drv) if drv.isdigit() else 0, drv)


@dataclass
class Track:
    t: np.ndarray        # times this car had live data, ascending
    x: np.ndarray
    y: np.ndarray
    in_pit: np.ndarray
    msector: np.ndarray


class Cars:
    def __init__(self, rows: pd.DataFrame, stop_times: np.ndarray, t_start: float) -> None:
        """rows: one row per (t, drv) with live data (t, drv, x, y, in_pit, msector);
        stop_times: ascending tick times at which the whole field was stopped."""
        self.t_start = t_start
        self.stop_times = np.asarray(stop_times, dtype=float)
        self.tracks: dict[str, Track] = {}
        for drv, g in rows.sort_values("t").groupby("drv", sort=False):
            self.tracks[str(drv)] = Track(g["t"].to_numpy(float), g["x"].to_numpy(float), g["y"].to_numpy(float),
                                          g["in_pit"].to_numpy(bool), g["msector"].to_numpy())
        self.ids = sorted(self.tracks, key=car_order)

    @classmethod
    def from_race(cls, race) -> "Cars":
        """From a src.replay.engine.RaceData: live rows as the ticks send them, and the
        field stopped when the race is suspended (red flag, most of the field in the pit
        lane) or most cars on track are below 30 km/h (the grid, a standing restart)."""
        f = race.frame
        rows = f.loc[race._live, ["t", "drv", "x", "y", "in_pit", "msector"]]
        per_t = f.drop_duplicates("t").set_index("t")
        stopped = pd.Series(False, index=per_t.index)
        if "suspended" in per_t:
            stopped |= per_t["suspended"].astype(bool)
        if "field_slow" in per_t:
            stopped |= per_t["field_slow"].fillna(0.0) >= 0.5
        return cls(rows, stopped.index[stopped.to_numpy()].to_numpy(float), float(race.times[0]))

    @classmethod
    def from_ticks(cls, ticks: list[dict]) -> "Cars":
        """From tick envelopes' data (the mock server's fixtures)."""
        rows, stops = [], []
        for tk in ticks:
            cars = tk.get("cars", [])
            for c in cars:
                rows.append((float(tk["t"]), str(c["drv"]), float(c["x"]), float(c["y"]), bool(c["in_pit"]),
                             int(c["msector"])))
            on = [c for c in cars if not c["in_pit"]]
            if cars and (sum(c["in_pit"] for c in cars) >= len(cars) / 2
                         or (on and sum(float(c["speed"]) < 30 for c in on) >= len(on) / 2)):
                stops.append(float(tk["t"]))
        frame = pd.DataFrame(rows, columns=["t", "drv", "x", "y", "in_pit", "msector"])
        t0 = float(ticks[0]["t"]) if ticks else 0.0
        return cls(frame, np.array(sorted(stops)), t0)

    def still_while_moving(self, since: float, t: float) -> float:
        """The longest stretch between since and t with the field moving (no field stop in
        it): once a car has stood still that long it stays out, also through a later red
        flag or grid, until it moves."""
        a, b = np.searchsorted(self.stop_times, [since, t], side="left")
        cuts = np.concatenate(([since], self.stop_times[a:b], [t]))
        return float(np.max(np.diff(cuts))) if len(cuts) > 1 else 0.0

    def at(self, t: float) -> dict:
        """The cars out of the race at replay time t, oldest first, and every car of the race."""
        i = bisect.bisect_right(self.stop_times, t) - 1
        field_stop_t = float(self.stop_times[i]) if i >= 0 else None
        out = []
        for drv in self.ids:
            tr = self.tracks[drv]
            k = int(np.searchsorted(tr.t, t, side="right")) - 1
            if k < 0:
                if t - self.t_start >= OUT_AFTER_S:
                    out.append({"drv": drv, "why": "no data", "since": None, "msector": None, "x": None, "y": None})
                continue
            seen = float(tr.t[k])
            where = {"msector": int(tr.msector[k]), "x": round(float(tr.x[k]), 1), "y": round(float(tr.y[k]), 1)}
            if t - seen >= OUT_AFTER_S:
                out.append({"drv": drv, "why": "no data", "since": seen, **where})
                continue
            far = np.flatnonzero(np.hypot(tr.x[:k + 1] - tr.x[k], tr.y[:k + 1] - tr.y[k]) > STILL_MOVE_M)
            still_since = float(tr.t[far[-1] + 1]) if len(far) else float(tr.t[0])
            if self.still_while_moving(still_since, t) >= OUT_AFTER_S:
                why = "in the pit lane" if bool(tr.in_pit[k]) else "stopped"
                out.append({"drv": drv, "why": why, "since": still_since, **where})
        out.sort(key=lambda o: (o["since"] is None, o["since"] or 0.0))
        return {"t": round(float(t), 2), "field_stop_t": field_stop_t, "out": out, "cars": self.ids}

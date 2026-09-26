"""Per-circuit reference line, speed profile, KD-tree mapping and marshal sectors.

Built from the same weekend's qualifying session, which happens before the race,
so nothing here can leak future race data. All outputs are in metres
(FastF1 X/Y are 1/10 m).

Run: python -m src.ingest.reference 2023 Australia   (prints a summary)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

POS_SCALE = 0.1          # FastF1 position units -> metres
REF_STEP_M = 5.0         # reference line resolution
SPEED_BIN_M = 10.0       # speed profile bin size
MAX_PROFILE_LAPS = 30
QUICK_FACTOR = 1.07      # "reasonably quick": within 107% of the fastest clean lap
PIT_NEAR_M = 4.0         # a position this close to a known pit lane point ...
PIT_MIN_LAT_M = 7.0      # ... and this far off the racing line counts as pit lane
PIT_ROAD_S = 8.0         # entry / exit road driven around the pit timing lines
PIT_LANE_S = 60.0        # pit lane driving between a timing line and the garage


def secs(td: pd.Series) -> np.ndarray:
    return td.dt.total_seconds().to_numpy(float)


def valid_pos(pos: pd.DataFrame) -> pd.DataFrame:
    """Drop placeholder samples: FastF1 reports X = Y = Z = 0 when a car's position
    is lost (e.g. retired into the garage), even with Status 'OnTrack'."""
    return pos[~((pos["X"] == 0) & (pos["Y"] == 0) & (pos["Z"] == 0))]


@dataclass
class TrackRef:
    ref_xy: np.ndarray                  # (N, 2) metres, resampled every REF_STEP_M
    ref_dist: np.ndarray                # (N,) distance along the lap, metres
    length: float                       # lap length, metres
    msectors: list[dict]                # [{"id", "start_dist", "end_dist"}]
    corners: list[dict]                 # [{"number", "x", "y"}]
    speed_profile: np.ndarray = field(default_factory=lambda: np.zeros(0))  # median km/h per bin
    pit_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # pit lane points, metres
    tree: cKDTree | None = None

    def __post_init__(self) -> None:
        if self.tree is None:
            self.tree = cKDTree(self.ref_xy)
        self.pit_xy = np.asarray(self.pit_xy, float).reshape(-1, 2)
        self._pit_tree = cKDTree(self.pit_xy) if len(self.pit_xy) else None

    def near_pit(self, x: np.ndarray, y: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """True where a position lies on the pit lane learned from qualifying."""
        out = np.zeros(len(x), dtype=bool)
        if self._pit_tree is None:
            return out
        pts = np.column_stack([x, y])
        ok = np.isfinite(pts).all(axis=1) & np.isfinite(lat)
        d, _ = self._pit_tree.query(pts[ok])
        out[ok] = (d < PIT_NEAR_M) & (np.abs(lat[ok]) > PIT_MIN_LAT_M)
        return out

    def project(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map metre X/Y to (distance along lap, signed lateral offset)."""
        pts = np.column_stack([x, y])
        ok = np.isfinite(pts).all(axis=1)
        dist = np.full(len(pts), np.nan)
        lat = np.full(len(pts), np.nan)
        if not ok.any():
            return dist, lat
        _, idx = self.tree.query(pts[ok])
        nxt = (idx + 1) % len(self.ref_xy)
        tangent = self.ref_xy[nxt] - self.ref_xy[idx]
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
        rel = pts[ok] - self.ref_xy[idx]
        along = np.einsum("ij,ij->i", rel, tangent)
        dist[ok] = (self.ref_dist[idx] + along) % self.length
        lat[ok] = tangent[:, 0] * rel[:, 1] - tangent[:, 1] * rel[:, 0]
        return dist, lat

    def project_path(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, k: int = 8,
                     max_extra_m: float = 20.0, reset_gap_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        """Like project(), for one car's positions in time order.

        Where two parts of the circuit run close together, the nearest reference
        point can belong to the wrong part. Among the k nearest points within
        max_extra_m of the nearest one, pick the one closest along the lap to the
        car's previous position. Only uses earlier samples, so it stays causal."""
        pts = np.column_stack([x, y])
        ok = np.isfinite(pts).all(axis=1)
        dist = np.full(len(pts), np.nan)
        lat = np.full(len(pts), np.nan)
        if not ok.any():
            return dist, lat
        dd, cand = self.tree.query(pts[ok], k=k)
        valid = dd <= dd[:, :1] + max_extra_m
        spread = np.where(valid, np.abs(circ(self.ref_dist[cand] - self.ref_dist[cand[:, :1]], self.length)), 0)
        chosen = cand[:, 0].copy()
        tt = np.asarray(t, float)[ok]
        for r in np.flatnonzero(spread.max(axis=1) > 50.0):     # ambiguous rows only
            if r == 0 or tt[r] - tt[r - 1] > reset_gap_s:
                continue
            jump = np.abs(circ(self.ref_dist[cand[r]] - self.ref_dist[chosen[r - 1]], self.length))
            jump[~valid[r]] = np.inf
            chosen[r] = cand[r, int(np.argmin(jump))]
        nxt = (chosen + 1) % len(self.ref_xy)
        tangent = self.ref_xy[nxt] - self.ref_xy[chosen]
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
        rel = pts[ok] - self.ref_xy[chosen]
        along = np.einsum("ij,ij->i", rel, tangent)
        dist[ok] = (self.ref_dist[chosen] + along) % self.length
        lat[ok] = tangent[:, 0] * rel[:, 1] - tangent[:, 1] * rel[:, 0]
        return dist, lat

    def msector_of(self, dist: np.ndarray) -> np.ndarray:
        """Marshal sector id for each distance (0 where unknown)."""
        out = np.zeros(len(dist), dtype=int)
        for s in self.msectors:
            a, b = s["start_dist"], s["end_dist"]
            m = (dist >= a) & (dist < b) if a <= b else (dist >= a) | (dist < b)
            out[m] = s["id"]
        return out

    def ref_speed(self, dist: np.ndarray) -> np.ndarray:
        if not len(self.speed_profile):
            return np.full(len(dist), np.nan)
        b = np.clip((np.nan_to_num(dist) // SPEED_BIN_M).astype(int), 0, len(self.speed_profile) - 1)
        out = self.speed_profile[b]
        return np.where(np.isfinite(dist), out, np.nan)

    def to_track_json(self, race: str, step: int = 2) -> dict:
        """Shape of GET /track (Section 7.5)."""
        return {
            "race": race,
            "ref_line": [[round(float(x), 1), round(float(y), 1)] for x, y in self.ref_xy[::step]],
            "msectors": self.msectors,
            "corners": self.corners,
        }

    def to_json(self) -> dict:
        return {"ref_xy": np.round(self.ref_xy, 2).tolist(), "ref_dist": np.round(self.ref_dist, 2).tolist(),
                "length": self.length, "msectors": self.msectors, "corners": self.corners,
                "speed_profile": np.round(self.speed_profile, 1).tolist(),
                "pit_xy": np.round(self.pit_xy).astype(int).tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "TrackRef":
        return cls(np.asarray(d["ref_xy"], float), np.asarray(d["ref_dist"], float), float(d["length"]),
                   d["msectors"], d["corners"], np.asarray(d["speed_profile"], float),
                   np.asarray(d.get("pit_xy", []), float))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrackRef":
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


def circ(d: np.ndarray, length: float) -> np.ndarray:
    """Signed circular difference along the lap, in (-length / 2, length / 2]."""
    return (d + length / 2) % length - length / 2


def clean_laps(laps: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Green-flag, non-pit, reasonably quick laps."""
    m = laps["LapTime"].notna() & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
    m &= laps["TrackStatus"].astype(str) == "1"
    if strict and "IsAccurate" in laps:
        m &= laps["IsAccurate"].fillna(False).astype(bool)
    out = laps[m]
    if len(out):
        out = out[out["LapTime"] <= out["LapTime"].min() * QUICK_FACTOR]
    return out


def reference_laps(laps: pd.DataFrame) -> pd.DataFrame:
    out = clean_laps(laps, strict=True)
    return out if len(out) >= 3 else clean_laps(laps, strict=False)


def resample_line(xy: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    xy = xy[keep]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    grid = np.arange(0.0, cum[-1], step)
    out = np.column_stack([np.interp(grid, cum, xy[:, 0]), np.interp(grid, cum, xy[:, 1])])
    return out, grid


def build_ref_line(session) -> tuple[np.ndarray, np.ndarray, float]:
    fastest = reference_laps(session.laps).sort_values("LapTime").iloc[0]
    pos = fastest.get_pos_data()
    xy = pos[["X", "Y"]].to_numpy(float) * POS_SCALE
    ref_xy, ref_dist = resample_line(xy, REF_STEP_M)
    length = float(ref_dist[-1] + np.linalg.norm(ref_xy[0] - ref_xy[-1]))
    return ref_xy, ref_dist, length


def build_msectors(session, ref: TrackRef) -> list[dict]:
    """Marshal sector k spans from its marker to the next marker along the lap."""
    ms = session.get_circuit_info().marshal_sectors.sort_values("Number")
    d, _ = ref.project(ms["X"].to_numpy(float) * POS_SCALE, ms["Y"].to_numpy(float) * POS_SCALE)
    starts = sorted(zip(d, ms["Number"].astype(int)))
    out = []
    for i, (start, num) in enumerate(starts):
        end = starts[(i + 1) % len(starts)][0]
        out.append({"id": int(num), "start_dist": round(float(start), 1), "end_dist": round(float(end), 1)})
    return sorted(out, key=lambda s: s["id"])


def msectors_in_order(msectors: list[dict]) -> bool:
    """True if sector ids increase along the lap (allowing one wrap at the line)."""
    ids = [s["id"] for s in sorted(msectors, key=lambda s: s["start_dist"])]
    k = ids.index(min(ids))
    return ids[k:] + ids[:k] == sorted(ids)


def build_corners(session) -> list[dict]:
    c = session.get_circuit_info().corners
    return [{"number": int(r.Number), "x": round(float(r.X) * POS_SCALE, 1),
             "y": round(float(r.Y) * POS_SCALE, 1)} for r in c.itertuples()]


def build_speed_profile(session, ref: TrackRef) -> np.ndarray:
    laps = reference_laps(session.laps).sort_values("LapTime").head(MAX_PROFILE_LAPS)
    nbins = int(np.ceil(ref.length / SPEED_BIN_M))
    dists, speeds = [], []
    for _, lap in laps.iterrows():
        try:
            tel = lap.get_telemetry()
        except Exception:
            continue
        d, _ = ref.project(tel["X"].to_numpy(float) * POS_SCALE, tel["Y"].to_numpy(float) * POS_SCALE)
        dists.append(d)
        speeds.append(tel["Speed"].to_numpy(float))
    d, v = np.concatenate(dists), np.concatenate(speeds)
    ok = np.isfinite(d) & np.isfinite(v)
    bins = np.minimum((d[ok] // SPEED_BIN_M).astype(int), nbins - 1)
    prof = pd.Series(v[ok]).groupby(bins).median().reindex(range(nbins))
    return prof.interpolate(limit_direction="both").to_numpy()


def pit_lane_points(session, ref: TrackRef) -> np.ndarray:
    """Pit lane geometry: positions of cars around their pit in / out times that are
    well off the racing line. Built from qualifying, so it is known before the race."""
    laps, pts = session.laps, []
    for drv in session.drivers:
        if drv not in session.pos_data:
            continue
        pos = valid_pos(session.pos_data[drv])
        t = secs(pos["SessionTime"])
        dl = laps[laps["DriverNumber"] == drv]
        m = np.zeros(len(t), dtype=bool)
        for a in dl["PitInTime"].dropna().dt.total_seconds():
            m |= (t >= a - PIT_ROAD_S) & (t <= a + PIT_LANE_S)
        for b in dl["PitOutTime"].dropna().dt.total_seconds():
            m |= (t >= b - PIT_LANE_S) & (t <= b + PIT_ROAD_S)
        pts.append(pos.loc[m, ["X", "Y"]].to_numpy(float) * POS_SCALE)
    xy = np.concatenate(pts) if pts else np.zeros((0, 2))
    _, lat = ref.project(xy[:, 0], xy[:, 1])
    xy = xy[np.abs(lat) > PIT_MIN_LAT_M]
    return np.unique(np.round(xy), axis=0)


def build_track_ref(session, with_profile: bool = True) -> TrackRef:
    ref_xy, ref_dist, length = build_ref_line(session)
    ref = TrackRef(ref_xy, ref_dist, length, [], [])
    ref.msectors = build_msectors(session, ref)
    ref.corners = build_corners(session)
    if with_profile:
        ref.speed_profile = build_speed_profile(session, ref)
    pit = pit_lane_points(session, ref)
    return TrackRef(ref.ref_xy, ref.ref_dist, ref.length, ref.msectors, ref.corners,
                    ref.speed_profile, pit, tree=ref.tree)


def main() -> None:
    import fastf1
    year, event = int(sys.argv[1]), sys.argv[2]
    fastf1.Cache.enable_cache(str(Path("data") / "fastf1_cache"))
    s = fastf1.get_session(year, event, "Q")
    s.load(weather=False, messages=False)
    ref = build_track_ref(s)
    print(f"lap length {ref.length:.0f} m, {len(ref.ref_xy)} ref points, "
          f"{len(ref.msectors)} marshal sectors (in order: {msectors_in_order(ref.msectors)}), "
          f"{len(ref.corners)} corners")
    print(f"pit lane points {len(ref.pit_xy)}")
    print(f"speed profile min/median/max {np.nanmin(ref.speed_profile):.0f}/"
          f"{np.nanmedian(ref.speed_profile):.0f}/{np.nanmax(ref.speed_profile):.0f} km/h")


if __name__ == "__main__":
    main()

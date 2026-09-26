"""Per-circuit reference line, speed profile, KD-tree mapping and marshal sectors.

Run: python -m src.ingest.reference 2023 Australia   (prints a summary)
All outputs are in metres (FastF1 X/Y are 1/10 m).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

POS_SCALE = 0.1          # FastF1 position units -> metres
REF_STEP_M = 5.0         # reference line resolution
SPEED_BIN_M = 10.0       # speed profile bin size
MAX_PROFILE_LAPS = 30


@dataclass
class TrackRef:
    ref_xy: np.ndarray                  # (N, 2) metres, resampled every REF_STEP_M
    ref_dist: np.ndarray                # (N,) distance along the lap, metres
    length: float                       # lap length, metres
    msectors: list[dict]                # [{"id", "start_dist", "end_dist"}]
    corners: list[dict]                 # [{"number", "x", "y"}]
    speed_profile: np.ndarray = field(default_factory=lambda: np.zeros(0))  # median km/h per bin
    tree: cKDTree | None = None

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


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Green-flag, non-pit, accurate laps."""
    m = (laps["TrackStatus"].astype(str) == "1") & laps["PitInTime"].isna() \
        & laps["PitOutTime"].isna() & laps["LapTime"].notna()
    if "IsAccurate" in laps:
        m &= laps["IsAccurate"].fillna(False).astype(bool)
    return laps[m]


def resample_line(xy: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    xy = xy[keep]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    grid = np.arange(0.0, cum[-1], step)
    out = np.column_stack([np.interp(grid, cum, xy[:, 0]), np.interp(grid, cum, xy[:, 1])])
    return out, grid


def build_ref_line(session) -> tuple[np.ndarray, np.ndarray, float]:
    fastest = clean_laps(session.laps).sort_values("LapTime").iloc[0]
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


def build_corners(session) -> list[dict]:
    c = session.get_circuit_info().corners
    return [{"number": int(r.Number), "x": round(float(r.X) * POS_SCALE, 1),
             "y": round(float(r.Y) * POS_SCALE, 1)} for r in c.itertuples()]


def build_speed_profile(session, ref: TrackRef) -> np.ndarray:
    laps = clean_laps(session.laps).sort_values("LapTime").head(MAX_PROFILE_LAPS)
    nbins = int(np.ceil(ref.length / SPEED_BIN_M))
    binned: list[list[float]] = [[] for _ in range(nbins)]
    for _, lap in laps.iterrows():
        try:
            tel = lap.get_telemetry()
        except Exception:
            continue
        d, _ = ref.project(tel["X"].to_numpy(float) * POS_SCALE, tel["Y"].to_numpy(float) * POS_SCALE)
        ok = np.isfinite(d)
        for b, v in zip((d[ok] // SPEED_BIN_M).astype(int), tel["Speed"].to_numpy(float)[ok]):
            binned[min(b, nbins - 1)].append(v)
    prof = np.array([np.median(v) if v else np.nan for v in binned])
    return pd.Series(prof).interpolate(limit_direction="both").to_numpy()


def build_track_ref(session, with_profile: bool = True) -> TrackRef:
    ref_xy, ref_dist, length = build_ref_line(session)
    ref = TrackRef(ref_xy, ref_dist, length, [], [], tree=cKDTree(ref_xy))
    ref.msectors = build_msectors(session, ref)
    ref.corners = build_corners(session)
    if with_profile:
        ref.speed_profile = build_speed_profile(session, ref)
    return ref


def main() -> None:
    import fastf1
    year, event = int(sys.argv[1]), sys.argv[2]
    fastf1.Cache.enable_cache("data/fastf1_cache")
    s = fastf1.get_session(year, event, "R")
    s.load(weather=False)
    ref = build_track_ref(s)
    print(f"lap length {ref.length:.0f} m, {len(ref.ref_xy)} ref points, "
          f"{len(ref.msectors)} marshal sectors, {len(ref.corners)} corners")
    print(f"speed profile min/median/max {np.nanmin(ref.speed_profile):.0f}/"
          f"{np.nanmedian(ref.speed_profile):.0f}/{np.nanmax(ref.speed_profile):.0f} km/h")


if __name__ == "__main__":
    main()

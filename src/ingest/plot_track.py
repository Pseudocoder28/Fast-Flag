"""Plot a reference line coloured by marshal sector, with sector ids, corners and direction.

Run: python -m src.ingest.plot_track 2023_Australian   (reads data/features/<race>_ref.json)
Output: data/plots/<race>_track.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.ingest.reference import TrackRef  # noqa: E402

PLOTS = Path("data/plots")


def sector_label_points(ref: TrackRef, offset: float) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(id, point halfway through the sector, label position beside the track).

    The label sits on the track normal, on the side facing away from the track
    centre, so it stays next to its own segment in tight parts of the circuit."""
    out = []
    n = len(ref.ref_xy)
    centre = ref.ref_xy.mean(axis=0)
    for s in ref.msectors:
        span = (s["end_dist"] - s["start_dist"]) % ref.length
        mid = (s["start_dist"] + span / 2) % ref.length
        i = min(int(np.searchsorted(ref.ref_dist, mid)), n - 1)
        p = ref.ref_xy[i]
        tangent = ref.ref_xy[(i + 1) % n] - ref.ref_xy[(i - 1) % n]
        normal = np.array([-tangent[1], tangent[0]]) / (np.linalg.norm(tangent) + 1e-9)
        if np.dot(normal, p - centre) < 0:
            normal = -normal
        out.append((s["id"], p, p + normal * offset))
    return out


def plot_track(ref: TrackRef, title: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 9))
    ids = ref.msector_of(ref.ref_dist)
    cmap = plt.get_cmap("tab10")
    extent = float(np.ptp(ref.ref_xy, axis=0).max())
    for k, s in enumerate(sorted(ref.msectors, key=lambda s: s["id"])):
        m = ids == s["id"]
        ax.scatter(ref.ref_xy[m, 0], ref.ref_xy[m, 1], s=9, color=cmap(k % 10), linewidths=0)
    labels = sorted(sector_label_points(ref, extent * 0.035), key=lambda x: x[0])
    for k, (sid, p, q) in enumerate(labels):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=cmap(k % 10), lw=0.8)
        ax.annotate(str(sid), q, ha="center", va="center", fontsize=9, color="white", weight="bold",
                    bbox={"boxstyle": "round,pad=0.25", "fc": cmap(k % 10), "ec": "none"})
    for c in ref.corners:
        ax.annotate(f"T{c['number']}", (c["x"], c["y"]), fontsize=6.5, color="0.35",
                    ha="center", va="center")
    p0, p1 = ref.ref_xy[0], ref.ref_xy[min(12, len(ref.ref_xy) - 1)]
    ax.annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 2})
    ax.plot(*p0, marker="s", color="black", markersize=7)
    ax.set_title(f"{title}\nmarshal sectors (coloured), corners (grey), start line and direction (black)",
                 fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    rid = sys.argv[1]
    ref = TrackRef.load(Path("data") / "features" / f"{rid}_ref.json")
    print(plot_track(ref, rid, PLOTS / f"{rid}_track.png"))


if __name__ == "__main__":
    main()

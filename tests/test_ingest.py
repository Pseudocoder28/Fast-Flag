"""Ingest: causal as-of join, no future leakage in features, track geometry helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.sectors import sector_matches as flag_matches
from src.ingest.features import add_car_dynamics
from src.ingest.merge import add_gaps, asof
from src.ingest.reference import TrackRef


def circle_ref(radius: float = 500.0, n: int = 600) -> TrackRef:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack([radius * np.cos(a), radius * np.sin(a)])
    step = 2 * np.pi * radius / n
    msectors = [{"id": 1, "start_dist": 0.0, "end_dist": 1000.0},
                {"id": 2, "start_dist": 1000.0, "end_dist": 2000.0},
                {"id": 3, "start_dist": 2000.0, "end_dist": 0.0}]
    pit = np.column_stack([np.linspace(-100, 100, 50), np.full(50, -radius - 20)])
    return TrackRef(xy, np.arange(n) * step, n * step, msectors, [], np.full(400, 250.0), pit)


def test_asof_never_uses_future_samples() -> None:
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([10.0, 20.0, 30.0])
    out = asof(np.array([0.5, 1.0, 1.5, 3.5]), t, v)
    assert out[:3].tolist() == [10.0, 20.0, 20.0]
    assert np.isnan(out[3])          # latest sample is 1.5 s old: stale, never extrapolated


def test_features_identical_when_future_is_removed() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(0, 60, 0.25)
    rows = []
    for drv in ("1", "44"):
        rows.append(pd.DataFrame({
            "t": t, "drv": drv, "speed": rng.uniform(80, 300, len(t)), "ref_speed": 250.0,
            "lat_off": rng.normal(0, 1, len(t)), "throttle": rng.uniform(0, 100, len(t)),
            "brake": rng.integers(0, 2, len(t)).astype(float), "gap_ahead_m": rng.uniform(5, 200, len(t)),
            "ahead_drv": "44" if drv == "1" else "1"}))
    full = pd.concat(rows, ignore_index=True)
    cut = 30.0
    a = add_car_dynamics(full.copy())
    b = add_car_dynamics(full[full["t"] <= cut].copy())
    cols = ["speed_dev", "speed_ratio", "dspeed_1s", "lat_off_std_2s", "throttle_std_2s", "closing_rate"]
    a = a[a["t"] <= cut].set_index(["t", "drv"]).sort_index()[cols]
    b = b.set_index(["t", "drv"]).sort_index()[cols]
    pd.testing.assert_frame_equal(a, b)


def test_project_msector_and_json_round_trip() -> None:
    ref = circle_ref()
    dist, lat = ref.project(np.array([500.0, 0.0]), np.array([0.0, 510.0]))
    assert abs(dist[0]) < 1 or abs(dist[0] - ref.length) < 1
    assert abs(lat[1]) > 9                  # 10 m outside the circle
    assert ref.msector_of(np.array([10.0, 1500.0, 2500.0])).tolist() == [1, 2, 3]
    back = TrackRef.from_json(ref.to_json())
    assert back.length == ref.length and len(back.pit_xy) == len(ref.pit_xy)


def test_near_pit_needs_pit_lane_and_offset() -> None:
    ref = circle_ref()
    x, y = np.array([0.0, 0.0]), np.array([-520.0, -500.0])
    _, lat = ref.project(x, y)
    assert ref.near_pit(x, y, lat).tolist() == [True, False]


def test_gaps_wrap_around_the_line() -> None:
    df = pd.DataFrame({"t": [0.0] * 3, "drv": ["1", "2", "3"], "dist": [0.0, 100.0, 4000.0],
                       "in_pit": [False] * 3})
    out = add_gaps(df, length=4200.0).set_index("drv")
    assert out.loc["3", "gap_ahead_m"] == 200.0 and out.loc["3", "ahead_drv"] == "1"
    assert out.loc["1", "gap_behind_m"] == 200.0


def test_flag_matches_upstream_sectors() -> None:
    assert flag_matches(18, 16, 20) and flag_matches(18, 18, 20) and flag_matches(17, 18, 20)
    assert not flag_matches(18, 15, 20)
    assert flag_matches(1, 20, 20) and flag_matches(20, 1, 20)   # wraps at the line


def test_project_path_keeps_the_car_on_its_own_part_of_the_track() -> None:
    # hairpin: out along y = 0 to x = 200, back along y = 10 (two parts 10 m apart)
    out = np.column_stack([np.arange(0, 200, 5.0), np.zeros(40)])
    back = np.column_stack([np.arange(200, 0, -5.0), np.full(40, 10.0)])
    xy = np.vstack([out, back])
    ref = TrackRef(xy, np.arange(len(xy)) * 5.0, len(xy) * 5.0 + 10, [], [])
    # a car on the way out that drifts 6 m wide, towards the return leg
    x, y = np.arange(50, 150, 5.0), np.r_[np.zeros(5), np.full(15, 6.0)]
    t = np.arange(20) * 0.25
    naive, _ = ref.project(x, y)
    path, _ = ref.project_path(x, y, t)
    assert (naive > 200).any()                  # nearest point snaps to the return leg
    assert (path < 200).all()                   # continuity keeps it on the way out


def test_backward_position_glitch_is_held() -> None:
    from src.ingest.merge import hold_position_glitches
    f = pd.DataFrame({"x": [0.0, 10, 20, 11, 30], "y": 0.0, "dist": [0.0, 10, 20, 11, 30],
                      "lat_off": 0.0, "speed": 100.0})
    out = hold_position_glitches(f.copy(), length=5000.0)
    assert out["dist"].tolist() == [0.0, 10, 20, 20, 30]      # the out-of-order sample is replaced


def test_safety_car_messages_of_every_season() -> None:
    from src.ingest.official import classify

    def row(msg: str) -> pd.Series:
        return pd.Series({"Category": "SafetyCar", "Message": msg, "Flag": None})

    assert classify(row("SAFETY CAR DEPLOYED")) == "SC"
    assert classify(row("VIRTUAL SAFETY CAR DEPLOYED")) == "VSC"       # 2023 to 2025
    assert classify(row("VSC DEPLOYED")) == "VSC"                       # 2026
    for ignored in ("VSC ENDING", "VIRTUAL SAFETY CAR ENDING", "SAFETY CAR IN THIS LAP"):
        assert classify(row(ignored)) is None

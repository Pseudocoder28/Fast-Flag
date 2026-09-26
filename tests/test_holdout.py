"""The holdout races must be refused by training and tuning loaders."""

import pytest

from src.ingest.holdout import HoldoutError, assert_not_holdout, is_holdout, race_id


def test_holdout_list() -> None:
    assert is_holdout(2026, "Baku") and is_holdout(2026, "Madrid")
    assert not is_holdout(2025, "Baku") and not is_holdout(2026, "Barcelona")
    assert race_id(2026, "Azerbaijan Grand Prix") == "2026_Azerbaijan"


def test_guard_refuses_holdout() -> None:
    with pytest.raises(HoldoutError):
        assert_not_holdout(2026, "Baku")
    with pytest.raises(HoldoutError):
        assert_not_holdout(rid="2026_Spanish")
    assert_not_holdout(2023, "Melbourne")


def test_scan_skipped_holdout() -> None:
    from pathlib import Path
    csv = Path("data/race_ranking.csv")
    if not csv.exists():
        pytest.skip("scan not run on this machine")
    text = csv.read_text(encoding="utf-8")
    assert "2026_Azerbaijan" not in text and "2026_Spanish" not in text


def test_holdout_runner_detects_changes_and_runs_once() -> None:
    from src.eval.holdout import check_once, verify
    manifest = {"commit": "abc1234def", "models": {"risk_10.txt": "h1", "anomaly.joblib": "h2"}}
    assert verify(manifest, [], {"risk_10.txt": "h1", "anomaly.joblib": "h2"}) == []
    problems = verify(manifest, ["src/detect/detectors.py"], {"risk_10.txt": "CHANGED", "anomaly.joblib": "h2"})
    assert len(problems) == 2 and "src/detect/detectors.py" in problems[0] and "risk_10.txt" in problems[1]
    assert check_once([], None) is None
    runs = [{"at": "2026-09-27T06:00:00", "commit": "abc1234def", "reason": None}]
    assert "already ran" in check_once(runs, None)
    assert check_once(runs, "report writer crashed before writing") is None


def test_holdout_run_and_dry_run_refuse_the_wrong_races() -> None:
    from src.eval.holdout import dry_run, run
    with pytest.raises(SystemExit):
        run("2023_Australian", None)            # run is for holdout races only
    with pytest.raises(SystemExit):
        dry_run("2026_Azerbaijan")              # a holdout never goes through dry-run

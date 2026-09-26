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

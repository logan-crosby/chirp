"""Unit tests for the scheduler time-window helper."""
from datetime import time
from unittest.mock import patch

import pytest

from chirp.scheduler import is_within_window


def mock_now(hour: int, minute: int = 0):
    """Return a context manager that patches datetime.now to return a fixed time."""
    from datetime import datetime
    fixed = datetime(2024, 6, 1, hour, minute, 0)
    return patch("chirp.scheduler.datetime", wraps=datetime,
                 **{"now.return_value": fixed})


def test_no_schedule_always_active():
    assert is_within_window(None, None) is True
    assert is_within_window(None, "20:00") is True
    assert is_within_window("06:00", None) is True


def test_within_window():
    with mock_now(10):
        assert is_within_window("06:00", "20:00") is True


def test_before_window():
    with mock_now(5):
        assert is_within_window("06:00", "20:00") is False


def test_after_window():
    with mock_now(21):
        assert is_within_window("06:00", "20:00") is False


def test_at_window_start():
    with mock_now(6, 0):
        assert is_within_window("06:00", "20:00") is True


def test_at_window_end():
    with mock_now(20, 0):
        assert is_within_window("06:00", "20:00") is True


def test_overnight_window_active_before_midnight():
    with mock_now(23):
        assert is_within_window("22:00", "06:00") is True


def test_overnight_window_active_after_midnight():
    with mock_now(3):
        assert is_within_window("22:00", "06:00") is True


def test_overnight_window_inactive_midday():
    with mock_now(12):
        assert is_within_window("22:00", "06:00") is False

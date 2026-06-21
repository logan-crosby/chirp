"""Unit tests for SessionMetrics (in-memory live stats)."""
from datetime import datetime, timedelta

import pytest

from chirp.metrics import SessionMetrics


NOW = datetime(2024, 1, 1, 12, 0, 0)


def test_initial_total_is_zero():
    m = SessionMetrics()
    assert m.total_visits == 0


def test_record_arrival_increments_total():
    m = SessionMetrics()
    m.record_arrival(1, "chickadee", "feeder", NOW)
    m.record_arrival(2, "finch", "feeder", NOW)
    assert m.total_visits == 2


def test_species_counts_accumulated():
    m = SessionMetrics()
    m.record_arrival(1, "chickadee", "feeder", NOW)
    m.record_arrival(2, "chickadee", "feeder", NOW)
    m.record_arrival(3, "robin", "feeder", NOW)
    assert m.species_counts["chickadee"] == 2
    assert m.species_counts["robin"] == 1


def test_none_species_stored_as_unknown():
    m = SessionMetrics()
    m.record_arrival(1, None, "full_frame", NOW)
    assert m.species_counts["unknown"] == 1


def test_visit_duration_computed():
    m = SessionMetrics()
    t0 = NOW
    t1 = NOW + timedelta(seconds=45)
    m.record_arrival(1, "wren", "feeder", t0)
    m.record_departure(1, t1)
    assert len(m.visit_durations_s) == 1
    assert abs(m.visit_durations_s[0] - 45.0) < 0.01


def test_avg_visit_duration_none_when_no_departures():
    m = SessionMetrics()
    m.record_arrival(1, "wren", "feeder", NOW)
    assert m.avg_visit_duration_s() is None


def test_avg_visit_duration_multiple():
    m = SessionMetrics()
    for i, dur in enumerate([10, 20, 30]):
        t0 = NOW + timedelta(hours=i)
        m.record_arrival(i, "jay", "feeder", t0)
        m.record_departure(i, t0 + timedelta(seconds=dur))
    assert abs(m.avg_visit_duration_s() - 20.0) < 0.01


def test_species_table_lines_sorted_by_count():
    m = SessionMetrics()
    for _ in range(3):
        m.record_arrival(len(m.species_counts), "sparrow", "f", NOW)
    m.record_arrival(99, "hawk", "f", NOW)
    lines = m.species_table_lines()
    assert "sparrow" in lines[0]
    assert "hawk" in lines[1]


def test_summary_text_contains_total():
    m = SessionMetrics()
    m.record_arrival(1, "dove", "feeder", NOW)
    text = m.summary_text()
    assert "1" in text
    assert "dove" in text


def test_departure_for_unknown_tracker_is_ignored():
    m = SessionMetrics()
    m.record_departure(999, NOW)  # Should not raise
    assert m.visit_durations_s == []

"""Unit tests for ZoneMonitor."""
import numpy as np
import pytest
import supervision as sv
from datetime import datetime

from chirp.zone_monitor import ZoneMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_det(tracker_id: int, class_id: int = 0) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[10, 10, 30, 30]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([class_id]),
        tracker_id=np.array([tracker_id]),
    )


def concat(*dets: sv.Detections) -> sv.Detections:
    """Merge multiple single-detection objects into one Detections."""
    if not dets:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.vstack([d.xyxy for d in dets]),
        confidence=np.concatenate([d.confidence for d in dets]),
        class_id=np.concatenate([d.class_id for d in dets]),
        tracker_id=np.concatenate([d.tracker_id for d in dets]),
    )


NOW = datetime(2024, 1, 1, 12, 0, 0)
EMPTY = sv.Detections.empty()


# ---------------------------------------------------------------------------
# Arrival tests
# ---------------------------------------------------------------------------

def test_no_arrival_below_threshold():
    m = ZoneMonitor(in_threshold=5, out_timeout=20)
    total_entered = 0
    for i in range(4):
        entered, _ = m.update(make_det(1), i, NOW)
        total_entered += len(entered)
    assert total_entered == 0


def test_arrival_fires_exactly_at_threshold():
    m = ZoneMonitor(in_threshold=5, out_timeout=20)
    entered_frames = []
    for i in range(8):
        entered, _ = m.update(make_det(1), i, NOW)
        if entered:
            entered_frames.append(i)
    assert entered_frames == [4]   # fires on frame index 4 (5th frame)


def test_no_double_count_for_long_visit():
    m = ZoneMonitor(in_threshold=3, out_timeout=20)
    total = 0
    for i in range(50):
        entered, _ = m.update(make_det(1), i, NOW)
        total += len(entered)
    assert total == 1


def test_arrival_event_contains_correct_tracker_id():
    m = ZoneMonitor(in_threshold=2, out_timeout=10)
    for i in range(2):
        entered, _ = m.update(make_det(tracker_id=42, class_id=7), i, NOW)
    assert entered[0]["tracker_id"] == 42


# ---------------------------------------------------------------------------
# Departure tests
# ---------------------------------------------------------------------------

def test_exit_fires_after_timeout():
    m = ZoneMonitor(in_threshold=2, out_timeout=3)
    # Confirm arrival
    for i in range(2):
        m.update(make_det(1), i, NOW)
    # Bird leaves
    total_exited = 0
    for i in range(2, 10):
        _, exited = m.update(EMPTY, i, NOW)
        total_exited += len(exited)
    assert total_exited == 1


def test_no_exit_for_unconfirmed_track():
    """A track that never reached in_threshold should not generate an exit event."""
    m = ZoneMonitor(in_threshold=10, out_timeout=3)
    for i in range(2):           # only 2 frames, below threshold of 10
        m.update(make_det(1), i, NOW)
    total_exited = 0
    for i in range(2, 20):
        _, exited = m.update(EMPTY, i, NOW)
        total_exited += len(exited)
    assert total_exited == 0


def test_brief_absence_does_not_trigger_exit():
    m = ZoneMonitor(in_threshold=2, out_timeout=5)
    for i in range(2):
        m.update(make_det(1), i, NOW)
    # 3 frames absent (< timeout of 5)
    for i in range(2, 5):
        _, exited = m.update(EMPTY, i, NOW)
        assert len(exited) == 0
    # Bird returns
    _, exited = m.update(make_det(1), 5, NOW)
    assert len(exited) == 0


# ---------------------------------------------------------------------------
# Species / majority vote tests
# ---------------------------------------------------------------------------

def test_species_majority_vote_overrides_first_class():
    """Exit event species should reflect the majority over the *full* visit.

    The enter event fires at in_threshold frames; by that point only a small
    sample has been seen. The exit event fires after the full track lifetime,
    so its species_class_id reflects the true majority.
    """
    m = ZoneMonitor(in_threshold=3, out_timeout=3)
    # 2 frames class=1 (misclassification), then 8 frames class=0 (correct)
    for i in range(10):
        cls = 1 if i < 2 else 0
        m.update(make_det(tracker_id=99, class_id=cls), i, NOW)
    # Bird leaves; collect exit event which has full-visit majority
    all_exited = []
    for i in range(10, 25):
        _, exited = m.update(EMPTY, i, NOW)
        all_exited.extend(exited)
    assert len(all_exited) == 1
    assert all_exited[0]["species_class_id"] == 0  # 0 saw 8 frames vs 2 for class 1


def test_species_majority_vote_on_exit():
    m = ZoneMonitor(in_threshold=2, out_timeout=3)
    class_sequence = [5, 5, 5, 3, 3]
    for i, cls in enumerate(class_sequence):
        m.update(make_det(tracker_id=7, class_id=cls), i, NOW)
    total_exited = 0
    for i in range(5, 12):
        _, exited = m.update(EMPTY, i, NOW)
        total_exited += len(exited)
    assert total_exited == 1


# ---------------------------------------------------------------------------
# Multi-bird tests
# ---------------------------------------------------------------------------

def test_two_birds_counted_independently():
    m = ZoneMonitor(in_threshold=2, out_timeout=10)
    for i in range(3):
        m.update(concat(make_det(1), make_det(2)), i, NOW)
    # Both should have entered
    entered_ids = set()
    # Collect from the first 3 updates
    m2 = ZoneMonitor(in_threshold=2, out_timeout=10)
    for i in range(3):
        entered, _ = m2.update(concat(make_det(1), make_det(2)), i, NOW)
        for e in entered:
            entered_ids.add(e["tracker_id"])
    assert entered_ids == {1, 2}


def test_second_bird_arrival_does_not_reset_first():
    m = ZoneMonitor(in_threshold=3, out_timeout=10)
    total = 0
    for i in range(3):
        entered, _ = m.update(make_det(1), i, NOW)
        total += len(entered)
    # Second bird joins
    for i in range(3, 6):
        entered, _ = m.update(concat(make_det(1), make_det(2)), i, NOW)
        total += len(entered)
    assert total == 2  # bird 1 + bird 2, not bird 1 twice

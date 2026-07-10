"""Regression tests for visit-duration event pairing."""
from datetime import datetime, timezone

from chirp.database import ChirpDatabase


def _create_session(db: ChirpDatabase) -> int:
    return db.create_session(
        started_at=datetime(2024, 6, 1, 8, 0, tzinfo=timezone.utc),
        location="40.71,-74.00",
        model_source="yolo11n.pt",
        confidence_threshold=0.35,
        iou_threshold=0.5,
        in_threshold_frames=15,
        track_buffer_seconds=8,
    )


def _log_event(
    db: ChirpDatabase,
    session_id: int,
    event_type: str,
    occurred_at: datetime,
    *,
    tracker_id: int = 7,
    zone_name: str = "feeder",
) -> None:
    db.log_event(
        session_id=session_id,
        event_type=event_type,
        occurred_at=occurred_at,
        frame_number=1,
        zone_name=zone_name,
        tracker_id=tracker_id,
        class_id=0,
        class_name="jay",
        frames_seen=30,
        class_id_counts={0: 30},
    )


def test_visit_durations_pairs_repeated_tracker_visits_in_order():
    db = ChirpDatabase(":memory:")
    session_id = _create_session(db)

    _log_event(
        db,
        session_id,
        "enter",
        datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    _log_event(
        db,
        session_id,
        "exit",
        datetime(2024, 6, 1, 12, 0, 10, tzinfo=timezone.utc),
    )
    _log_event(
        db,
        session_id,
        "enter",
        datetime(2024, 6, 1, 12, 1, 0, tzinfo=timezone.utc),
    )
    _log_event(
        db,
        session_id,
        "exit",
        datetime(2024, 6, 1, 12, 1, 20, tzinfo=timezone.utc),
    )

    rows = db.visit_durations(session_id=session_id)

    assert len(rows) == 2
    assert [row["duration_seconds"] for row in rows] == [20.0, 10.0]
    assert all(row["duration_seconds"] >= 0 for row in rows)
    db.close()


def test_visit_durations_ignores_arrival_without_departure():
    db = ChirpDatabase(":memory:")
    session_id = _create_session(db)
    _log_event(
        db,
        session_id,
        "enter",
        datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert db.visit_durations(session_id=session_id) == []
    db.close()

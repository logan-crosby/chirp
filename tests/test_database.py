"""Unit tests for ChirpDatabase."""
import pytest
from datetime import datetime, timezone

from chirp.database import ChirpDatabase


@pytest.fixture
def db():
    """In-memory SQLite database, fresh for each test."""
    d = ChirpDatabase(":memory:")
    yield d
    d.close()


@pytest.fixture
def session_id(db):
    return db.create_session(
        started_at=datetime(2024, 6, 1, 8, 0, tzinfo=timezone.utc),
        location="40.71,-74.00",
        model_source="yolo11n.pt",
        confidence_threshold=0.35,
        iou_threshold=0.5,
        in_threshold_frames=15,
        track_buffer_seconds=8,
    )


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------

def test_create_session_returns_int(db):
    sid = db.create_session(
        started_at=datetime.now(tz=timezone.utc),
        location=None,
        model_source="test.pt",
        confidence_threshold=0.5,
        iou_threshold=0.5,
        in_threshold_frames=10,
        track_buffer_seconds=5,
    )
    assert isinstance(sid, int)
    assert sid >= 1


def test_sessions_are_unique(db):
    sid1 = db.create_session(
        started_at=datetime.now(tz=timezone.utc),
        location=None, model_source="a.pt",
        confidence_threshold=0.3, iou_threshold=0.5,
        in_threshold_frames=10, track_buffer_seconds=5,
    )
    sid2 = db.create_session(
        started_at=datetime.now(tz=timezone.utc),
        location=None, model_source="b.pt",
        confidence_threshold=0.3, iou_threshold=0.5,
        in_threshold_frames=10, track_buffer_seconds=5,
    )
    assert sid1 != sid2


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

def _log(db, session_id, event_type="enter", class_name="chickadee",
         class_id=0, tracker_id=1, occurred_at=None):
    if occurred_at is None:
        occurred_at = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    return db.log_event(
        session_id=session_id,
        event_type=event_type,
        occurred_at=occurred_at,
        frame_number=1,
        zone_name="full_frame",
        tracker_id=tracker_id,
        class_id=class_id,
        class_name=class_name,
        frames_seen=30,
        class_id_counts={class_id: 30},
    )


def test_log_event_returns_id(db, session_id):
    eid = _log(db, session_id)
    assert isinstance(eid, int)
    assert eid >= 1


def test_total_visits_counts_enter_events(db, session_id):
    _log(db, session_id, "enter")
    _log(db, session_id, "enter")
    _log(db, session_id, "exit")
    assert db.total_visits(session_id=session_id) == 2


def test_total_visits_scoped_to_session(db, session_id):
    sid2 = db.create_session(
        started_at=datetime.now(tz=timezone.utc),
        location=None, model_source="x.pt",
        confidence_threshold=0.3, iou_threshold=0.5,
        in_threshold_frames=10, track_buffer_seconds=5,
    )
    _log(db, session_id, "enter")
    _log(db, sid2, "enter")
    _log(db, sid2, "enter")
    assert db.total_visits(session_id=session_id) == 1
    assert db.total_visits(session_id=sid2) == 2


# ---------------------------------------------------------------------------
# Species summary
# ---------------------------------------------------------------------------

def test_species_summary_groups_by_name(db, session_id):
    _log(db, session_id, "enter", class_name="chickadee", class_id=1, tracker_id=1)
    _log(db, session_id, "enter", class_name="chickadee", class_id=1, tracker_id=2)
    _log(db, session_id, "enter", class_name="finch", class_id=2, tracker_id=3)

    summary = db.species_summary(session_id=session_id)
    assert len(summary) == 2
    by_name = {r["species"]: r["visits"] for r in summary}
    assert by_name["chickadee"] == 2
    assert by_name["finch"] == 1


def test_species_summary_excludes_exit_events(db, session_id):
    _log(db, session_id, "enter", class_name="robin", tracker_id=1)
    _log(db, session_id, "exit", class_name="robin", tracker_id=1)

    summary = db.species_summary(session_id=session_id)
    assert len(summary) == 1
    assert summary[0]["visits"] == 1


def test_species_summary_ordered_by_visits_desc(db, session_id):
    for i in range(3):
        _log(db, session_id, class_name="sparrow", tracker_id=i)
    _log(db, session_id, class_name="cardinal", tracker_id=10)

    summary = db.species_summary(session_id=session_id)
    assert summary[0]["species"] == "sparrow"


# ---------------------------------------------------------------------------
# Zone definitions
# ---------------------------------------------------------------------------

def test_log_zone_def(db, session_id):
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    db.log_zone_def(session_id, "feeder", poly)
    conn = db.get_connection()
    row = conn.execute(
        "SELECT zone_name FROM zone_defs WHERE session_id=?", (session_id,)
    ).fetchone()
    assert row["zone_name"] == "feeder"


# ---------------------------------------------------------------------------
# Recent visitors
# ---------------------------------------------------------------------------

def test_recent_visitors_most_recent_first(db, session_id):
    from datetime import timedelta
    base = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
    _log(db, session_id, occurred_at=base, class_name="wren", tracker_id=1)
    _log(db, session_id, occurred_at=base + timedelta(minutes=5),
         class_name="jay", tracker_id=2)

    visitors = db.recent_visitors(limit=5)
    assert visitors[0]["class_name"] == "jay"
    assert visitors[1]["class_name"] == "wren"

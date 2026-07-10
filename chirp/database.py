"""SQLite-backed persistence for chirp sessions and bird events.

Schema
------
sessions      – one row per camera session
zone_defs     – polygon zone definitions per session
bird_events   – enter/exit events per tracker per zone

The raw sqlite3 connection is exposed via `get_connection()` so callers can
run arbitrary queries when the high-level API isn't enough.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Session:
    id: int
    started_at: str
    location: Optional[str]
    model_source: str
    confidence_threshold: float
    iou_threshold: float
    in_threshold_frames: int
    track_buffer_seconds: int


@dataclass
class BirdEvent:
    id: int
    session_id: int
    event_type: str          # "enter" | "exit"
    occurred_at: str         # ISO datetime string
    frame_number: int
    zone_name: str
    tracker_id: int
    class_id: Optional[int]
    class_name: Optional[str]
    frames_seen: int
    class_id_counts: Dict[int, int]


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at            TEXT    NOT NULL,
    location              TEXT,
    model_source          TEXT    NOT NULL,
    confidence_threshold  REAL    NOT NULL,
    iou_threshold         REAL    NOT NULL,
    in_threshold_frames   INTEGER NOT NULL,
    track_buffer_seconds  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS zone_defs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    zone_name   TEXT    NOT NULL,
    polygon     TEXT    NOT NULL    -- JSON array of [x,y] pairs
);

CREATE TABLE IF NOT EXISTS bird_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL REFERENCES sessions(id),
    event_type       TEXT    NOT NULL CHECK (event_type IN ('enter','exit')),
    occurred_at      TEXT    NOT NULL,
    frame_number     INTEGER NOT NULL,
    zone_name        TEXT    NOT NULL,
    tracker_id       INTEGER NOT NULL,
    class_id         INTEGER,
    class_name       TEXT,
    frames_seen      INTEGER NOT NULL DEFAULT 0,
    class_id_counts  TEXT    NOT NULL DEFAULT '{}'   -- JSON
);

CREATE INDEX IF NOT EXISTS idx_events_session
    ON bird_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at
    ON bird_events(occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_class_name
    ON bird_events(class_name);
CREATE INDEX IF NOT EXISTS idx_events_type
    ON bird_events(event_type);
"""


class ChirpDatabase:
    """High-level interface to the chirp SQLite database.

    Usage::

        db = ChirpDatabase("~/chirp_data/chirp.db")
        session_id = db.create_session(...)
        db.log_zone_def(session_id, "feeder", polygon_points)
        db.log_event(session_id, event_type="enter", ...)
        summary = db.species_summary(days=7)
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._path = str(db_path) if db_path != ":memory:" else ":memory:"
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._apply_schema()

    # ------------------------------------------------------------------
    # Low-level access
    # ------------------------------------------------------------------

    def get_connection(self) -> sqlite3.Connection:
        """Return the underlying connection for custom queries."""
        return self._conn

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self,
                       started_at: datetime,
                       location: Optional[str],
                       model_source: str,
                       confidence_threshold: float,
                       iou_threshold: float,
                       in_threshold_frames: int,
                       track_buffer_seconds: int) -> int:
        """Insert a new session row and return its ID."""
        cur = self._conn.execute(
            """INSERT INTO sessions
               (started_at, location, model_source, confidence_threshold,
                iou_threshold, in_threshold_frames, track_buffer_seconds)
               VALUES (?,?,?,?,?,?,?)""",
            (started_at.isoformat(), location, model_source,
             confidence_threshold, iou_threshold, in_threshold_frames,
             track_buffer_seconds),
        )
        self._conn.commit()
        return cur.lastrowid

    def log_zone_def(self,
                     session_id: int,
                     zone_name: str,
                     polygon: List[Any]) -> None:
        """Record a zone polygon definition for a session."""
        self._conn.execute(
            "INSERT INTO zone_defs (session_id, zone_name, polygon) VALUES (?,?,?)",
            (session_id, zone_name, json.dumps(polygon)),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log_event(self,
                  session_id: int,
                  event_type: str,
                  occurred_at: datetime,
                  frame_number: int,
                  zone_name: str,
                  tracker_id: int,
                  class_id: Optional[int],
                  class_name: Optional[str],
                  frames_seen: int,
                  class_id_counts: Dict[int, int]) -> int:
        """Insert a bird enter/exit event. Returns the new row ID."""
        cur = self._conn.execute(
            """INSERT INTO bird_events
               (session_id, event_type, occurred_at, frame_number, zone_name,
                tracker_id, class_id, class_name, frames_seen, class_id_counts)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (session_id, event_type, occurred_at.isoformat(), frame_number,
             zone_name, tracker_id, class_id, class_name, frames_seen,
             json.dumps({str(k): v for k, v in class_id_counts.items()})),
        )
        self._conn.commit()
        return cur.lastrowid

    # ------------------------------------------------------------------
    # Analytics queries
    # ------------------------------------------------------------------

    def species_summary(self, session_id: Optional[int] = None,
                        days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return visit counts grouped by species.

        Filters to "enter" events (each arrival = one visit).
        Optionally restrict to a single session or the last N days.
        """
        conditions = ["event_type='enter'"]
        params: list = []
        if session_id is not None:
            conditions.append("session_id=?")
            params.append(session_id)
        if days is not None:
            conditions.append("occurred_at >= datetime('now', ?)")
            params.append(f"-{days} days")
        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"""SELECT COALESCE(class_name,'unknown') AS species,
                       COUNT(*) AS visits
                FROM bird_events
                WHERE {where}
                GROUP BY species
                ORDER BY visits DESC""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def hourly_activity(self, days: int = 7) -> List[Dict[str, Any]]:
        """Return visit counts by hour of day over the last N days."""
        rows = self._conn.execute(
            """SELECT strftime('%H', occurred_at) AS hour,
                      COUNT(*) AS visits
               FROM bird_events
               WHERE event_type='enter'
                 AND occurred_at >= datetime('now', ?)
               GROUP BY hour
               ORDER BY hour""",
            (f"-{days} days",),
        ).fetchall()
        return [dict(r) for r in rows]

    def recent_visitors(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent bird arrival events."""
        rows = self._conn.execute(
            """SELECT occurred_at, zone_name, class_name, tracker_id
               FROM bird_events
               WHERE event_type='enter'
               ORDER BY occurred_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def total_visits(self, session_id: Optional[int] = None) -> int:
        """Total bird arrivals (optionally scoped to a session)."""
        if session_id is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM bird_events WHERE event_type='enter' AND session_id=?",
                (session_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM bird_events WHERE event_type='enter'",
            ).fetchone()
        return row[0] if row else 0

    def visit_durations(self, session_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Pair each enter event with its next valid exit and compute duration.

        A tracker ID can appear in more than one visit during a session. Pairing
        every enter with every exit would create duplicate, negative, or inflated
        durations. Each arrival is therefore matched only to the first exit after
        it and before the next arrival for the same tracker and zone.
        """
        conditions = ""
        params: list = []
        if session_id is not None:
            conditions = "AND e.session_id=?"
            params = [session_id]
        rows = self._conn.execute(
            f"""WITH paired AS (
                    SELECT e.class_name AS species,
                           e.tracker_id,
                           e.occurred_at AS arrived_at,
                           (
                               SELECT MIN(x.occurred_at)
                               FROM bird_events x
                               WHERE x.session_id = e.session_id
                                 AND x.tracker_id = e.tracker_id
                                 AND x.zone_name = e.zone_name
                                 AND x.event_type = 'exit'
                                 AND x.occurred_at >= e.occurred_at
                                 AND x.occurred_at < COALESCE(
                                     (
                                         SELECT MIN(n.occurred_at)
                                         FROM bird_events n
                                         WHERE n.session_id = e.session_id
                                           AND n.tracker_id = e.tracker_id
                                           AND n.zone_name = e.zone_name
                                           AND n.event_type = 'enter'
                                           AND n.occurred_at > e.occurred_at
                                     ),
                                     '9999-12-31T23:59:59.999999+00:00'
                                 )
                           ) AS departed_at
                    FROM bird_events e
                    WHERE e.event_type = 'enter'
                    {conditions}
                )
                SELECT species,
                       tracker_id,
                       arrived_at,
                       departed_at,
                       ROUND(
                           (julianday(departed_at) - julianday(arrived_at)) * 86400
                       ) AS duration_seconds
                FROM paired
                WHERE departed_at IS NOT NULL
                ORDER BY arrived_at DESC""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

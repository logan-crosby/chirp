"""Live and historical metrics for chirp sessions.

SessionMetrics accumulates in-memory stats during a live run.
The module-level functions query the ChirpDatabase for historical analysis.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from chirp.database import ChirpDatabase


# ---------------------------------------------------------------------------
# Live session metrics (in-memory, reset each run)
# ---------------------------------------------------------------------------

@dataclass
class SessionMetrics:
    """Accumulates arrival/departure counts during a single camera session."""

    total_visits: int = 0
    species_counts: Counter = field(default_factory=Counter)
    zone_counts: Counter = field(default_factory=Counter)

    # Per-tracker state for computing visit duration
    _active_arrivals: Dict[int, datetime] = field(default_factory=dict)
    visit_durations_s: List[float] = field(default_factory=list)

    def record_arrival(self, tracker_id: int, species: Optional[str],
                       zone: str, timestamp: datetime) -> None:
        self.total_visits += 1
        self.species_counts[species or "unknown"] += 1
        self.zone_counts[zone] += 1
        self._active_arrivals[tracker_id] = timestamp

    def record_departure(self, tracker_id: int, timestamp: datetime) -> None:
        if tracker_id in self._active_arrivals:
            elapsed = (timestamp - self._active_arrivals.pop(tracker_id)).total_seconds()
            self.visit_durations_s.append(elapsed)

    def avg_visit_duration_s(self) -> Optional[float]:
        if not self.visit_durations_s:
            return None
        return sum(self.visit_durations_s) / len(self.visit_durations_s)

    def species_table_lines(self) -> List[str]:
        """Return a list of formatted strings for the live overlay."""
        if not self.species_counts:
            return []
        lines = []
        for species, count in self.species_counts.most_common():
            lines.append(f"  {species}: {count}")
        return lines

    def summary_text(self) -> str:
        lines = [f"Total visits this session: {self.total_visits}"]
        for species, count in self.species_counts.most_common():
            lines.append(f"  {species}: {count}")
        avg = self.avg_visit_duration_s()
        if avg is not None:
            lines.append(f"Avg visit: {avg:.0f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Historical analytics (database queries)
# ---------------------------------------------------------------------------

def print_species_summary(db: ChirpDatabase,
                           session_id: Optional[int] = None,
                           days: Optional[int] = None) -> None:
    """Print a formatted species visit count table to stdout."""
    rows = db.species_summary(session_id=session_id, days=days)
    label = f"last {days} days" if days else ("this session" if session_id else "all time")
    print(f"\n{'='*40}")
    print(f"  Species Summary ({label})")
    print(f"{'='*40}")
    if not rows:
        print("  No data.")
    else:
        max_name = max(len(r["species"]) for r in rows)
        for r in rows:
            bar = "█" * min(r["visits"], 30)
            print(f"  {r['species']:<{max_name}}  {r['visits']:>4}  {bar}")
    print()


def print_hourly_activity(db: ChirpDatabase, days: int = 7) -> None:
    """Print an ASCII bar chart of visits by hour of day."""
    rows = db.hourly_activity(days=days)
    if not rows:
        print("No hourly data available.")
        return
    max_v = max(r["visits"] for r in rows)
    print(f"\n{'='*40}")
    print(f"  Hourly Activity (last {days} days)")
    print(f"{'='*40}")
    for r in rows:
        bar_len = int(r["visits"] / max_v * 30) if max_v else 0
        bar = "█" * bar_len
        print(f"  {r['hour']}:00  {r['visits']:>4}  {bar}")
    print()


def print_recent_visitors(db: ChirpDatabase, limit: int = 10) -> None:
    """Print the most recent bird arrivals."""
    rows = db.recent_visitors(limit=limit)
    print(f"\n{'='*40}")
    print(f"  Recent Visitors")
    print(f"{'='*40}")
    if not rows:
        print("  No visitors recorded yet.")
    else:
        for r in rows:
            ts = r["occurred_at"][:19].replace("T", " ")
            print(f"  {ts}  {r['class_name'] or 'unknown':<24}  zone:{r['zone_name']}")
    print()


def print_visit_durations(db: ChirpDatabase,
                           session_id: Optional[int] = None) -> None:
    """Print a summary of how long birds stay."""
    rows = db.visit_durations(session_id=session_id)
    if not rows:
        return
    durations = [r["duration_seconds"] for r in rows if r["duration_seconds"] is not None]
    if not durations:
        return
    avg = sum(durations) / len(durations)
    print(f"\n{'='*40}")
    print(f"  Visit Duration Summary")
    print(f"{'='*40}")
    print(f"  Visits with paired exit: {len(durations)}")
    print(f"  Average duration:        {avg:.0f}s")
    print(f"  Shortest:                {min(durations):.0f}s")
    print(f"  Longest:                 {max(durations):.0f}s")
    print()

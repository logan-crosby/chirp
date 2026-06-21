"""Time-window scheduler for chirp.

Allows restricting bird counting to certain hours (e.g. daylight only) so
the process doesn't waste resources or generate empty sessions overnight.
"""
from __future__ import annotations

import time
import logging
from datetime import datetime, time as dtime
from typing import Optional

logger = logging.getLogger(__name__)


def is_within_window(start_time: Optional[str],
                     stop_time: Optional[str]) -> bool:
    """Return True if the current local time is within [start_time, stop_time].

    Args:
        start_time: "HH:MM" 24-hour string, or None to mean "always running".
        stop_time:  "HH:MM" 24-hour string, or None to mean "always running".

    Overnight schedules are supported: if start > stop (e.g. "22:00" to
    "06:00"), the window wraps around midnight.
    """
    if start_time is None or stop_time is None:
        return True

    now = datetime.now().time().replace(second=0, microsecond=0)
    start = dtime.fromisoformat(start_time)
    stop = dtime.fromisoformat(stop_time)

    if start <= stop:
        return start <= now <= stop
    else:
        # Overnight wrap (e.g. 22:00–06:00)
        return now >= start or now <= stop


def wait_for_window(start_time: Optional[str],
                    stop_time: Optional[str],
                    poll_interval_s: int = 60) -> None:
    """Block until the current time falls within [start_time, stop_time].

    Logs a message at INFO level once per minute while waiting. Does nothing
    if no schedule is configured.
    """
    if is_within_window(start_time, stop_time):
        return

    logger.info(
        "Outside scheduled window (%s – %s). Waiting...",
        start_time, stop_time,
    )
    while not is_within_window(start_time, stop_time):
        time.sleep(poll_interval_s)

    logger.info("Schedule window opened. Starting session.")

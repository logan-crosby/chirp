"""Contains ZoneMonitor implementation. Tracks when detections enter and exit a
zone and identifies species via majority vote over the full track lifetime.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import supervision as sv


class ZoneMonitor:
    """Tracks detections entering and exiting a zone.

    Species is determined by majority vote across all frames a track was
    visible, rather than the class predicted on any single frame.
    """

    def __init__(self,
                 in_threshold: Optional[int] = 15,
                 out_timeout: Optional[int] = 30) -> None:
        """
        Args:
            in_threshold: Frames a detection must be continuously tracked before
                it is counted as having entered the zone.
            out_timeout: Frames a confirmed detection can be absent before it is
                counted as having exited. Should match the tracker's lost_track_buffer.
        """
        self._in_threshold = in_threshold
        self._out_timeout = out_timeout
        self._monitored: Dict[int, dict] = {}

    def update(self,
               detections_in_zone: sv.Detections,
               frame_index: int,
               frame_datetime: datetime) -> Tuple[List[dict], List[dict]]:
        """Update zone state for one frame.

        Args:
            detections_in_zone: Tracked detections currently inside the zone.
            frame_index: Current frame number.
            frame_datetime: Timestamp for the current frame.

        Returns:
            (entered_events, exited_events) — lists of event dicts for tracks
            that crossed the in_threshold this frame and tracks that timed out.
        """
        entered_events: List[dict] = []
        exited_events: List[dict] = []

        active_ids = (
            set(int(t) for t in detections_in_zone.tracker_id if t is not None)
            if detections_in_zone.tracker_id is not None
            else set()
        )

        # Process detections present this frame.
        for detection in list(detections_in_zone):
            raw_tracker_id = detection[4]
            raw_class_id = detection[3]
            if raw_tracker_id is None:
                continue
            tracker_id = int(raw_tracker_id)
            class_id = int(raw_class_id) if raw_class_id is not None else None

            if tracker_id in self._monitored:
                entry = self._monitored[tracker_id]
                if class_id is not None:
                    entry["class_id_counts"][class_id] = (
                        entry["class_id_counts"].get(class_id, 0) + 1
                    )
                entry["out_timeout_counter"] = self._out_timeout
                entry["last_detection"] = detection
                if entry["frames_present"] < self._in_threshold:
                    entry["frames_present"] += 1
                    if entry["frames_present"] == self._in_threshold:
                        entered_events.append(self._build_event(tracker_id))
            else:
                self._monitored[tracker_id] = {
                    "tracker_id": tracker_id,
                    "last_detection": detection,
                    "frames_present": 1,
                    "out_timeout_counter": self._out_timeout,
                    "frame_entered": frame_index,
                    "datetime_entered": frame_datetime,
                    "frame_exited": -1,
                    "datetime_exited": -1,
                    "class_id_counts": {class_id: 1} if class_id is not None else {},
                }

        # Decrement timeout for detections absent this frame.
        for tracker_id in list(self._monitored.keys()):
            if tracker_id not in active_ids:
                entry = self._monitored[tracker_id]
                if entry["out_timeout_counter"] == self._out_timeout:
                    entry["frame_exited"] = frame_index
                    entry["datetime_exited"] = frame_datetime
                entry["out_timeout_counter"] -= 1
                if entry["out_timeout_counter"] <= 0:
                    if entry["frames_present"] >= self._in_threshold:
                        exited_events.append(self._build_event(tracker_id))
                    del self._monitored[tracker_id]

        return entered_events, exited_events

    def _build_event(self, tracker_id: int) -> dict:
        """Build an event dict for a tracker, using majority-vote species ID."""
        entry = self._monitored[tracker_id]
        counts = entry["class_id_counts"]
        species_class_id = max(counts, key=counts.get) if counts else None
        return {
            "tracker_id": tracker_id,
            "species_class_id": species_class_id,
            "last_detection": entry["last_detection"],
            "frame_entered": entry["frame_entered"],
            "datetime_entered": entry["datetime_entered"],
            "frame_exited": entry["frame_exited"],
            "datetime_exited": entry["datetime_exited"],
            "class_id_counts": dict(counts),
        }

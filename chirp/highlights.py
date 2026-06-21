"""Video highlight clip recorder.

HighlightRecorder keeps a rolling pre-roll buffer of recent frames. When a
bird arrival is confirmed, it writes the pre-roll + ongoing frames to a .mp4
file. Recording stops (with a short post-roll tail) when the bird departs.

Multiple birds can have overlapping clips: each tracker gets its own writer.
"""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Dict, Deque

import cv2 as cv
import numpy as np

logger = logging.getLogger(__name__)


class HighlightRecorder:
    """Saves per-visit video clips with configurable pre- and post-roll."""

    def __init__(self,
                 output_dir: Path,
                 fps: float,
                 frame_size: tuple[int, int],  # (width, height)
                 pre_roll_s: int = 5,
                 post_roll_s: int = 3) -> None:
        self._output_dir = Path(output_dir)
        self._fps = fps
        self._frame_size = frame_size
        self._pre_roll_frames = int(fps * pre_roll_s)
        self._post_roll_frames = int(fps * post_roll_s)

        # Rolling buffer used as pre-roll for the next clip
        self._buffer: Deque[np.ndarray] = deque(maxlen=self._pre_roll_frames)

        # Active recordings: tracker_id -> recorder state
        self._active: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def feed_frame(self, frame: np.ndarray) -> None:
        """Call once per frame, every frame, regardless of bird presence."""
        # Add to pre-roll buffer (used when the next bird arrives)
        self._buffer.append(frame)

        # Write to any active recordings and manage post-roll countdown
        for tracker_id in list(self._active.keys()):
            rec = self._active[tracker_id]
            rec["writer"].write(frame)
            if rec["finishing"]:
                rec["post_roll_remaining"] -= 1
                if rec["post_roll_remaining"] <= 0:
                    self._close_recording(tracker_id)

    def start_clip(self, tracker_id: int, label: str = "") -> Path | None:
        """Open a new clip for *tracker_id*, prepending the pre-roll buffer.

        Args:
            tracker_id: Unique tracker ID for this bird.
            label: Optional species label embedded in the filename.

        Returns:
            Path to the clip file, or None if a recording is already active.
        """
        if tracker_id in self._active:
            return None

        safe_label = label.replace(" ", "_").replace("/", "-") if label else "bird"
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = self._output_dir / f"highlight_{ts}_{safe_label}_t{tracker_id}.mp4"

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(
            str(filename), fourcc, self._fps, self._frame_size
        )
        if not writer.isOpened():
            logger.warning("Could not open VideoWriter for %s", filename)
            return None

        # Flush the pre-roll buffer into the clip
        for buffered_frame in self._buffer:
            writer.write(buffered_frame)

        self._active[tracker_id] = {
            "writer": writer,
            "filename": filename,
            "finishing": False,
            "post_roll_remaining": 0,
        }
        logger.info("Highlight recording started: %s", filename.name)
        return filename

    def end_clip(self, tracker_id: int) -> None:
        """Schedule the clip for *tracker_id* to close after post-roll frames."""
        if tracker_id not in self._active:
            return
        rec = self._active[tracker_id]
        if not rec["finishing"]:
            rec["finishing"] = True
            rec["post_roll_remaining"] = self._post_roll_frames

    def end_all_clips(self) -> None:
        """Immediately close all open recordings (call on shutdown)."""
        for tracker_id in list(self._active.keys()):
            self._close_recording(tracker_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _close_recording(self, tracker_id: int) -> None:
        rec = self._active.pop(tracker_id, None)
        if rec:
            rec["writer"].release()
            logger.info("Highlight saved: %s", rec["filename"].name)

"""Contains ZoneMonitor implementation. This class is meant to serve as a helper
class to be used alongside a Supervision PolygonZone for counting the number of
objects that enter and exit a zone.
"""

from datetime import datetime
from typing import List, Optional, Tuple

import supervision as sv

class ZoneMonitor:
    """Class to keep track of the detections that enter and exit a zone."""

    def __init__(self,
                 in_threshold: Optional[int] = 15,
                 out_timeout: Optional[int] = 30) -> None:
        """Creates a new ZoneMonitor instance to track when detections enter and
        leave a zone.

        Args:
            in_threshold (Optional[int], optional): Number of frames that a
            detection must be tracked for before it is considered "in" the zone.
            Defaults to 15.
            out_timeout (Optional[int], optional): Number of frames that a
            detection considered to be "in" the zone can be absent for before it
            is deemed as "exited." This should probably match the tracker's
            "track_thresh" so that objects picked back up by tracker aren't
            double counted by this monitor.
        """
        # Grab constructor parameters.
        self._in_threshold = in_threshold
        self._out_timeout = out_timeout
        # Create dictionary for monitored detections. Maps a detection's
        # tracker_id to the number of times it has been seen and its timeout.
        # TODO: There should be an additional class made specifically just for
        # handling this dictionary and its entries.
        self._monitored_detections = {}

    def update(self, 
               detections_in_zone: sv.Detections,
               frame_index: int,
               frame_datetime: datetime) -> Tuple[List, List]:
        """Takes the detections from a zone and figures out which tracked
        detections have newly entered the zone and those that have exited the
        zone since the last call to update.

        Args:
            detections_in_zone (sv.Detections): Supervision Detections instance
            containing the detections from a zone.
            frame_index (int): The offset / number of frames since the
            beginning of the stream. I.e., the frame index / number.
            frame_datetime (datetime): The datetime the image the detections are
            from was captured. I.e., the output of datetime.now() upstream.

        Returns:
            Tuple[List, List]: Returns a list of detection vectors that have
            recently entered the zone (been present for at least in_threshold
            frames) and the list of detection vectors that haven't been seen in
            out_timeout frames.
        """

        # Create list for detections that have been present in the video
        # sequence for at least in_threshold frames.
        entered_detections = []
        # Create list for detections that have not been present for at least
        # out_timeout frames.
        exited_detections = []
        
        # Add any new detections to the monitored list. Increment the number of
        # frames present for detections already in the monitored list.
        for detection in list(detections_in_zone):
            # Accessing elements of each detection according to
            # https://supervision.roboflow.com/detection/core/#detections
            tracker_id = detection[4]

            # If this tracked detection is already being monitored, then just
            # increments its frames_since_added count. Also refresh its out
            # timeout counter.
            if tracker_id in list(self._monitored_detections.keys()):
                if self._monitored_detections[tracker_id]["frames_present"] < self._in_threshold:
                    self._monitored_detections[tracker_id]["frames_present"] += 1
                    if self._monitored_detections[tracker_id]["frames_present"] == self._in_threshold:
                        entered_detections.append(self._monitored_detections[tracker_id])
                self._monitored_detections[tracker_id]["last_detection"] = detection
                self._monitored_detections[tracker_id]["out_timeout_counter"] = self._out_timeout

            # If not, this is a new detection. Add a new entry to the monitored
            # detections dictionary.
            else:
                self._monitored_detections[tracker_id] = {
                    "last_detection": detection,
                    "frames_present": 1,
                    "out_timeout_counter": self._out_timeout,
                    "frame_entered": frame_index,
                    "datetime_entered": frame_datetime,
                    "frame_exited": -1, # Indicates detection hasn't exited.
                    "datetime_exited": -1
                }

        # Remove any detections that haven't been present for out_timeout
        # frames. Decrement out_timeout_counter for all other missing
        # detections. Also, if it's the first frame that a detection isn't
        # present, set this frame as the "exited" frame.
        for tracker_id in list(self._monitored_detections.keys()):
            if tracker_id not in detections_in_zone.tracker_id:
                if self._monitored_detections[tracker_id]["out_timeout_counter"] == self._out_timeout:
                    self._monitored_detections[tracker_id]["frame_exited"] = frame_index
                    self._monitored_detections[tracker_id]["datetime_exited"] = frame_datetime
                self._monitored_detections[tracker_id]["out_timeout_counter"] -= 1
                if self._monitored_detections[tracker_id]["out_timeout_counter"] == 0:
                    if self._monitored_detections[tracker_id]["frames_present"] == self._in_threshold:
                        exited_detections.append(self._monitored_detections[tracker_id])
                    del(self._monitored_detections[tracker_id])
        
        return entered_detections, exited_detections
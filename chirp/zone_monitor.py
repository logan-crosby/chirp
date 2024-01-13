
from typing import List, Optional, Tuple

import supervision as sv


class ZoneMonitor:

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
        self._monitored_detections = {}

    def update(detections_in_zone: sv.Detections) -> Tuple[sv.Detections, sv.Detections]:
        
        # Rough idea:
        # For each of the detections, check if it's TRACKER_ID is already in the
        # tracked_detections dictionary.

        # If it is, increment its frames_since_added counter. Maybe update with
        # the ceil of the threshold so the numbers don't overflow.
        
        # If it's not already in there, create a new entry in the dictionary for
        # it. The entry should map the tracker id to a new dictionary, whose
        # values will be the detection vector, the frames_since_added_counter,
        # and the out_timeout_counter.

        
        # Also, have to check for existing detections, if any of their
        # out_timeout_counter values have reached 0, if their frames_since_added
        # == in_threshold, then that entry can be removed from the dictionary
        # and returned as an EXITED detection.

        # If it's frames_since_added is not == in_threshold, then that detection
        # wasn't around long enough to be considered present, and therefore
        # shouldn't be returned as exited or counted.

        # Also, have to check for existing detections, if their
        # frames_since_added reaches the in_threshold, then those instances can
        # be returned in the list of "entered detections" from this call to
        # update.
        

        pass
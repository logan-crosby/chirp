
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
        # Create dictionary for tracked detections. Maps a detection's
        # tracker_id to the number of times it has been seen and its timeout.
        tracked_detections = {}

    def update(detections_in_zone: sv.Detections) -> Tuple[sv.Detections, sv.Detections]:
        
        

        pass
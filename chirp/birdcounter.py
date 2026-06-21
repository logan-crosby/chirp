"""Core bird detection, tracking, and counting pipeline.

Usage
-----
    # Via installed CLI entry-point:
    chirp --config chirp_config.yaml

    # As a module:
    python -m chirp --config chirp_config.yaml

    # Direct (all options as CLI flags):
    python chirp/birdcounter.py -v /dev/video0 -l 40.71,-74.00
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
import supervision as sv
from ultralytics import YOLO

from chirp.config import ChirpConfig, ZoneConfig, load_config
from chirp.database import ChirpDatabase
from chirp.highlights import HighlightRecorder
from chirp.metrics import SessionMetrics
from chirp.scheduler import wait_for_window
from chirp.zone_monitor import ZoneMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# COCO class ID for the generic "bird" category.
COCO_BIRD_CLASS_ID = 14


# ---------------------------------------------------------------------------
# BirdCounter
# ---------------------------------------------------------------------------

class BirdCounter:
    """End-to-end pipeline: open camera → detect → track → count → log."""

    def __init__(self, config: ChirpConfig) -> None:
        self.config = config
        self._model: Optional[YOLO] = None
        self._tracker: Optional[sv.ByteTrack] = None
        self._class_map: dict[int, str] = {}
        self._bird_class_ids: Optional[set[int]] = None

        # Zone state: list of (zone_name, sv.PolygonZone, ZoneMonitor)
        self._zones: List[Tuple[str, sv.PolygonZone, ZoneMonitor]] = []

        self._db: Optional[ChirpDatabase] = None
        self._session_id: Optional[int] = None
        self._metrics = SessionMetrics()
        self._highlight_recorder: Optional[HighlightRecorder] = None

        # Annotators (set up after camera is opened)
        self._circle_ann = sv.CircleAnnotator()
        self._label_ann = sv.LabelAnnotator()
        self._trace_ann = sv.TraceAnnotator()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the camera, block until done, then clean up."""
        wait_for_window(
            self.config.schedule.start_time,
            self.config.schedule.stop_time,
        )

        cfg = self.config
        cfg.output_directory.mkdir(parents=True, exist_ok=True)

        self._db = ChirpDatabase(cfg.effective_database())

        logger.info("Loading model: %s", cfg.model)
        self._model = YOLO(model=cfg.model)
        self._class_map = {
            cid: str(name) for cid, name in self._model.names.items()
        }
        self._resolve_class_filter()

        camera = cv.VideoCapture(str(cfg.video_source))
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open video source: {cfg.video_source}")

        fps = camera.get(cv.CAP_PROP_FPS) or 30.0
        height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
        logger.info("Video: %dx%d @ %.1f fps", width, height, fps)

        in_threshold = cfg.in_threshold if cfg.in_threshold else max(1, int(fps) // 2)
        out_timeout = int(fps) * cfg.track_buffer

        self._tracker = sv.ByteTrack(
            frame_rate=int(fps),
            lost_track_buffer=out_timeout,
        )

        self._setup_zones(cfg, width, height, fps, in_threshold, out_timeout)

        # Highlight recorder
        if cfg.highlights.enabled:
            highlights_dir = cfg.output_directory / "chirp_sessions" / "highlights"
            highlights_dir.mkdir(parents=True, exist_ok=True)
            self._highlight_recorder = HighlightRecorder(
                output_dir=highlights_dir,
                fps=fps,
                frame_size=(width, height),
                pre_roll_s=cfg.highlights.pre_roll,
                post_roll_s=cfg.highlights.post_roll,
            )

        self._session_id = self._db.create_session(
            started_at=datetime.now(tz=timezone.utc),
            location=cfg.location,
            model_source=cfg.model,
            confidence_threshold=cfg.confidence,
            iou_threshold=cfg.iou,
            in_threshold_frames=in_threshold,
            track_buffer_seconds=cfg.track_buffer,
        )
        for zone_name, zone, _ in self._zones:
            self._db.log_zone_def(
                self._session_id, zone_name,
                zone.polygon.tolist(),
            )

        try:
            self._main_loop(camera, fps, width, height)
        finally:
            camera.release()
            cv.destroyAllWindows()
            if self._highlight_recorder:
                self._highlight_recorder.end_all_clips()
            self._print_session_summary()
            if self._db:
                self._db.close()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_class_filter(self) -> None:
        """Set self._bird_class_ids based on config and model labels."""
        cfg = self.config
        if cfg.classes is not None:
            # Explicit list (empty list = track everything)
            self._bird_class_ids = set(cfg.classes) if cfg.classes else None
            return

        # Auto-detect COCO model
        if self._class_map.get(COCO_BIRD_CLASS_ID) == "bird":
            logger.info(
                "COCO model detected – auto-filtering to class %d (bird).",
                COCO_BIRD_CLASS_ID,
            )
            self._bird_class_ids = {COCO_BIRD_CLASS_ID}
        else:
            logger.info(
                "Non-COCO model with %d classes. Tracking all classes. "
                "Pass classes: [<id>,...] in config to restrict.",
                len(self._class_map),
            )
            self._bird_class_ids = None

    def _is_nms_free(self) -> bool:
        """Return True when the loaded model uses end-to-end NMS-free inference.

        YOLO26 has a one-to-one detection head that eliminates duplicate boxes
        natively, so running with_nms() post-inference is unnecessary.
        """
        name = str(self.config.model).lower()
        return "yolo26" in name

    def _setup_zones(self, cfg: ChirpConfig, width: int, height: int,
                     fps: float, in_threshold: int, out_timeout: int) -> None:
        """Build sv.PolygonZone + ZoneMonitor for each configured zone."""
        zone_configs: List[ZoneConfig] = cfg.zones
        if not zone_configs:
            # Default: full camera frame
            zone_configs = [ZoneConfig(
                name="full_frame",
                polygon=[[0, 0], [width, 0], [width, height], [0, height]],
            )]

        for zc in zone_configs:
            poly = np.array(zc.polygon, dtype=np.int64)
            zone = sv.PolygonZone(
                polygon=poly,
                triggering_anchors=[sv.Position.CENTER],
            )
            monitor = ZoneMonitor(
                in_threshold=in_threshold,
                out_timeout=out_timeout,
            )
            self._zones.append((zc.name, zone, monitor))
            logger.info("Zone '%s' configured with %d vertices.", zc.name, len(poly))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _reconnect(self, camera: cv.VideoCapture,
                   max_retries: int = 8) -> bool:
        """Try to reopen a dropped camera device with exponential backoff."""
        src = str(self.config.video_source)
        for attempt in range(1, max_retries + 1):
            delay = min(2 ** attempt, 60)
            logger.warning(
                "Camera read failed. Reconnect attempt %d/%d (wait %ds)…",
                attempt, max_retries, delay,
            )
            time.sleep(delay)
            camera.release()
            camera.open(src)
            if camera.isOpened():
                logger.info("Camera reconnected.")
                return True
        logger.error("Could not reconnect after %d attempts. Stopping.", max_retries)
        return False

    def _main_loop(self, camera: cv.VideoCapture,
                   fps: float, width: int, height: int) -> None:
        frame_counter = 0
        is_device = str(self.config.video_source).startswith("/dev/video")

        while True:
            # Enforce schedule (check once per second at most)
            if frame_counter % max(1, int(fps)) == 0:
                if not _schedule_active(self.config):
                    logger.info("Reached schedule end time. Stopping.")
                    break

            status, frame = camera.read()
            if not status:
                if is_device:
                    # Camera device — attempt to reconnect before giving up
                    if not self._reconnect(camera):
                        break
                    continue
                else:
                    logger.info("End of video file.")
                    break

            frame_counter += 1
            frame_datetime = datetime.now(tz=timezone.utc)

            detections = self._detect(frame)

            # Feed highlight recorder before annotation
            if self._highlight_recorder:
                self._highlight_recorder.feed_frame(frame)

            # Update each zone
            for zone_name, zone, monitor in self._zones:
                mask = zone.trigger(detections=detections)
                in_zone = detections[mask]
                entered, exited = monitor.update(in_zone, frame_counter, frame_datetime)
                self._handle_events(entered, exited, zone_name, frame_datetime)

            # Display
            if self.config.show_display:
                annotated = self._annotate(frame, detections)
                preview = cv.resize(annotated, (width // 2, height // 2))
                cv.imshow("Chirp – Bird Counter", preview)
                if cv.waitKey(1) == ord("q"):
                    logger.info("User quit.")
                    break

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> sv.Detections:
        """Run inference, filter, NMS, and track; return updated Detections.

        Class IDs are passed directly to the model so the inference kernel
        skips non-bird categories entirely — faster than post-filtering.
        YOLO26 outputs are NMS-free by design; we skip the NMS step for it.
        """
        infer_classes = (
            list(self._bird_class_ids)
            if self._bird_class_ids is not None
            else None
        )
        result = self._model(
            frame,
            verbose=False,
            conf=self.config.confidence,
            classes=infer_classes,
        )[0]
        detections = sv.Detections.from_ultralytics(ultralytics_results=result)

        # NMS deduplication — skip for YOLO26 which is already NMS-free.
        if len(detections) > 0 and not self._is_nms_free():
            if detections.class_id is not None:
                detections = detections.with_nms(threshold=self.config.iou)
            else:
                detections = detections.with_nms(
                    threshold=self.config.iou, class_agnostic=True
                )

        # Track (ByteTrack is deprecated in sv 0.28+ but still functional in 0.29)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            detections = self._tracker.update_with_detections(detections=detections)

        return detections

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_events(self,
                       entered: list, exited: list,
                       zone_name: str,
                       frame_datetime: datetime) -> None:
        for event in entered:
            cid = event["species_class_id"]
            name = self._class_map.get(cid, "unknown") if cid is not None else "unknown"
            tid = event["tracker_id"]
            fseen = event.get("class_id_counts", {})
            frames_seen = sum(fseen.values()) if fseen else 0

            self._db.log_event(
                session_id=self._session_id,
                event_type="enter",
                occurred_at=event["datetime_entered"],
                frame_number=event["frame_entered"],
                zone_name=zone_name,
                tracker_id=tid,
                class_id=cid,
                class_name=name,
                frames_seen=frames_seen,
                class_id_counts={int(k): v for k, v in fseen.items()},
            )
            self._metrics.record_arrival(tid, name, zone_name, event["datetime_entered"])
            logger.info("ARRIVED  %-24s (track %s  zone:%s)", name, tid, zone_name)

            if self._highlight_recorder:
                self._highlight_recorder.start_clip(tracker_id=tid, label=name)

        for event in exited:
            cid = event["species_class_id"]
            name = self._class_map.get(cid, "unknown") if cid is not None else "unknown"
            tid = event["tracker_id"]
            fseen = event.get("class_id_counts", {})
            frames_seen = sum(fseen.values()) if fseen else 0

            self._db.log_event(
                session_id=self._session_id,
                event_type="exit",
                occurred_at=event["datetime_exited"],
                frame_number=event["frame_exited"],
                zone_name=zone_name,
                tracker_id=tid,
                class_id=cid,
                class_name=name,
                frames_seen=frames_seen,
                class_id_counts={int(k): v for k, v in fseen.items()},
            )
            self._metrics.record_departure(tid, event["datetime_exited"])
            logger.info("DEPARTED %-24s (track %s  zone:%s)", name, tid, zone_name)

            if self._highlight_recorder:
                self._highlight_recorder.end_clip(tracker_id=tid)

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate(self, frame: np.ndarray,
                  detections: sv.Detections) -> np.ndarray:
        labels = []
        cids = detections.class_id if detections.class_id is not None else []
        tids = detections.tracker_id if detections.tracker_id is not None else []
        for cid, tid in zip(cids, tids):
            name = self._class_map.get(int(cid), "?") if cid is not None else "?"
            labels.append(f"#{tid} {name}" if tid is not None else name)

        out = self._circle_ann.annotate(scene=frame.copy(), detections=detections)
        out = self._label_ann.annotate(scene=out, detections=detections, labels=labels)
        out = self._trace_ann.annotate(scene=out, detections=detections)

        # Live stats overlay
        overlay_lines = [f"Visits: {self._metrics.total_visits}"]
        overlay_lines += self._metrics.species_table_lines()
        y = 30
        for line in overlay_lines:
            out = sv.draw_text(
                scene=out,
                text=line,
                text_anchor=sv.Point(x=10, y=y),
                text_color=sv.Color.WHITE,
                background_color=sv.Color.BLACK,
                text_scale=0.6,
                text_thickness=1,
            )
            y += 22

        return out

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------

    def _print_session_summary(self) -> None:
        from chirp.metrics import (
            print_species_summary,
            print_visit_durations,
            print_recent_visitors,
        )
        print("\n" + "=" * 50)
        print("  SESSION COMPLETE")
        print("=" * 50)
        print(self._metrics.summary_text())
        if self._db and self._session_id:
            print_visit_durations(self._db, session_id=self._session_id)
            print_recent_visitors(self._db, limit=10)


# ---------------------------------------------------------------------------
# CLI / entrypoint
# ---------------------------------------------------------------------------

def _export_csv(cfg: ChirpConfig, output_path: Path) -> None:
    """Dump all bird_events from the database to a CSV file."""
    import csv
    db = ChirpDatabase(cfg.effective_database())
    conn = db.get_connection()
    rows = conn.execute(
        """SELECT s.started_at AS session_started, s.location,
                  e.event_type, e.occurred_at, e.zone_name,
                  e.class_name, e.tracker_id, e.frames_seen
           FROM bird_events e
           JOIN sessions s ON s.id = e.session_id
           ORDER BY e.occurred_at"""
    ).fetchall()
    db.close()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "session_started", "location", "event_type", "occurred_at",
            "zone_name", "class_name", "tracker_id", "frames_seen",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    logger.info("Exported %d events to %s", len(rows), output_path)


def _schedule_active(cfg: ChirpConfig) -> bool:
    from chirp.scheduler import is_within_window
    return is_within_window(cfg.schedule.start_time, cfg.schedule.stop_time)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="chirp – real-time bird feeder detection and counting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", metavar="FILE", default=None,
                   help="Path to YAML config file.")
    p.add_argument("-v", "--video-source", metavar="PATH",
                   help="Camera device or video file.")
    p.add_argument("-m", "--model", metavar="WEIGHTS",
                   help="Ultralytics model name or path to .pt file.")
    p.add_argument("-o", "--output-directory", metavar="DIR",
                   help="Root directory for session data and the SQLite DB.")
    p.add_argument("-l", "--location", metavar="LAT,LON",
                   help="Approximate feeder location (stored in metadata).")
    p.add_argument("--confidence", metavar="FLOAT", type=float,
                   help="Minimum detection confidence (0–1).")
    p.add_argument("--iou", metavar="FLOAT", type=float,
                   help="NMS IoU deduplication threshold.")
    p.add_argument("--classes", metavar="IDS",
                   help="Comma-separated class IDs to count, or 'all'.")
    p.add_argument("--in-threshold", metavar="N", type=int,
                   help="Frames before a detection is counted as an arrival.")
    p.add_argument("--track-buffer", metavar="S", type=int,
                   help="Seconds a lost track is held before dropping.")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless (no preview window).")
    p.add_argument("--no-highlights", action="store_true",
                   help="Disable highlight clip recording.")
    p.add_argument("--stats", action="store_true",
                   help="Print historical stats from the database and exit.")
    p.add_argument("--stats-days", metavar="N", type=int, default=7,
                   help="Days of history to include when --stats is used.")
    p.add_argument("--export", metavar="FILE",
                   help="Export all bird_events from the database to a CSV file and exit.")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    if config_path is None:
        # Look for chirp_config.yaml in the current directory
        default_cfg = Path("chirp_config.yaml")
        if default_cfg.exists():
            config_path = default_cfg

    cfg = load_config(args=args, config_file=config_path)

    # --export mode: dump DB to CSV and exit
    if getattr(args, "export", None):
        _export_csv(cfg, Path(args.export))
        return

    # --stats mode: print analytics and exit
    if args.stats:
        db = ChirpDatabase(cfg.effective_database())
        from chirp.metrics import (
            print_species_summary,
            print_hourly_activity,
            print_recent_visitors,
        )
        print_species_summary(db, days=args.stats_days)
        print_hourly_activity(db, days=args.stats_days)
        print_recent_visitors(db, limit=20)
        db.close()
        return

    if cfg.location is None:
        parser.error("--location LAT,LON is required (or set 'location' in config).")

    counter = BirdCounter(cfg)
    counter.run()


if __name__ == "__main__":
    main()

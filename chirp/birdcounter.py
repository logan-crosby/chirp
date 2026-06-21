"""Core object detection + tracking bird feeder analysis pipeline.

Reads from a camera device or video file and logs bird visit events (CSV +
JSON) to a session directory. Updated for supervision>=0.29 and
ultralytics>=8.4.
"""
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from csv import DictWriter
import json

import numpy as np
import cv2 as cv

from ultralytics import YOLO
import supervision as sv

from zone_monitor import ZoneMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# COCO class ID for the generic "bird" category.
COCO_BIRD_CLASS_ID = 14


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Chirp – real-time bird feeder detection and counting"
    )
    parser.add_argument(
        "-v", "--video-source",
        default="/dev/video0", type=str,
        help="Path to video device (/dev/videoX) or video file. Default: /dev/video0",
    )
    parser.add_argument(
        "-w", "--yolo-weights",
        default=None, type=str,
        help=(
            "Path to YOLO model weights file, or an Ultralytics model name that "
            "will be auto-downloaded (e.g. 'yolo11n.pt'). "
            "Defaults to 'yolo11n.pt' when omitted."
        ),
    )
    parser.add_argument(
        "-o", "--output-directory",
        default=".", type=str,
        help="Directory where session output folders will be created. Default: .",
    )
    parser.add_argument(
        "-l", "--location",
        default=None, type=str,
        help="Approximate feeder location as 'LAT,LON' (used for metadata).",
    )
    parser.add_argument(
        "--confidence",
        default=0.35, type=float,
        help="Minimum detection confidence to accept (0–1). Default: 0.35",
    )
    parser.add_argument(
        "--iou",
        default=0.5, type=float,
        help="NMS IoU overlap threshold for deduplicating detections. Default: 0.5",
    )
    parser.add_argument(
        "--classes",
        default=None, type=str,
        help=(
            "Comma-separated class IDs to count (e.g. '14' for COCO bird). "
            "When omitted with a COCO model, class 14 is used automatically. "
            "Pass 'all' to disable filtering entirely."
        ),
    )
    parser.add_argument(
        "--in-threshold",
        default=None, type=int,
        help=(
            "Frames a track must be seen before counting as an arrival. "
            "Defaults to half a second worth of frames."
        ),
    )
    parser.add_argument(
        "--track-buffer",
        default=8, type=int,
        help="Seconds a track can be absent before being dropped. Default: 8",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless (no preview window). Useful for unattended yard use.",
    )
    args = parser.parse_args()

    # --- Validate inputs ---
    video_source_path = Path(args.video_source)
    if not video_source_path.exists():
        raise FileNotFoundError(f"Video source not found: {video_source_path}")

    model_source = args.yolo_weights if args.yolo_weights else "yolo11n.pt"

    output_directory_path = Path(args.output_directory)
    if not output_directory_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_directory_path}")

    if args.location is None:
        raise ValueError("--location LAT,LON is required.")

    confidence_threshold: float = args.confidence
    iou_threshold: float = args.iou

    # Parse class filter.
    bird_class_ids: set | None
    if args.classes and args.classes.lower() != "all":
        bird_class_ids = {int(c.strip()) for c in args.classes.split(",")}
    else:
        bird_class_ids = None  # Resolved after model loads when using COCO weights.

    # --- Create session output directory ---
    sessions_dir = output_directory_path / "chirp_sessions"
    sessions_dir.mkdir(exist_ok=True)

    session_datetime = datetime.now(tz=timezone.utc)
    safe_dt = session_datetime.strftime("%Y%m%dT%H%M%SZ")
    session_dir = sessions_dir / f"chirp_session_{safe_dt}"
    session_dir.mkdir()
    logger.info("Session directory: %s", session_dir)

    # --- Open session CSV ---
    events_csv_path = session_dir / f"{safe_dt}_session_events.csv"
    events_file = open(file=events_csv_path, mode="w", newline="")
    fieldnames = [
        "datetime", "frame", "zone_id",
        "class_id", "class_name", "tracker_id", "event_type_id",
    ]
    csv_writer = DictWriter(events_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    EVENT_TYPE_MAPPINGS = {0: "enter", 1: "exit"}
    ZONE_MAPPINGS = {0: "full_zone"}

    # --- Open camera ---
    feeder_camera = cv.VideoCapture(str(video_source_path))
    if not feeder_camera.isOpened():
        raise RuntimeError("Failed to open video source.")
    video_fps = feeder_camera.get(cv.CAP_PROP_FPS) or 30.0
    video_height = int(feeder_camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_width = int(feeder_camera.get(cv.CAP_PROP_FRAME_WIDTH))
    logger.info("Video: %dx%d @ %.1f fps", video_width, video_height, video_fps)

    # --- Define counting zone (full frame) ---
    # frame_resolution_wh was removed in supervision 0.24; polygon bounds are sufficient.
    full_zone_polygon = np.array([
        [0, 0],
        [video_width, 0],
        [video_width, video_height],
        [0, video_height],
    ])
    # triggering_position was renamed to triggering_anchors (list) in supervision 0.20.
    full_zone = sv.PolygonZone(
        polygon=full_zone_polygon,
        triggering_anchors=[sv.Position.CENTER],
    )

    track_buffer_s: int = args.track_buffer
    in_threshold = args.in_threshold if args.in_threshold else max(1, int(video_fps) // 2)
    out_timeout = int(video_fps) * track_buffer_s

    full_zone_monitor = ZoneMonitor(
        in_threshold=in_threshold,
        out_timeout=out_timeout,
    )
    bird_visit_count = 0

    # --- Load model ---
    logger.info("Loading model: %s", model_source)
    model = YOLO(model=model_source)
    CLASS_MAPPINGS: dict[int, str] = {
        class_id: str(model.names[class_id]) for class_id in range(len(model.names))
    }

    # Auto-filter to COCO bird class when no explicit filter is set and the model
    # uses COCO labels (class 14 == "bird").
    if bird_class_ids is None and args.classes is None:
        if CLASS_MAPPINGS.get(COCO_BIRD_CLASS_ID) == "bird":
            logger.info(
                "COCO model detected – auto-filtering detections to class %d (bird).",
                COCO_BIRD_CLASS_ID,
            )
            bird_class_ids = {COCO_BIRD_CLASS_ID}
        else:
            logger.info(
                "Non-COCO model with %d classes loaded. Tracking all classes. "
                "Pass --classes to restrict to specific IDs.",
                len(CLASS_MAPPINGS),
            )

    # --- Tracker ---
    # 'track_buffer' was renamed to 'lost_track_buffer' in supervision 0.23.
    tracker = sv.ByteTrack(
        frame_rate=int(video_fps),
        lost_track_buffer=out_timeout,
    )

    # --- Annotators ---
    circle_annotator = sv.CircleAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    # --- Write session metadata ---
    with open(session_dir / "session_metadata.json", "w") as f:
        json.dump(
            {
                "session_datetime": str(session_datetime),
                "session_location": args.location,
                "model_source": model_source,
                "tracked_class_ids": sorted(bird_class_ids) if bird_class_ids else "all",
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "in_threshold_frames": in_threshold,
                "track_buffer_seconds": track_buffer_s,
                "event_type_mappings": EVENT_TYPE_MAPPINGS,
                "zone_mappings": ZONE_MAPPINGS,
                "class_mappings": {str(k): v for k, v in CLASS_MAPPINGS.items()},
            },
            fp=f,
            indent=2,
        )

    frame_counter = 0

    try:
        while True:
            status, frame = feeder_camera.read()
            if not status:
                logger.info("End of video stream.")
                break

            frame_counter += 1
            frame_datetime = datetime.now(tz=timezone.utc)

            # --- Inference ---
            result = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(ultralytics_results=result)

            # --- Filter by confidence ---
            if detections.confidence is not None and len(detections) > 0:
                detections = detections[detections.confidence >= confidence_threshold]

            # --- NMS: remove overlapping boxes for the same object ---
            if len(detections) > 0 and detections.class_id is not None:
                detections = detections.with_nms(threshold=iou_threshold)
            elif len(detections) > 0:
                detections = detections.with_nms(threshold=iou_threshold, class_agnostic=True)

            # --- Filter to target bird classes ---
            if (
                bird_class_ids is not None
                and detections.class_id is not None
                and len(detections) > 0
            ):
                class_mask = np.isin(detections.class_id, list(bird_class_ids))
                detections = detections[class_mask]

            # --- Track ---
            detections = tracker.update_with_detections(detections=detections)

            # --- Zone counting ---
            zone_mask = full_zone.trigger(detections=detections)
            detections_in_zone = detections[zone_mask]
            entered_events, exited_events = full_zone_monitor.update(
                detections_in_zone=detections_in_zone,
                frame_index=frame_counter,
                frame_datetime=frame_datetime,
            )
            bird_visit_count += len(entered_events)

            for event in entered_events:
                class_id = event["species_class_id"]
                class_name = CLASS_MAPPINGS.get(class_id, str(class_id)) if class_id is not None else "unknown"
                csv_writer.writerow({
                    "datetime": event["datetime_entered"],
                    "frame": event["frame_entered"],
                    "zone_id": 0,
                    "class_id": class_id,
                    "class_name": class_name,
                    "tracker_id": event["tracker_id"],
                    "event_type_id": 0,
                })
                logger.info("ARRIVED  %-20s (track %s)", class_name, event["tracker_id"])

            for event in exited_events:
                class_id = event["species_class_id"]
                class_name = CLASS_MAPPINGS.get(class_id, str(class_id)) if class_id is not None else "unknown"
                csv_writer.writerow({
                    "datetime": event["datetime_exited"],
                    "frame": event["frame_exited"],
                    "zone_id": 0,
                    "class_id": class_id,
                    "class_name": class_name,
                    "tracker_id": event["tracker_id"],
                    "event_type_id": 1,
                })
                logger.info("DEPARTED %-20s (track %s)", class_name, event["tracker_id"])

            # Flush CSV periodically so data isn't lost if the process is killed.
            if frame_counter % 300 == 0:
                events_file.flush()

            # --- Visual annotation ---
            if not args.no_display:
                labels = []
                class_ids = detections.class_id if detections.class_id is not None else []
                tracker_ids = detections.tracker_id if detections.tracker_id is not None else []
                for cid, tid in zip(class_ids, tracker_ids):
                    name = CLASS_MAPPINGS.get(int(cid), "?") if cid is not None else "?"
                    label = f"#{tid} {name}" if tid is not None else name
                    labels.append(label)

                annotated = circle_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                annotated = trace_annotator.annotate(scene=annotated, detections=detections)
                annotated = sv.draw_text(
                    scene=annotated,
                    text=f"Bird Visits Today: {bird_visit_count}",
                    text_anchor=sv.Point(x=200, y=40),
                    text_color=sv.Color.WHITE,
                    background_color=sv.Color.BLACK,
                    text_scale=1.0,
                    text_thickness=2,
                )

                preview = cv.resize(annotated, (video_width // 2, video_height // 2))
                cv.imshow("Chirp – Bird Counter", preview)
                if cv.waitKey(1) == ord("q"):
                    logger.info("User quit.")
                    break

    finally:
        feeder_camera.release()
        cv.destroyAllWindows()
        events_file.close()
        logger.info("Session complete. Total bird visits: %d", bird_visit_count)
        logger.info("Events logged to: %s", events_csv_path)


if __name__ == "__main__":
    main()

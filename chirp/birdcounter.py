"""Contains code implementing the core object detection + object tracking bird
feeder analysis pipeline. Uses input video feed from a file or video device and
outputs to an RTMP server.
"""
import argparse
from pathlib import Path
import argparse
from pathlib import Path

import numpy as np
import cv2 as cv

from ultralytics import YOLO
import supervision as sv

def main():

    # Set up argparser.
    parser = argparse.ArgumentParser(description="Chirp bird feeder live stream bird tracking")
    parser.add_argument("-v", 
                        "--video-source", 
                        default="/dev/video0", 
                        type=str,
                        help="Path to video device (/dev/videoX) or video file.")
    parser.add_argument("-w",
                        "--yolo-weights",
                        help="Filepath of the YOLOv8 detection model weights being used",
                        type=str)
    # Parse received arguments.
    args = parser.parse_args()
    video_source_path = Path(args.video_source)
    if not video_source_path.exists():
        raise FileNotFoundError(f"Failed to find video source at provided path: {video_source_path}")
    model_weights_path = Path(args.yolo_weights)
    if not model_weights_path.exists():
        raise FileNotFoundError(f"YOLOv8 model weights not found at provided path: {model_weights_path}")

    # TODO: Create a "BirdCounter" class of some sort where, given the parsed
    # arguments from whatever interface you're using above (like a CLI), you
    # instantiate a new BirdCounter instance and then run the stream as a method
    # of that class.

    # Set camera parameters.
    feeder_camera = cv.VideoCapture(str(video_source_path))
    if not feeder_camera.isOpened():
        raise Exception(f"Failed to open camera")
    else:
        video_framerate = feeder_camera.get(cv.CAP_PROP_FPS)
        print(f"Feeder camera framerate: {video_framerate}")

    # Define counting zone polygons.
    # Define zone that covers the entire camera frame. Handles counting all
    # tracked birds in view of the camera.
    video_height = int(feeder_camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_width = int(feeder_camera.get(cv.CAP_PROP_FRAME_WIDTH))
    print(f"Camera height/width: {video_height}, {video_width}")
    full_zone_polygon = np.array([[0,0],
                                 [video_width, 0],
                                 [video_width, video_height],
                                 [0, video_height]])
    full_zone = sv.PolygonZone(polygon=full_zone_polygon,
                               frame_resolution_wh=(video_width, video_height),
                               triggering_position=sv.Position.CENTER)
    
    # Load YOLO model weights.
    model = YOLO(model=model_weights_path)

    # Initilize a ByteTrack tracker.
    tracker = sv.ByteTrack(frame_rate=video_framerate)

    # Create a supervision bounding box annotator.
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()
    full_zone_annotator = sv.PolygonZoneAnnotator(zone=full_zone,
                                                  color=sv.Color.green())

    while True:
        status, frame = feeder_camera.read()

        # If the frame could not be grabbed successfully, bail out.
        if not status:
            print(f"Failed to retrieve frame from video {video_source_path}.")
            break

        # Grab frame attributes.
        frame_height, frame_width, _ = frame.shape
        preview_height = frame_height // 2
        preview_width = frame_width // 2

        # Run inference on the captured frame.
        result = model(frame)[0] # Only want the results for the single frame passed in.
        # Convert the results from ultralytics format to supervision's format.
        detections = sv.Detections.from_ultralytics(ultralytics_results=result)
        # Pass detections to tracker. The tracker will update the
        # "tracker_id"field for each detected instance.
        # TODO: Figure out if this should be done BEFORE or AFTER we filter out
        # detections from the detector. Something tells me ByteTrack does better
        # by using any low confidence detections. But before NMS as well?
        detections = tracker.update_with_detections(detections=detections)
        
        # Pass the tracked detections to any zones. The Zone objects will take
        # those tracked detections and determine the number of instances
        # currently in that zone.
        full_zone.trigger(detections=detections)

        # NMS_IOU_THRESHOLD = 0.5
        # filtered_detections = detections.with_nms(threshold=NMS_IOU_THRESHOLD)
        # # Filter out any remaining detections with confidence less than a
        # # specified threshold.
        # CONFIDENCE_THRESHOLD = 0.9
        # filtered_detections = filtered_detections[filtered_detections.confidence < CONFIDENCE_THRESHOLD]
        
        # Create labels maintained by model.
        # https://supervision.roboflow.com/how_to/track_objects/#annotate-video-with-tracking-ids
        labels = [f"{tracker_id} {model.names[class_id]}" for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)]

        # Add annotations to frame using supervision.
        annotated_image = bounding_box_annotator.annotate(scene=frame,
                                                          detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image,
                                                   detections=detections,
                                                   labels=labels)
        annotated_image = trace_annotator.annotate(scene=annotated_image,
                                                   detections=detections)
        # Annotate image with zone.
        annotated_image = full_zone_annotator.annotate(scene=annotated_image)
        

        # If successfully retrieved, display the retrieved frame. Create a
        # duplicate frame resized for viewing output.
        preview_frame = cv.resize(src=annotated_image, dsize=(preview_width, preview_height))

        # Display the resized, annotated frame.
        cv.imshow(winname="Camera Frames", mat=preview_frame)
        if cv.waitKey(1) == ord('q'):
            break

    # Before leaving, release the capture device (I.e., close).
    feeder_camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":

    main()
    
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
        print(f"Failed to open camera")
    else:
        print(f"Feeder camera framerate: {feeder_camera.get(cv.CAP_PROP_FPS)}")

    # Load YOLO model weights.
    model = YOLO(model=model_weights_path)

    # Create a supervision bounding box annotator.
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

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
        NMS_THRESHOLD = 0.5
        filtered_detections = detections.with_nms(threshold=NMS_THRESHOLD)

        # Create labels maintained by model.
        # https://supervision.roboflow.com/how_to/detect_and_annotate/#annotate-image
        labels = [model.names[class_id] for class_id in detections.class_id]

        # Add annotations to frame using supervision.
        annotated_image = bounding_box_annotator.annotate(scene=frame,
                                                          detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image,
                                                   detections=detections,
                                                   labels=labels)

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
    
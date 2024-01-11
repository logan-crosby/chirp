"""Contains code implementing the core object detection + object tracking bird
feeder analysis pipeline. Uses input video feed from a file or video device and
outputs to an RTMP server.
"""

import numpy as np
import cv2 as cv

def usbcam_detect_track_stream():

    camera_index = 0 # TODO: Parameterize.
    feeder_camera = cv.VideoCapture(camera_index)
    if not feeder_camera.isOpened():
        print(f"Failed to open camera")
    else:
        print(f"Feeder camera framerate: {feeder_camera.get(cv.CAP_PROP_FPS)}")

    while True:
        status, frame = feeder_camera.read()

        # If the frame could not be grabbed successfully, bail out.
        if not status:
            print(f"Failed to retrieve frame from camera {camera_index}.")
            break

        # Grab frame attributes.
        frame_height, frame_width, _ = frame.shape
        preview_height = frame_height // 2
        preview_width = frame_width // 2
    
        # If successfully retrieved, display the retrieved frame. Create a
        # duplicate frame resized for viewing output.
        preview_frame = cv.resize(src=frame, dsize=(preview_width, preview_height))
        cv.imshow(winname="Camera Frames", mat=preview_frame)
        if cv.waitKey(1) == ord('q'):
            break

    # Before leaving, release the capture device (I.e., close).
    feeder_camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":

    usbcam_detect_track_stream()
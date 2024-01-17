"""Contains code implementing the core object detection + object tracking bird
feeder analysis pipeline. Uses input video feed from a file or video device and
outputs to an RTMP server.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import argparse
from pathlib import Path
from csv import DictWriter

import numpy as np
import cv2 as cv

from ultralytics import YOLO
import supervision as sv

from zone_monitor import ZoneMonitor

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
    parser.add_argument("-o",
                        "--output-directory",
                        default=None,
                        help="Directory where any output files will be created.",
                        type=str)
    # Parse received arguments.
    args = parser.parse_args()
    video_source_path = Path(args.video_source)
    if not video_source_path.exists():
        raise FileNotFoundError(f"Failed to find video source at provided path: {video_source_path}")
    model_weights_path = Path(args.yolo_weights)
    if not model_weights_path.exists():
        raise FileNotFoundError(f"YOLOv8 model weights not found at provided path: {model_weights_path}")
    output_directory_path = Path(args.output_directory)
    if not output_directory_path.exists():
        raise FileNotFoundError(f"Provided directory {output_directory_path} does not exist or cannot be accessed.")

    # TODO: Create a "BirdCounter" class of some sort where, given the parsed
    # arguments from whatever interface you're using above (like a CLI), you
    # instantiate a new BirdCounter instance and then run the stream as a method
    # of that class.

    # Create a new output directory for session file storage if it doesn't
    # already exist.
    # TODO: This should (eventually) run in the constructor of the BirdCounter
    # class.
    sessions_directory = output_directory_path/"chirp_sessions"
    if not sessions_directory.exists():
        try:
            sessions_directory.mkdir()
            print(f"Successfully reated new directory for session file storage: {sessions_directory}")
        except Exception as exc:
            print(f"Failed to create session file directory in {output_directory_path}")
            raise(exc)
        
    
    # Create a directory for the newly started session.
    # TODO: This should be run only once the BirdCounter's "run" function (or
    # similar) has been invoked.
    session_datetime = datetime.now(tz=timezone.utc)
    current_session_directory = sessions_directory/f"chirp_session_{session_datetime}"
    try:
        current_session_directory.mkdir()
        print(f"Successfully created directory for new session: {current_session_directory}")
    except Exception as exc:
        print(f"Failed to create directory for new session.")
        raise(exc)
    
    # Open/create new events csv file for current session.
    session_events_filepath = current_session_directory/f"{session_datetime}_session_events.csv"
    session_events_file = open(file=session_events_filepath, mode='w', newline='')
    # Create new dictwriter and initialize the header in the file.
    fieldnames = ["datetime", "frame", "zone_id", "class_id", "tracker_id", "event_type_id"]
    session_writer = DictWriter(session_events_file, fieldnames=fieldnames)
    session_writer.writeheader()

    # TODO: Create file containing the class_id to string name mapping used
    # during this session. This is really a property of the model currently
    # being used, but in case the model changes, there's a record of this
    # mapping so the logged events don't become useless.
    # TODO: Also include the mapping from zone_id to zone name in this file.
    # Consider naming this file like "session_mappings.json" or something like
    # that. Can essentially just write those mapping dictionaries directly to
    # file as JSON (json.dumps(mapping_dict)).


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
    TRACK_BUFFER_S = 8
    full_zone_monitor = ZoneMonitor(in_threshold=int(video_framerate) // 2,
                                    out_timeout=int(video_framerate)*TRACK_BUFFER_S)
    total_bird_count = 0
    
    # Load YOLO model weights.
    model = YOLO(model=model_weights_path)

    # Initilize a ByteTrack tracker.
    tracker = sv.ByteTrack(frame_rate=video_framerate,
                           track_buffer=video_framerate*TRACK_BUFFER_S)

    # Create a supervision bounding box annotator.
    bounding_box_annotator = sv.CircleAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()
    full_zone_annotator = sv.PolygonZoneAnnotator(zone=full_zone,
                                                  color=sv.Color.green())
    
    # Create a counter to maintain each frame's index.
    frame_counter = 0

    while True:
        
        # Grab frame from camera.
        status, frame = feeder_camera.read()

        # If the frame could not be grabbed successfully, bail out.
        if not status:
            print(f"Failed to retrieve frame from video {video_source_path}.")
            break
        
        # Increment the frame counter.
        frame_counter += 1
        frame_datetime = datetime.now(tz=timezone.utc)

        # Grab frame attributes.
        frame_height, frame_width, _ = frame.shape
        preview_height = frame_height // 2
        preview_width = frame_width // 2

        # Run inference on the captured frame.
        result = model(frame, verbose=False)[0] # Only want the results for the single frame passed in.
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
        # TODO: Have to use the boolean (mask) list returned by this function to
        # determine which detections are in the zone and which are not.
        zone_detections_mask = full_zone.trigger(detections=detections)
        detections_in_zone = detections[zone_detections_mask]
        entered_events, exited_events = full_zone_monitor.update(detections_in_zone=detections_in_zone,
                                                                 frame_index=frame_counter,
                                                                 frame_datetime=frame_datetime)
        total_bird_count += len(entered_events)

        # Log the events from the "full_zone" to file. 
        # TODO: Encapsulate this behavior so that it can be extended to any
        # zone. Consider including it in the zone monitor itself.
        for entrance_event in entered_events:
            event = {"datetime": entrance_event["datetime_entered"],
                      "frame": entrance_event["frame_entered"],
                      "zone_id": 0,
                      "class_id": entrance_event["last_detection"][3],
                      "tracker_id": entrance_event["last_detection"][4],
                      "event_type_id": 0
                    }
            session_writer.writerow(rowdict=event)
            print(f"{datetime.now()} - A {model.names[entrance_event['last_detection'][3]]} (Tracker ID {entrance_event['last_detection'][4]}) arrived.")
        
        for exited_event in exited_events:
            event = {"datetime": exited_event["datetime_exited"],
                      "frame": exited_event["frame_exited"],
                      "zone_id": 0,
                      "class_id": exited_event["last_detection"][3],
                      "tracker_id": exited_event["last_detection"][4],
                      "event_type_id": 1
                    }
            print(f"{datetime.now()} - {model.names[exited_event['last_detection'][3]]} (Tracker ID {exited_event['last_detection'][4]}) headed out.")
        
        # for detection in exited_detections:
        #     print(f"{datetime.now()} - {model.names[detection[3]]} (Tracker ID {detection[4]}) headed out.")
        #     # It would also be cool to print out the elapsed time, how long it
        #     # stayed. Can add this if I have the timestamp in the returned
        #     # detections as well. Not yet, but maybe a future feature. That's
        #     # simple enough to compute externally in a spreadsheet, too.

        # TODO: Add some temporary dictionary to track the number of each
        # species and use that (again, temporarily) to annotate the stream with.

        # TODO Set up a logger and log these human readable outputs to a
        # file--or just use a simple file for now.
        # TODO Set up logging the entered and exited detections to a CSV or
        # excel workbook. OR going directly to google sheets.
        # TODO: Could even take stats/metrics (as soon as they're computed
        # externally) and report them with a YouTube chat bot. I.e., a bird
        # shows up and then leaves. Leaving could trigger an event for the bot
        # to report "Black capped chickadee (ID 3021) just headed out :wave".
        # TODO: Reply to supervision thread to see if this is something that
        # they'd want to add to the PolygonZone class.


        # TODO: Rough idea:
        # For the detections that are within the zone: maintain a dictionary of
        # detections within the zone and their tracker_id. Maybe use a counter.
        # Basically, increment a count for each element for each frame that they
        # are present.
        # OR, set a timeout number of frames for each tracker_id. Each frame
        # that a tracker_id is present, just set the number to whatever the
        # timeout is. If a tracker_id is in the dictionary and is not present in
        # the current frame, then decrement its timeout counter by one. If the
        # timeout counter goes to zero, then remove that tracked instance from
        # the dictionary. Make sure that timeout matches the tracker's timeout
        # if you to avoid some duplicate counting. Ideally, the tracker doesn't
        # lose the object and maintains a track on it--but I think that's also a
        # function of the detection model being used. Need to look into the
        # tracker's parameters.

        # NOTE: Does a separate "in-count" and "out-count" make sense for the
        # zone? I.e., I feel like I only care about "how many objects were in
        # this zone cumulatively?" Maybe that's the same thing as saying "How
        # many objects entered this zone," where enter could mean "appeared" or
        # came in from outside of the zone. In theory, though, the number of
        # instances that enter a zone should also exit a zone (whether exit
        # means "disappear suddenly" or move from the zone to a region still in
        # view but no longer in the zone). I feel like those would always be the
        # same?
        
        # UNLESS you are in fact interested in knowing how many objects entered
        # a zone in a certain time range. Granted, you should just be able to
        # determine this from the change in count.
        # BUT, what if you want to know how many instances LEFT/exited a
        # particular zone during a certain time range. Well, in that case,
        # yes--you could increment a "out-count" that rises as elements leave.
        
        # BUT if you're really interested in that, you could also just
        # plot/store the number of instances in the zone at each unit time, and
        # then figure out how many left in that time range later. I suppose you
        # could say the same about how many were in the zone in total /
        # cumulatively--I.e., it's something that you could extract from
        # recorded data later on / after the fact.

        # ALSO: On a slightly different note--I think there also needs to be
        # some kind of threshold number of frames that a tracker_id must be
        # present for before we can count it as a present object. I.e., if a
        # cardinal shows up, but is mistakenly classified as a male housefinch
        # for two frames, I don't want that housefinch getting counted--as its
        # not really there, and was only detected as that for two frames.
        # Rather, I only want to count an instance of a class if it has been
        # tracked for >= some number of frames.

        # So maybe, when a new tracker_id is added to the dict, we have a couple
        # of data points/counters associated with it:
        # 1. Timestamp / time arrived?
        # 2. Number of frames appeared in (that tracker_id is present in)
        # 3. Timeout frames == refreshed/set to TIMEOUT each time the tracker_id
        #    is present.
        # 

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
        # annotated_image = full_zone_annotator.annotate(scene=annotated_image,
        #                                                label="")
        # Annotate image with total bird count.
        annotated_image = sv.draw_text(scene=annotated_image, 
                                       text=f"Birds Seen Today: {total_bird_count}", 
                                       text_anchor=sv.Point(x=200, y=40),
                                       text_color=sv.Color.green(),
                                       text_scale=1.0,
                                       text_thickness=2)

        # If successfully retrieved, display the retrieved frame. Create a
        # duplicate frame resized for viewing output.
        preview_frame = cv.resize(src=annotated_image, dsize=(preview_width, preview_height))

        # Display the resized, annotated frame.
        cv.imshow(winname="Camera Frames", mat=preview_frame)
        if cv.waitKey(1) == ord('q'):
            break

    # Before leaving, release the capture device (I.e., close) and close any
    # open files.
    feeder_camera.release()
    cv.destroyAllWindows()
    session_events_file.close()

if __name__ == "__main__":

    main()
    
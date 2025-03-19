import time
import numpy as np
from collections import defaultdict
from utilities.detect_stop import stop_detection_memory, STOP_THRESHOLD_FRAMES, MOVEMENT_THRESHOLD
from utilities.detect_overlap import check_overlap_from_boxes
from utilities.get_orientation import get_plank_orientation


CONFIDENCE_THRESHOLD = 0.3
MINIMUM_W_THRESHOLD = 20 # in pixels
MINIMUM_H_THRESHOLD = 20 # in pixels

orientation_memory = defaultdict(lambda: {"orientation": "unknown", "angle": 0, "aspect_ratio": 0})


def process_frame(frame, model, tracker, server):
    """
    Process each frame for object detection, tracking, orientation analysis, and status monitoring.
    
    This function is the main processing pipeline that:
    1. Performs object detection using a YOLO model to identify planks
    2. Filters detections based on confidence and minimum size thresholds
    3. Tracks objects across frames to maintain consistent IDs
    4. Analyzes each plank's orientation (correct/vertical vs incorrect/horizontal)
    5. Detects when planks have stopped moving
    6. Identifies overlapping planks that may cause issues
    7. Updates the server with the current status of all tracked objects
    8. Measures and reports performance metrics for each processing stage
    
    The function maintains memory of object orientations and movement history
    across frames to provide stable tracking and status detection.
    
    Args:
        frame: The input video frame to process
        model: The YOLO object detection model
        tracker: Object tracker that maintains object IDs across frames
        server: Server instance to update with current detection status
        
    Returns:
        tuple: (tracked_objects, final_orientations, roi_bounds)
            - tracked_objects: List of tracked objects with position and ID information
            - final_orientations: List of orientation data for each tracked object
            - roi_bounds: Region of interest boundaries (start_x, end_x)
    """
    global orientation_memory, stop_detection_memory
    
    process_start = time.time()

    height, width = frame.shape[:2]
    # print(f"YOLO input shape: {width}x{height}")
    
    # YOLO detection on ROI frame
    yolo_start = time.time()

    # the trained model only detects the plank class
    results = model(frame, conf=0.5)[0]

    # but if the model is trained on multiple classes, you can specify the class
    # results = model(frame, conf=0.5, classes=[0])[0]
    yolo_end = time.time()

    # print(f"\nDetected objects: {len(results.boxes)}")

    # Parse detections
    parse_start = time.time()
    detections = []
    detection_points = {}
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        
        # Get class name from model's names dictionary
        class_name = results.names[class_id]
        
        # Only process if it's a plank
        if class_name != "plank":
            continue

        # Calculate width and height
        width = x2 - x1
        height = y2 - y1
        # 20 represents the minimum size threshold (in pixels) for both the width and height of detected objects
        if confidence > CONFIDENCE_THRESHOLD and width > MINIMUM_W_THRESHOLD and height > MINIMUM_H_THRESHOLD:
            bbox = [x1, y1, width, height]
            detections.append((bbox, confidence, class_id))

            points = np.array([
                [x1, y1], [x2, y1],
                [x2, y2], [x1, y2]
            ], dtype=np.float32)

            # Store points using bbox as key
            detection_points[tuple(bbox)] = points
    parse_end = time.time()

    # Tracking
    track_start = time.time()
    tracked_objects = tracker.update_tracks(detections, frame=frame)
    tracked_objects = [t for t in tracked_objects if t.is_confirmed() and t.time_since_update <= 1]
    track_end = time.time()

    # Orientation and stop detection processing
    orient_start = time.time()
    final_orientations = []
    
    # Store overlap pairs
    overlapped_pairs = []
    
    # Check for overlaps between all pairs of tracked objects
    for i, track1 in enumerate(tracked_objects):
        ltrb1 = track1.to_ltrb()
        
        # Check overlap with all other boxes
        for j, track2 in enumerate(tracked_objects[i+1:], i+1):
            ltrb2 = track2.to_ltrb()
            
            is_overlapped, percent1, percent2 = check_overlap_from_boxes(ltrb1, ltrb2)
            if is_overlapped:
                overlapped_pairs.append((track1.track_id, track2.track_id))
                # print(f"Warning: Overlap detected between planks {track1.track_id} and {track2.track_id}")
                # print(f"Overlap percentage: {percent1:.1f}% of plank {track1.track_id}, "
                #       f"{percent2:.1f}% of plank {track2.track_id}")

    # Continue with existing orientation processing
    for track in tracked_objects:
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        width = x2 - x1
        height = y2 - y1
        
        bbox = [x1, y1, width, height]
        points = detection_points.get(tuple(bbox), np.array([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2]
        ], dtype=np.float32))

        # Calculate orientation first
        orientation, angle, aspect_ratio = get_plank_orientation(points, width, height)
        if orientation != "unknown":
            orientation_memory[track_id] = {
                "orientation": orientation,
                "angle": angle,
                "aspect_ratio": aspect_ratio,
                "in_roi": True
            }
        
        # Calculate center point for stop detection
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        current_pos = (center_x, center_y)
        
        # Update stop detection memory
        stop_memory = stop_detection_memory[track_id]
        stop_memory["positions"].append(current_pos)
        
        # Keep only last STOP_THRESHOLD_FRAMES positions
        if len(stop_memory["positions"]) > STOP_THRESHOLD_FRAMES:
            stop_memory["positions"].pop(0)
        
        # Check if object has stopped
        if len(stop_memory["positions"]) == STOP_THRESHOLD_FRAMES:
            max_movement = max(
                abs(current_pos[0] - pos[0]) + abs(current_pos[1] - pos[1])
                for pos in stop_memory["positions"][:-1]
            )
            
            if max_movement < MOVEMENT_THRESHOLD:
                stop_memory["stop_frames"] += 1
                if stop_memory["stop_frames"] >= STOP_THRESHOLD_FRAMES:
                    stop_memory["is_stopped"] = True
                    # print(f"Plank {track_id} has stopped in orientation: {orientation}")
            else:
                stop_memory["stop_frames"] = 0
                stop_memory["is_stopped"] = False
                # print(f"Plank {track_id} has started moving again")

        memory = orientation_memory[track_id]
        final_orientations.append((
            points,
            memory["angle"],
            memory["orientation"],
            memory["aspect_ratio"],
            True,  # Always in ROI since we're only processing ROI
            stop_memory["is_stopped"]  # Add stop status to orientations
        ))

    # Clean up memory for tracks that are no longer active
    active_track_ids = {track.track_id for track in tracked_objects}
    for track_id in list(orientation_memory.keys()):
        if track_id not in active_track_ids:
            del orientation_memory[track_id]
            del stop_detection_memory[track_id]  # Clean up stop detection memory too
    orient_end = time.time()
    
    process_end = time.time()

    # Calculate all time differences
    yolo_time = yolo_end - yolo_start
    parse_time = parse_end - parse_start
    track_time = track_end - track_start
    orient_time = orient_end - orient_start
    total_time = process_end - process_start
    
    # print(
    #     f"[process_frame] Total: {total_time:.3f}s | "
    #     f"YOLO: {yolo_time:.3f}s | "
    #     f"Parse: {parse_time:.3f}s | "
    #     f"Track: {track_time:.3f}s | "
    #     f"Orientation: {orient_time:.3f}s"
    # )

    # Update server with current status - now using pre-calculated overlapped_pairs
    server.update_status(
        overlapped=overlapped_pairs,
        stopped=[track.track_id 
                for track, (_, _, _, _, _, is_stopped) in zip(tracked_objects, final_orientations) 
                if is_stopped],
        incorrect=[track.track_id 
                  for track, (_, _, orientation, _, _, _) in zip(tracked_objects, final_orientations) 
                  if orientation == "incorrect"]
    )

    return tracked_objects, final_orientations, (0, frame.shape[1])


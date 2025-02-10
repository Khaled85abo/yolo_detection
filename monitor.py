import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
# Add this at the top with other imports
from collections import defaultdict
from picamera2 import Picamera2
from libcamera import controls
import time

# Add this as a global variable
orientation_memory = defaultdict(lambda: {"orientation": "unknown", "angle": 0, "aspect_ratio": 0})

# Initialize YOLO model
model = YOLO('/home/rise/enter/train_yolo11x/weights/best_yolo11x.pt')
output_path = 'videos_output/yolo11s/output_ori_1.mp4'
# Initialize DeepSORT tracker
# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=30,              # Maximum number of frames to keep dead tracks
    n_init=2,                # Number of frames for track initialization
    nms_max_overlap=0.7,     # NMS threshold for suppressing overlapping detections
    max_iou_distance=0.7,    # Maximum IOU distance for matching
    max_cosine_distance=0.3, # Maximum cosine distance for feature matching
    nn_budget=100,           # Maximum size of the appearance descriptors gallery
    embedder="mobilenet",    # Feature extractor
    half=True,              # Use half precision for better speed
    bgr=True,
    embedder_gpu=True
)

def process_frame(frame, model, tracker):
    """
    Process each frame for object detection and orientation tracking
    """
    global orientation_memory
    
    process_start = time.time()
    
    frame_height, frame_width = frame.shape[:2]
    roi_x_start = int(frame_width * 0.35)
    roi_x_end = int(frame_width * 0.65)

    # YOLO detection
    yolo_start = time.time()
    results = model(frame, conf=0.5)[0]
    yolo_end = time.time()

    print(f"\nDetected objects: {len(results.boxes)}")

    # Parse detections
    parse_start = time.time()
    detections = []
    detection_points = {}
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())

        # Calculate width and height
        width = x2 - x1
        height = y2 - y1

        if confidence > 0.3 and width > 20 and height > 20:
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

    # Orientation processing
    orient_start = time.time()
    final_orientations = []
    for track in tracked_objects:
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) // 2

        # Add status print for each tracked object
        roi_status = "in ROI" if roi_x_start <= center_x <= roi_x_end else "outside ROI"
        print(f"Track ID {track_id}: {roi_status}, Size: {width}x{height}")

        bbox = [x1, y1, width, height]
        points = detection_points.get(tuple(bbox), np.array([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2]
        ], dtype=np.float32))

        if center_x < roi_x_start or center_x > roi_x_end:
            orientation_memory[track_id] = {
                "orientation": "unknown",
                "angle": 0,
                "aspect_ratio": 0,
                "in_roi": False
            }
        else:
            orientation, angle, aspect_ratio = get_plank_orientation(points, width, height)
            if orientation != "unknown":
                orientation_memory[track_id] = {
                    "orientation": orientation,
                    "angle": angle,
                    "aspect_ratio": aspect_ratio,
                    "in_roi": True
                }

        memory = orientation_memory[track_id]
        final_orientations.append((
            points,
            memory["angle"],
            memory["orientation"],
            memory["aspect_ratio"],
            memory.get("in_roi", False)
        ))

    # Clean up memory for tracks that are no longer active
    active_track_ids = {track.track_id for track in tracked_objects}
    for track_id in list(orientation_memory.keys()):
        if track_id not in active_track_ids:
            del orientation_memory[track_id]
    orient_end = time.time()
    
    process_end = time.time()

    # Calculate all time differences
    yolo_time = yolo_end - yolo_start
    parse_time = parse_end - parse_start
    track_time = track_end - track_start
    orient_time = orient_end - orient_start
    total_time = process_end - process_start
    
    print(
        f"[process_frame] Total: {total_time:.3f}s | "
        f"YOLO: {yolo_time:.3f}s | "
        f"Parse: {parse_time:.3f}s | "
        f"Track: {track_time:.3f}s | "
        f"Orientation: {orient_time:.3f}s"
    )

    return tracked_objects, final_orientations, (roi_x_start, roi_x_end)

def get_plank_orientation(points, width, height):
    """
    Determine plank orientation based on bounding box characteristics
    """
    # Calculate aspect ratio
    aspect_ratio = width / height if height != 0 else 0

    # Calculate rotated rectangle for angle
    rect = cv2.minAreaRect(points)
    angle = rect[2]

    # Normalize angle
    if width < height:
        angle = angle - 90
    if angle < -90:
        angle += 180
    elif angle > 90:
        angle -= 180

    # More lenient aspect ratio thresholds
    if 0.2 <= aspect_ratio <= 0.6:  # Wider range for correct orientation
        orientation = "correct"
        print("Orientation: correct")
    else:
        orientation = "incorrect"
        print("Orientation: incorrect")

    return orientation, angle, aspect_ratio

def draw_boxes_and_orientations(frame, tracked_objects, orientations, roi_bounds):
    """
    Draw bounding boxes, track IDs, and orientations on the frame
    """
    roi_x_start, roi_x_end = roi_bounds
    frame_height = frame.shape[0]

    # Draw ROI region with semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (roi_x_start, frame_height), (100, 100, 100), -1)
    cv2.rectangle(overlay, (roi_x_end, 0), (frame.shape[1], frame_height), (100, 100, 100), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.line(frame, (roi_x_start, 0), (roi_x_start, frame_height), (255, 255, 0), 2)
    cv2.line(frame, (roi_x_end, 0), (roi_x_end, frame_height), (255, 255, 0), 2)

    used_positions = {}

    for track, (points, angle, orientation, aspect_ratio, in_roi) in zip(tracked_objects, orientations):
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Set color based on orientation, but only if in ROI
        if in_roi:
            if orientation == "correct":
                color = (0, 255, 0)  # Green for correct orientation
            elif orientation == "incorrect":
                color = (0, 0, 255)  # Red for incorrect orientation
        else:
            color = (128, 128, 128)  # Gray for outside ROI

        # Draw rotated bounding box
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int_(box)
        cv2.drawContours(frame, [box], 0, color, 2)

        # Only show orientation information if in ROI
        if in_roi:
            label_y = y1 - 10
            while (x1, label_y) in used_positions:
                label_y -= 20
            used_positions[(x1, label_y)] = True

            # Draw information
            label = f"ID: {track_id} | {orientation} | AR: {aspect_ratio:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, label_y - text_height - 4),
                         (x1 + text_width, label_y + 4), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw orientation line only in ROI
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 4, color, -1)

            line_length = min(x2 - x1, y2 - y1) // 2
            end_x = center_x + int(line_length * np.cos(np.radians(angle)))
            end_y = center_y + int(line_length * np.sin(np.radians(angle)))
            cv2.line(frame, (center_x, center_y), (end_x, end_y), color, 2)
        else:
            # Only show track ID outside ROI
            label_y = y1 - 10
            label = f"ID: {track_id}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, label_y - text_height - 4),
                         (x1 + text_width, label_y + 4), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def main():
    print("Initializing camera...")
    picam2 = Picamera2()

    print("Configuring camera settings...")
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameDurationLimits": (33333, 33333)}  # ~30fps
    )
    picam2.configure(config)

    # picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    picam2.set_controls({"AeEnable": True})

    print("Starting camera...")
    picam2.start()

    width, height = 640, 480
    custom_fps = 10

    print(f"Initializing video writer (Output: {output_path})")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, custom_fps, (width, height))

    print("\nStarting detection and tracking. Press 'q' to quit.")
    try:
        frame_count = 0
        while True:
            frame_count += 1
            print(f"\n--- Frame {frame_count} ---")
            
            loop_start = time.time()
            
            # Capture frame
            capture_start = time.time()
            frame = picam2.capture_array()
            capture_end = time.time()
            
            # Color conversion
            color_conv_start = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            color_conv_end = time.time()
            
            # Process frame
            process_start = time.time()
            tracked_objects, orientations, roi_bounds = process_frame(frame, model, tracker)
            process_end = time.time()
            
            # Print summary
            objects_in_roi = sum(1 for _, _, _, _, in_roi in orientations if in_roi)
            print(f"Summary: {len(tracked_objects)} tracked objects, {objects_in_roi} in ROI")
            
            # Draw results
            draw_start = time.time()
            frame = draw_boxes_and_orientations(frame, tracked_objects, orientations, roi_bounds)
            draw_end = time.time()
            
            # Write frame
            out_write_start = time.time()
            out.write(frame)
            out_write_end = time.time()
            
            loop_end = time.time()
            
            print(
                f"[main loop] Frame {frame_count} total: {(loop_end - loop_start):.3f}s | "
                f"capture: {(capture_end - capture_start):.3f}s | "
                f"color_conv: {(color_conv_end - color_conv_start):.3f}s | "
                f"process_frame: {(process_end - process_start):.3f}s | "
                f"draw: {(draw_end - draw_start):.3f}s | "
                f"write: {(out_write_end - out_write_start):.3f}s"
            )

    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        print("Cleaning up...")
        picam2.stop()
        out.release()
        print("Done!")

if __name__ == "__main__":
    main()
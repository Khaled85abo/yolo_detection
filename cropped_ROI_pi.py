import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
# Add this at the top with other imports
from collections import defaultdict
from picamera2 import Picamera2
import time

# Add this as a global variable
orientation_memory = defaultdict(lambda: {"orientation": "unknown", "angle": 0, "aspect_ratio": 0})

# ROI parameters
ROI_start = 0.40
ROI_end = 0.60

# aspect ratio
aspect_ratio_threshold = 0.60

# Initialize YOLO model
model = YOLO('/home/rise/enter/train_yolo11n/weights/best_yolo11n.pt')
output_path = 'videos_output/yolo11n/pi/output_ori_1.mp4'
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
    
    # YOLO detection on ROI frame
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
        
        # Get class name from model's names dictionary
        class_name = results.names[class_id]
        
        # Only process if it's a plank
        if class_name != "plank":
            continue

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

        bbox = [x1, y1, width, height]
        points = detection_points.get(tuple(bbox), np.array([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2]
        ], dtype=np.float32))

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
            True  # Always in ROI since we're only processing ROI
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

    return tracked_objects, final_orientations, (0, frame.shape[1])

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

    # For vertical planks (correct orientation), aspect ratio should be < 1
    if aspect_ratio < aspect_ratio_threshold:  # height is significantly larger than width
        print("Orientation: correct")
        orientation = "correct"
    # elif aspect_ratio > 1.5:  # width is significantly larger than height
    #     orientation = "incorrect"
    else:
        print("Orientation: incorrect")
        orientation = "incorrect"  # For cases where orientation is ambiguous


    return orientation, angle, aspect_ratio

def draw_boxes_and_orientations(frame, tracked_objects, orientations, roi_bounds):
    """
    Draw bounding boxes, track IDs, and orientations on the frame
    """
    used_positions = {}

    for track, (points, angle, orientation, aspect_ratio, _) in zip(tracked_objects, orientations):
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Set color based on orientation
        if orientation == "correct":
            color = (0, 255, 0)  # Green for correct orientation
        elif orientation == "incorrect":
            color = (0, 0, 255)  # Red for incorrect orientation
        else:
            color = (128, 128, 128)  # Gray for unknown

        # Draw rotated bounding box
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int_(box)
        cv2.drawContours(frame, [box], 0, color, 2)

        # Draw information
        label_y = y1 - 10
        while (x1, label_y) in used_positions:
            label_y -= 20
        used_positions[(x1, label_y)] = True

        label = f"ID: {track_id} | {orientation} | AR: {aspect_ratio:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, label_y - text_height - 4),
                     (x1 + text_width, label_y + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw orientation line
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame, (center_x, center_y), 4, color, -1)

        line_length = min(x2 - x1, y2 - y1) // 2
        end_x = center_x + int(line_length * np.cos(np.radians(angle)))
        end_y = center_y + int(line_length * np.sin(np.radians(angle)))
        cv2.line(frame, (center_x, center_y), (end_x, end_y), color, 2)

    return frame

def main():
    global output_path
    print("Initializing camera...")
    picam2 = Picamera2()

    print("Configuring camera settings...")
    full_width, full_height = 640, 480
    roi_width = int(full_width * (ROI_end - ROI_start))
    
    print(f"Full dimensions: {full_width}x{full_height}")
    print(f"ROI width: {roi_width}")
    
    config = picam2.create_video_configuration(
        main={"size": (full_width, full_height), "format": "RGB888"},
        controls={"FrameDurationLimits": (33333, 33333)}  # ~30fps
    )
    picam2.configure(config)

    picam2.set_controls({"AeEnable": True})

    print("Starting camera...")
    picam2.start()

    custom_fps = 10

    print(f"Initializing video writer (Output: {output_path})")
    print(f"Output dimensions: {roi_width}x{full_height}")
    
    # Try different codecs in order of preference, using simpler codecs
    codecs = [
        ('MJPG', '.avi'),    # Motion JPEG
        ('XVID', '.avi'),    # XVID
        ('I420', '.avi'),    # Raw I420 (YUV)
        ('DIVX', '.avi'),    # DIVX
        ('YUV4', '.avi'),    # Raw YUV
    ]
    
    out = None
    base_path = output_path.rsplit('.', 1)[0]
    
    for codec, ext in codecs:
        try:
            current_output = base_path + ext
            print(f"Trying codec {codec} with output: {current_output}")
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_out = cv2.VideoWriter(current_output, fourcc, custom_fps, (roi_width, full_height))
            
            # Try writing a test frame to verify it works
            test_frame = np.zeros((full_height, roi_width, 3), dtype=np.uint8)
            if test_out.isOpened() and test_out.write(test_frame):
                out = test_out
                output_path = current_output
                print(f"Successfully opened VideoWriter with codec {codec}")
                break
            else:
                print(f"Codec {codec} opened but failed to write")
                test_out.release()
        except Exception as e:
            print(f"Failed to open VideoWriter with codec {codec}: {str(e)}")
    
    if out is None:
        print("Error: Could not initialize VideoWriter with any codec!")
        return

    print(f"Final output path: {output_path}")

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
            
            # Crop frame to ROI
            roi_x_start = int(full_width * ROI_start)
            roi_x_end = int(full_width * ROI_end)
            frame = frame[:, roi_x_start:roi_x_end]
            
            # Add dimension check
            current_height, current_width = frame.shape[:2]
            if current_width != roi_width or current_height != full_height:
                print(f"Warning: Frame dimensions ({current_width}x{current_height}) "
                      f"don't match expected dimensions ({roi_width}x{full_height})")
            
            # Process frame
            process_start = time.time()
            tracked_objects, orientations, roi_bounds = process_frame(frame, model, tracker)
            process_end = time.time()
            
            # Print summary
            objects_in_roi = len(tracked_objects)
            print(f"Summary: {objects_in_roi} tracked objects")
            
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
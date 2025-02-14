import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
# Add this at the top with other imports
from collections import defaultdict

# Add this as a global variable
orientation_memory = defaultdict(lambda: {"orientation": "unknown", "angle": 0, "aspect_ratio": 0})

# ROI parameters
ROI_start = 0.40
ROI_end = 0.60

# aspect ratio
aspect_ratio_threshold = 0.60
# Initialize YOLO model
model = YOLO('C:/Users/khale/LIA/train_yolo11n/weights/best_yolo11n.pt')
print("Available classes in the model:", model.names)
video_path = 'C:/Users/khale/LIA/data/3.mp4'
output_path = 'videos_output/yolo11n/output_ori_5.mp4'
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
    
    # YOLO detection on ROI frame
    results = model(frame, conf=0.5)[0]
    
    # Print detected objects and their classes
    print(f"\nDetected objects: {len(results.boxes)}")
    
    detections = []
    detection_points = {}  # Use dictionary to maintain correspondence
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        
        print(f"Detected {class_name} (ID: {class_id}) with confidence: {confidence:.2f}")
        
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
    
    # Update tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)
    tracked_objects = [t for t in tracked_objects if t.is_confirmed() and t.time_since_update <= 1]
    
    # Process orientations for tracked objects
    final_orientations = []
    for track in tracked_objects:
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        width = x2 - x1
        height = y2 - y1
        
        # Reconstruct bbox in same format as detections
        bbox = [x1, y1, width, height]
        points = detection_points.get(tuple(bbox), np.array([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2]
        ], dtype=np.float32))
        
        # Update orientation
        orientation, angle, aspect_ratio = get_plank_orientation(points, width, height)
        if orientation != "unknown":
            orientation_memory[track_id] = {
                "orientation": orientation,
                "angle": angle,
                "aspect_ratio": aspect_ratio,
                "in_roi": True
            }
        
        # Use remembered orientation
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
        orientation = "correct"
    # elif aspect_ratio > 1.5:  # width is significantly larger than height
    #     orientation = "incorrect"
    else:
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
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    custom_fps = 10
    
    # Calculate ROI dimensions for output video
    roi_width = int(width * (ROI_end - ROI_start))
    
    # Initialize video writer with ROI dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, custom_fps, (roi_width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Crop frame to ROI before any processing
        roi_x_start = int(width * ROI_start)
        roi_x_end = int(width * ROI_end)
        frame = frame[:, roi_x_start:roi_x_end]
            
        # Process frame
        tracked_objects, orientations, roi_bounds = process_frame(frame, model, tracker)
        
        # Draw results
        frame = draw_boxes_and_orientations(frame, tracked_objects, orientations, roi_bounds)
        
        # Write frame to output video
        out.write(frame)
        
        # Display frame
        cv2.imshow('Tracking with Orientation', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

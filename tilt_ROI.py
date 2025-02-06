import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

# Initialize YOLO model
model = YOLO('C:/Users/khale/LIA/train/best.pt')  # or use your custom trained model path
video_path = 'C:/Users/khale/LIA/data/3.mp4'
output_path = 'videos_output/output_ori_1.mp4'
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

def get_plank_orientation(points, width, height):
    """
    Determine plank orientation based on bounding box characteristics
    Args:
        points: numpy array of box corner points
        width: width of the bounding box
        height: height of the bounding box
    Returns:
        orientation: "correct" or "incorrect"
        angle: orientation angle in degrees
        aspect_ratio: width/height ratio for debugging
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
    
    # Define correct orientation criteria
    # When plank is correctly oriented, height should be 2-4 times the width
    if 0.25 <= aspect_ratio <= 0.5:  # This means height is 2-4 times the width
        orientation = "correct"
    else:
        orientation = "incorrect"
        
    return orientation, angle, aspect_ratio

def process_frame(frame, model, tracker):
    """
    Process each frame for object detection and orientation tracking
    """
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Define the region of interest (middle of the frame)
    roi_x_start = int(frame_width * 0.4)  # Start at 40% of frame width
    roi_x_end = int(frame_width * 0.6)    # End at 60% of frame width
    roi_width = roi_x_end - roi_x_start
    
    results = model(frame, conf=0.5)[0]
    
    detections = []
    orientations = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        
        # Calculate width and height
        width = x2 - x1
        height = y2 - y1
        
        # Calculate center point of the detection
        center_x = (x1 + x2) // 2
        
        if confidence > 0.5 and width > 20 and height > 20:
            bbox = [x1, y1, width, height]
            detections.append((bbox, confidence, class_id))
            
            points = np.array([
                [x1, y1], [x2, y1],
                [x2, y2], [x1, y2]
            ], dtype=np.float32)
            
            # Only perform orientation detection if object is in ROI
            if roi_x_start <= center_x <= roi_x_end:
                orientation, angle, aspect_ratio = get_plank_orientation(points, width, height)
            else:
                orientation = "unknown"
                angle = 0
                aspect_ratio = width / height if height != 0 else 0
                
            orientations.append((points, angle, orientation, aspect_ratio))
    
    tracked_objects = tracker.update_tracks(detections, frame=frame)
    tracked_objects = [t for t in tracked_objects if t.is_confirmed() and t.time_since_update <= 1]
    
    return tracked_objects, orientations, (roi_x_start, roi_x_end)

def draw_boxes_and_orientations(frame, tracked_objects, orientations, roi_bounds):
    """
    Draw bounding boxes, track IDs, and orientations on the frame
    """
    roi_x_start, roi_x_end = roi_bounds
    frame_height = frame.shape[0]
    
    # Draw ROI region
    cv2.line(frame, (roi_x_start, 0), (roi_x_start, frame_height), (255, 255, 0), 2)
    cv2.line(frame, (roi_x_end, 0), (roi_x_end, frame_height), (255, 255, 0), 2)
    
    used_positions = {}
    
    for track, (points, angle, orientation, aspect_ratio) in zip(tracked_objects, orientations):
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        width = x2 - x1
        height = y2 - y1
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        
        # Set color based on orientation and position
        if roi_x_start <= center_x <= roi_x_end:
            if orientation == "correct":
                color = (0, 255, 0)  # Green for correct orientation in ROI
            else:
                color = (0, 0, 255)  # Red for incorrect orientation in ROI
        else:
            color = (128, 128, 128)  # Gray for objects outside ROI
        
        # Draw rotated bounding box
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int_(box)
        cv2.drawContours(frame, [box], 0, color, 2)
        
        # Calculate label position
        label_y = y1 - 10
        while (x1, label_y) in used_positions:
            label_y -= 20
        used_positions[(x1, label_y)] = True
        
        # Draw information
        if roi_x_start <= center_x <= roi_x_end:
            label = f"ID: {track_id} | {orientation} | AR: {aspect_ratio:.2f}"
        else:
            label = f"ID: {track_id} | outside ROI"
            
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, label_y - text_height - 4), 
                     (x1 + text_width, label_y + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw center point and orientation line only in ROI
        if roi_x_start <= center_x <= roi_x_end:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
            
            line_length = min(width, height) // 2
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
    custom_fps = 10  # or any other value

    
    # Initialize video writer
    # Different codec options:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
    # # or
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    # # or
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI format
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    out = cv2.VideoWriter(output_path, fourcc, custom_fps, (width, height))

    # optional: Set video writer properties
    if hasattr(out, 'set'):
        out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)  # Set quality (0-100)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        tracked_objects, orientations = process_frame(frame, model, tracker)
        
        # Draw results
        frame = draw_boxes_and_orientations(frame, tracked_objects, orientations)
        
        # Write frame to output video
        out.write(frame)
        
        # Display frame
        cv2.imshow('Tracking with Orientation', frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

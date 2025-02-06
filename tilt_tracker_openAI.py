import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

# Initialize YOLO model
model = YOLO('C:/Users/khale/LIA/train/best.pt')  # or use your custom trained model path
video_path = 'C:/Users/khale/LIA/data/3.mp4'
output_path = 'videos_output/output6.mp4'
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

def get_orientation(x1, y1, x2, y2):
    """
    Determines the orientation of the detected plank based on the aspect ratio.
    Returns:
        - "Correct Orientation" if height/width is within expected range.
        - "Incorrect Orientation" otherwise.
    """
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = height / width  # Compute aspect ratio

    if 2.0 <= aspect_ratio <= 4.0:
        return "Correct Orientation"
    else:
        return "Incorrect Orientation"

def process_frame(frame, model, tracker):
    """
    Process each frame for object detection and orientation tracking
    """
    results = model(frame, conf=0.4)  # Lower confidence threshold

    detections = []
    orientation_labels = {}

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        
        width = x2 - x1
        height = y2 - y1
        
        if confidence > 0.4 and width > 5 and height > 5:  # Adjusting minimum size for planks
            detections.append(([x1, y1, width, height], confidence, class_id))

            # Determine the orientation
            orientation = get_orientation(x1, y1, x2, y2)
            orientation_labels[(x1, y1, x2, y2)] = orientation  # Store result for drawing

    tracked_objects = tracker.update_tracks(detections, frame=frame)
    tracked_objects = [t for t in tracked_objects if t.is_confirmed() and t.time_since_update <= 1]

    return tracked_objects, orientation_labels

def draw_boxes_and_orientations(frame, tracked_objects, orientation_labels):
    """
    Draw bounding boxes, track IDs, and orientation on the frame
    """
    for track in tracked_objects:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Retrieve orientation label
        orientation = orientation_labels.get((x1, y1, x2, y2), "Unknown")
        
        # Display orientation label
        label = f"ID: {track_id}, {orientation}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
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

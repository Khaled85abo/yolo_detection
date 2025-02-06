import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

# Initialize YOLO model
model = YOLO('C:/Users/khale/LIA/train/best.pt')  # or use your custom trained model path
video_path = 'C:/Users/khale/LIA/data/3.mp4'
output_path = 'videos_output/output_annotated-3.mp4'
# Initialize DeepSORT tracker
# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=70,
    n_init=3,
    nms_max_overlap=1.0,
    max_iou_distance=0.7,
    max_cosine_distance=0.2,
    nn_budget=100,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True
)

def get_orientation(points):
    """
    Calculate the orientation angle of a bounding box
    Args:
        points: numpy array of box corner points
    Returns:
        angle: orientation angle in degrees
    """
    # Calculate rotated rectangle
    rect = cv2.minAreaRect(points)
    width = rect[1][0]
    height = rect[1][1]
    angle = rect[2]
    
    # Adjust angle based on width and height
    if width < height:
        angle = angle - 90
    
    # Normalize angle to be between -90 and 90 degrees
    if angle < -90:
        angle += 180
    elif angle > 90:
        angle -= 180
        
    return angle

def process_frame(frame, model, tracker):
    """
    Process each frame for object detection and orientation tracking
    """
    # Run YOLO detection
    results = model(frame)[0]
    
    detections = []
    orientations = []
    
    # Process each detected object
    for box in results.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get confidence and class
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        
        if confidence > 0.4:  # Confidence threshold
            # Format detection for DeepSORT
            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))
            
            # Get more points along the object boundary for better orientation detection
            # This helps in getting a more accurate minimum area rectangle
            num_points = 16
            x_coords = np.linspace(x1, x2, num_points//4)
            y_coords = np.linspace(y1, y2, num_points//4)
            
            points = []
            # Top edge
            for x in x_coords:
                points.append([int(x), y1])
            # Right edge
            for y in y_coords:
                points.append([x2, int(y)])
            # Bottom edge
            for x in x_coords[::-1]:
                points.append([int(x), y2])
            # Left edge
            for y in y_coords[::-1]:
                points.append([x1, int(y)])
                
            points = np.array(points, dtype=np.float32)
            angle = get_orientation(points)
            orientations.append((points, angle))
    
    # Update tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)
    
    return tracked_objects, orientations

def draw_boxes_and_orientations(frame, tracked_objects, orientations):
    """
    Draw bounding boxes, track IDs, and orientations on the frame
    """
    for track, (points, angle) in zip(tracked_objects, orientations):
        if not track.is_confirmed():
            continue
            
        # Get tracking info
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        # Draw rotated bounding box
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int_(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        
        # Draw track ID and orientation angle
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Angle: {angle:.1f}°", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw center point and orientation line
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)
        
        # Draw orientation line
        line_length = 50
        end_x = center_x + int(line_length * np.cos(np.radians(angle)))
        end_y = center_y + int(line_length * np.sin(np.radians(angle)))
        cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
    
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

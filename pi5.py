import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
# Add this at the top with other imports
from collections import defaultdict
from picamera2 import Picamera2
from libcamera import controls
# Add these new imports
from PIL import Image, ImageDraw, ImageFont
import board
import digitalio
import adafruit_rgb_display.st7789 as st7789
import time

# Add this as a global variable
orientation_memory = defaultdict(lambda: {"orientation": "unknown", "angle": 0, "aspect_ratio": 0})

# Initialize YOLO model
model = YOLO('C:/Users/khale/LIA/train_yolo11s/weights/best_yolo11s.pt')
video_path = 'C:/Users/khale/LIA/data/3.mp4'
output_path = 'videos_output/yolo11s/output_ori_1.mp4'
# Initialize DeepSORT tracker
# Initialize DeepSORT tracker
# tracker = DeepSort(
#     max_age=30,              # Maximum number of frames to keep dead tracks
#     n_init=2,                # Number of frames for track initialization
#     nms_max_overlap=0.7,     # NMS threshold for suppressing overlapping detections
#     max_iou_distance=0.7,    # Maximum IOU distance for matching
#     max_cosine_distance=0.3, # Maximum cosine distance for feature matching
#     nn_budget=100,           # Maximum size of the appearance descriptors gallery
#     embedder="mobilenet",    # Feature extractor
#     half=True,              # Use half precision for better speed
#     bgr=True,
#     embedder_gpu=True
# )

# DeepSORT tracker for faster processing
tracker = DeepSort(
    max_age=5,               # Reduced from 30 to prevent stale tracks
    n_init=1,                # Reduced from 2 to initialize tracks faster
    max_iou_distance=0.9,    # Increased from 0.7 to handle larger movements
    max_cosine_distance=0.4, # Increased from 0.3 for more flexible matching
    nn_budget=50,           # Reduced from 100 to save memory
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True
)

# Add display initialization
def init_display():
    # Configuration for CS and DC pins
    cs_pin = digitalio.DigitalInOut(board.CE0)
    dc_pin = digitalio.DigitalInOut(board.D25)
    reset_pin = digitalio.DigitalInOut(board.D24)
    
    # Configure SPI
    BAUDRATE = 24000000
    
    # Create the ST7789 display
    spi = board.SPI()
    display = st7789.ST7789(
        spi,
        rotation=90,  # Adjust rotation as needed
        width=240,
        height=320,
        x_offset=0,
        y_offset=0,
        cs=cs_pin,
        dc=dc_pin,
        rst=reset_pin,
        baudrate=BAUDRATE,
    )
    
    # Enable backlight
    backlight = digitalio.DigitalInOut(board.D22)
    backlight.switch_to_output()
    backlight.value = True
    
    return display

def update_display(display, incorrect_count=0, total_count=0):
    """
    Update the TFT display with warning and status information
    """
    # Create a blank image with mode 'RGB'
    image = Image.new('RGB', (display.width, display.height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not found
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw status information
    y_position = 10
    draw.text((10, y_position), "Plank Monitor", font=font, fill=(255, 255, 255))
    
    y_position += 40
    draw.text((10, y_position), f"Total: {total_count}", font=small_font, fill=(255, 255, 255))
    
    y_position += 30
    draw.text((10, y_position), f"Incorrect: {incorrect_count}", font=small_font, 
              fill=(255, 0, 0) if incorrect_count > 0 else (255, 255, 255))
    
    # Draw warning if there are incorrect planks
    if incorrect_count > 0:
        y_position += 50
        warning_text = "WARNING!"
        draw.text((10, y_position), warning_text, font=font, fill=(255, 0, 0))
        
        y_position += 40
        message = "Incorrect plank"
        message2 = "orientation detected!"
        draw.text((10, y_position), message, font=small_font, fill=(255, 255, 0))
        draw.text((10, y_position + 25), message2, font=small_font, fill=(255, 255, 0))
    
    # Display the image
    display.image(image)

def process_frame(frame, model, tracker):
    """
    Process each frame for object detection and orientation tracking
    """
    global orientation_memory
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Define the region of interest (middle of the frame)
    roi_x_start = int(frame_width * 0.35)
    roi_x_end = int(frame_width * 0.65)
    
    results = model(frame, conf=0.5)[0]
    
    detections = []
    detection_points = {}  # Use dictionary to maintain correspondence
    
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
        center_x = (x1 + x2) // 2
        
        # Reconstruct bbox in same format as detections
        bbox = [x1, y1, width, height]
        points = detection_points.get(tuple(bbox), np.array([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2]
        ], dtype=np.float32))
        
        # Reset orientation when outside ROI
        if center_x < roi_x_start or center_x > roi_x_end:
            orientation_memory[track_id] = {
                "orientation": "unknown",
                "angle": 0,
                "aspect_ratio": 0,
                "in_roi": False
            }
        else:
            # If object is in ROI, update its orientation
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
            memory.get("in_roi", False)  # Add in_roi status to orientations
        ))
    
    # Clean up memory for tracks that are no longer active
    active_track_ids = {track.track_id for track in tracked_objects}
    for track_id in list(orientation_memory.keys()):
        if track_id not in active_track_ids:
            del orientation_memory[track_id]
    
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
    else:
        orientation = "incorrect"
        
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
    # Initialize display
    display = init_display()
    
    # Initialize Pi camera with picamera2
    picam2 = Picamera2()
    
    # Configure camera
    config = picam2.create_video_configuration(
        # main={"size": (640, 480), "format": "RGB888"},
        main={"size": (320, 240), "format": "RGB888"},
        controls={"FrameDurationLimits": (33333, 33333)}  # ~30fps
    )
    picam2.configure(config)
    
    # Add frame rate management
    frame_interval = 1/10  # Process max 10 FPS
    last_frame_time = 0
    
    # Start camera
    picam2.start()
    
    # Get video properties for output
    # width, height = 640, 480  # Match with camera config
    width, height = 320, 240  # Match with camera config
    custom_fps = 10  # Adjust based on your Pi's performance
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, custom_fps, (width, height))
    
    try:
        incorrect_count = 0
        total_count = 0
        while True:
            current_time = time.time()
            
            # Skip frames if we're processing too quickly
            if current_time - last_frame_time < frame_interval:
                continue
                
            # Capture frame from picamera2
            frame = picam2.capture_array()
            last_frame_time = current_time
            
            # Calculate processing time
            process_start = time.time()
            
            # Convert frame from RGB to BGR for OpenCV compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process frame with ROI information
            tracked_objects, orientations, roi_bounds = process_frame(frame, model, tracker)
            
            # Count incorrect orientations in ROI
            incorrect_count = sum(1 for _, _, orientation, _, in_roi in orientations 
                                if in_roi and orientation == "incorrect")
            total_count = sum(1 for _, _, _, _, in_roi in orientations if in_roi)
            
            # Update display
            update_display(display, incorrect_count, total_count)
            
            # Draw results
            frame = draw_boxes_and_orientations(frame, tracked_objects, orientations, roi_bounds)
            
            # Write frame to output video
            out.write(frame)
            
            # Calculate and print FPS
            process_time = time.time() - process_start
            fps = 1 / (time.time() - last_frame_time)
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Tracking with Orientation', frame)
            
            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping capture...")
    finally:
        # Clean up
        picam2.stop()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
# Add this at the top with other imports
from collections import defaultdict
from picamera2 import Picamera2
import time
import os
import threading
from queue import Queue
import logging
import asyncio
import json
from enum import Enum, auto
from typing import Optional, Dict, List
# from mc.ESP32Controller import ESP32Controller, PlankStatus, WarningLevel

# Add this as a global variable
orientation_memory = defaultdict(lambda: {"orientation": "unknown", "angle": 0, "aspect_ratio": 0})

# ROI parameters
ROI_start = 0.40
ROI_end = 0.60

# aspect ratio
aspect_ratio_threshold = 0.60

# Initialize YOLO model
model = YOLO('/home/rise/enter/train_yolo11n/weights/best_yolo11n.pt')
output_path = 'videos_output'
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
    bgr=False,
    embedder_gpu=True
)

# Add these to the global variables section
STOP_THRESHOLD_FRAMES = 5  # Number of frames to consider as a stop
MOVEMENT_THRESHOLD = 10    # Pixel distance to consider as movement
stop_detection_memory = defaultdict(lambda: {
    "positions": [],
    "stop_frames": 0,
    "is_stopped": False
})

# class AsyncFrameProcessor:
#     def __init__(self, esp32_controller: ESP32Controller):
#         self.esp32 = esp32_controller
#         self.loop = asyncio.get_event_loop()

#     async def process_frame_async(self, frame, tracked_objects, orientations):
#         """Asynchronous processing of warnings and ESP32 communication"""
#         stopped_planks = []
#         overlapped_pairs = set()
#         incorrect_planks = []

#         # Process overlaps
#         for i, track1 in enumerate(tracked_objects):
#             for j, track2 in enumerate(tracked_objects[i+1:], i+1):
#                 if self._check_overlap(track1, track2):
#                     pair = tuple(sorted([track1.track_id, track2.track_id]))
#                     overlapped_pairs.add(pair)
#                     await self.esp32.send_warning(
#                         PlankStatus.OVERLAPPED,
#                         WarningLevel.ERROR,
#                         {"plank_ids": list(pair)}
#                     )

#         # Process orientations and stops
#         for track, (_, _, orientation, _, _, is_stopped) in zip(tracked_objects, orientations):
#             if is_stopped:
#                 stopped_planks.append(track.track_id)
#                 await self.esp32.send_warning(
#                     PlankStatus.STOPPED,
#                     WarningLevel.WARNING,
#                     {"plank_id": track.track_id}
#                 )

#             if orientation == "incorrect":
#                 incorrect_planks.append(track.track_id)
#                 await self.esp32.send_warning(
#                     PlankStatus.INCORRECT,
#                     WarningLevel.WARNING,
#                     {"plank_id": track.track_id}
#                 )

#     def _check_overlap(self, track1, track2):
#         # Existing overlap detection logic
#         # Returns True if overlap detected
#         pass

def process_frame(frame, model, tracker, server):
    """
    Process each frame for object detection and orientation tracking
    """
    global orientation_memory, stop_detection_memory
    
    process_start = time.time()

    height, width = frame.shape[:2]
    print(f"YOLO input shape: {width}x{height}")
    
    # YOLO detection on ROI frame
    yolo_start = time.time()

    # the trained model only detects the plank class
    results = model(frame, conf=0.5)[0]

    # but if the model is trained on multiple classes, you can specify the class
    # results = model(frame, conf=0.5, classes=[0])[0]
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
        # 20 represents the minimum size threshold (in pixels) for both the width and height of detected objects
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

    # Orientation and stop detection processing
    orient_start = time.time()
    final_orientations = []
    
    # Check for overlaps between all pairs of tracked objects
    for i, track1 in enumerate(tracked_objects):
        ltrb1 = track1.to_ltrb()
        
        # Check overlap with all other boxes
        for j, track2 in enumerate(tracked_objects[i+1:], i+1):
            ltrb2 = track2.to_ltrb()
            
            if _check_overlap_from_boxes(ltrb1, ltrb2):
                print(f"Warning: Overlap detected between planks {track1.track_id} and {track2.track_id}")
                # You might want to calculate the exact overlap percentages for logging
                box1 = [int(x) for x in ltrb1]
                box2 = [int(x) for x in ltrb2]
                x_left = max(box1[0], box2[0])
                y_top = max(box1[1], box2[1])
                x_right = min(box1[2], box2[2])
                y_bottom = min(box1[3], box2[3])
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                overlap_percent1 = (intersection_area / box1_area) * 100
                overlap_percent2 = (intersection_area / box2_area) * 100
                print(f"Overlap percentage: {overlap_percent1:.1f}% of plank {track1.track_id}, "
                      f"{overlap_percent2:.1f}% of plank {track2.track_id}")

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
                    print(f"Plank {track_id} has stopped in orientation: {orientation}")
            else:
                stop_memory["stop_frames"] = 0
                stop_memory["is_stopped"] = False
                print(f"Plank {track_id} has started moving again")

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
    
    print(
        f"[process_frame] Total: {total_time:.3f}s | "
        f"YOLO: {yolo_time:.3f}s | "
        f"Parse: {parse_time:.3f}s | "
        f"Track: {track_time:.3f}s | "
        f"Orientation: {orient_time:.3f}s"
    )

    # Update server with current status
    server.update_status(
        overlapped=[(track1.track_id, track2.track_id) 
                   for i, track1 in enumerate(tracked_objects)
                   for j, track2 in enumerate(tracked_objects[i+1:], i+1)
                   if _check_overlap_from_boxes(track1.to_ltrb(), track2.to_ltrb())],
        stopped=[track.track_id 
                for track, (_, _, _, _, _, is_stopped) in zip(tracked_objects, final_orientations) 
                if is_stopped],
        incorrect=[track.track_id 
                  for track, (_, _, orientation, _, _, _) in zip(tracked_objects, final_orientations) 
                  if orientation == "incorrect"]
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
    else:
        print("Orientation: incorrect")
        orientation = "incorrect" 


    return orientation, angle, aspect_ratio

def draw_boxes_and_orientations(frame, tracked_objects, orientations, roi_bounds):
    """
    Draw bounding boxes, track IDs, and orientations on the frame
    """
    used_positions = {}

    # First, draw all regular boxes and labels
    for track, (points, angle, orientation, aspect_ratio, _, is_stopped) in zip(tracked_objects, orientations):
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
        if is_stopped:
            label += " | STOPPED"
            # Add additional visual indicator for stopped objects
            cv2.drawContours(frame, [box], 0, (255, 255, 0), 4)  # Thicker yellow border for stopped objects
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, label_y - text_height - 4),
                     (x1 + text_width, label_y + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw orientation line
        # center_x = int((x1 + x2) / 2)
        # center_y = int((y1 + y2) / 2)
        # cv2.circle(frame, (center_x, center_y), 4, color, -1)

        # line_length = min(x2 - x1, y2 - y1) // 2
        # end_x = center_x + int(line_length * np.cos(np.radians(angle)))
        # end_y = center_y + int(line_length * np.sin(np.radians(angle)))
        # cv2.line(frame, (center_x, center_y), (end_x, end_y), color, 2)

    # Then check for overlaps and draw them
    for i, track1 in enumerate(tracked_objects):
        ltrb1 = track1.to_ltrb()
        box1 = [int(x) for x in ltrb1]
        
        for j, track2 in enumerate(tracked_objects[i+1:], i+1):
            ltrb2 = track2.to_ltrb()
            box2 = [int(x) for x in ltrb2]
            
            # Calculate intersection
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            
            if x_right > x_left and y_bottom > y_top:
                # Draw overlap area in semi-transparent red
                overlap_area = np.array([[x_left, y_top], [x_right, y_top],
                                       [x_right, y_bottom], [x_left, y_bottom]])
                overlay = frame.copy()
                cv2.fillPoly(overlay, [overlap_area], (0, 0, 255))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Draw warning text
                warning_text = f"Overlap: {track1.track_id}-{track2.track_id}"
                cv2.putText(frame, warning_text, (x_left, y_top - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame

def _check_overlap_from_boxes(ltrb1, ltrb2):
    """Helper function to check overlap using the same logic as in process_frame"""
    box1 = [int(x) for x in ltrb1]  # [x1, y1, x2, y2]
    box2 = [int(x) for x in ltrb2]  # [x1, y1, x2, y2]
    
    # Calculate intersection
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right > x_left and y_bottom > y_top:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        overlap_percent1 = (intersection_area / box1_area) * 100
        overlap_percent2 = (intersection_area / box2_area) * 100
        
        return overlap_percent1 > 20 or overlap_percent2 > 20
    return False

def main():
    print("Initializing camera...")
    picam2 = Picamera2()

    print("Configuring camera settings...")
    
    # Calculate padding to maintain aspect ratio
    target_width = 640
    target_height = 480
    custom_fps = 10


    # Ensure ROI width is even
    roi_width = int(target_width * (ROI_end - ROI_start))
    if roi_width % 2 != 0:
        roi_width += 1  # Make it even
    
    print(f"Full dimensions: {target_width}x{target_height}")
    print(f"ROI width: {roi_width}")
    
    config = picam2.create_video_configuration(
        main={"size": (target_width, target_height), "format": "RGB888"},
        controls={"FrameDurationLimits": (33333, 33333)}  # ~30fps
    )
    picam2.configure(config)

    picam2.set_controls({"AeEnable": True})

    print("Starting camera...")
    picam2.start()

    try:
        from stream_server_flask import StreamServer
        print("Starting stream server...")
        server = StreamServer()
        server.add_camera('camera1', frame_size=(roi_width, target_height))
        server_thread = server.run_threaded()
        print("Stream server started successfully")
    except Exception as e:
        print(f"Failed to start stream server: {e}")
        return

    global output_path

    try:
        from output_video import OutputVideo
        out_cls = OutputVideo(base_directory=output_path, fps=custom_fps, target_width=roi_width, target_height=target_height)
        out_cls.create_writer(name='camera1', subfolder='pi')

    except Exception as e:
        print(f"Failed to initialize OutputVideo: {str(e)}")
        return



    try:
        # Initialize ESP32 controller and frame processor
        # esp32 = ESP32Controller()
        # frame_processor = AsyncFrameProcessor(esp32)
        
        frame_count = 0
        while True:
            frame_count += 1
            print(f"\n--- Frame {frame_count} ---")
            
            loop_start = time.time()
            
            # Capture frame
            capture_start = time.time()
            frame = picam2.capture_array()
            capture_end = time.time()
            
            roi_x_start = int(target_width * ROI_start)
            roi_x_end = int(target_width * ROI_end)
            
            # First, maintain original aspect ratio by keeping the full height
            cropped_frame = frame[:, roi_x_start:roi_x_end]

            
            # Resize maintaining aspect ratio
            aspect_ratio = cropped_frame.shape[1] / cropped_frame.shape[0]
            if aspect_ratio > (target_width / target_height):
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
                vertical_padding = (target_height - new_height) // 2
                frame = cv2.resize(cropped_frame, (new_width, new_height))
                # Add padding
                frame = cv2.copyMakeBorder(frame, vertical_padding, vertical_padding, 
                                         0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
                horizontal_padding = (target_width - new_width) // 2
                frame = cv2.resize(cropped_frame, (new_width, new_height))
                # Add padding
                frame = cv2.copyMakeBorder(frame, 0, 0, horizontal_padding, 
                                         horizontal_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            # Add dimension check
            current_height, current_width = frame.shape[:2]
            if current_width != roi_width or current_height != target_height:
                print(f"Warning: Frame dimensions ({current_width}x{current_height}) "
                      f"don't match expected dimensions ({roi_width}x{target_height})")
            
            

            
            # Process frame
            process_start = time.time()
            tracked_objects, orientations, roi_bounds = process_frame(cropped_frame, model, tracker, server)
            process_end = time.time()
            
            # Process warnings asynchronously
            # asyncio.run(frame_processor.process_frame_async(
            #     cropped_frame, tracked_objects, orientations
            # ))
            
            # Print summary
            objects_in_roi = len(tracked_objects)
            print(f"Summary: {objects_in_roi} tracked objects")
            
            # Draw results
            draw_start = time.time()
            frame = draw_boxes_and_orientations(cropped_frame, tracked_objects, orientations, roi_bounds)
            draw_end = time.time()
            
            server.update_frame('camera1', frame)
            
            # Write original frame to video file
            out_write_start = time.time()
            out_cls.write_frame(frame, writer_key='camera1')
            out_write_end = time.time()
            
            loop_end = time.time()
            
            print(
                f"[main loop] Frame {frame_count} total: {(loop_end - loop_start):.3f}s | "
                f"capture: {(capture_end - capture_start):.3f}s | "
                f"process_frame: {(process_end - process_start):.3f}s | "
                f"draw: {(draw_end - draw_start):.3f}s | "
                f"write: {(out_write_end - out_write_start):.3f}s"
            )

    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        print("Cleaning up...")
        picam2.stop()
        out_cls.release(writer_key='camera1')
        # Give the server thread a moment to clean up
        time.sleep(0.5)
        print("Done!")

if __name__ == "__main__":
    main()
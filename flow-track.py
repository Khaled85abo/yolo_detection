import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("best.pt")  # Replace with your trained model path

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Open video feed
video_path = "conveyor_video.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())

        if confidence > 0.4:  # Confidence threshold
            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

    # Update tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    # Draw bounding boxes and track IDs
    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the video with tracking
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
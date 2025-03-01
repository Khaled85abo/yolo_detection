import cv2
import numpy as np

def draw_boxes_and_orientations(frame, tracked_objects, orientations, roi_bounds):
    """
    Draw bounding boxes, track IDs, and orientations on the frame.
    
    This function visualizes tracked objects with their orientation information:
    1. Draws rotated bounding boxes around detected objects with color coding:
       - Green: Correctly oriented objects
       - Red: Incorrectly oriented objects
       - Gray: Unknown orientation
    2. Displays text labels with track ID, orientation status, and aspect ratio
    3. Adds special visual indicators for stopped objects (thicker yellow border)
    4. Detects and highlights overlapping objects with semi-transparent red fill
       and warning text
    
    The function handles label positioning to avoid overlaps between labels.
    
    Args:
        frame: The image/video frame to draw on
        tracked_objects: List of tracking objects with position information
        orientations: List of tuples containing orientation data (points, angle, 
                     orientation, aspect_ratio, etc.)
        roi_bounds: Region of interest boundaries
        
    Returns:
        frame: The modified frame with all visual elements added
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


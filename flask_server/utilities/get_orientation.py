import cv2

# aspect ratio
aspect_ratio_threshold = 0.60

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


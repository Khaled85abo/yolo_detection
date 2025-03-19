import cv2

# aspect ratio
aspect_ratio_threshold = 0.60

def get_plank_orientation(points, width, height):
    """
    Determine plank orientation based on bounding box characteristics.
    
    This function analyzes a detected plank to determine if it's in the correct orientation:
    1. Calculates the aspect ratio (width/height) of the bounding box
    2. Computes the minimum area rectangle and its angle using OpenCV
    3. Normalizes the angle to be within a standard range
    4. Classifies the orientation as "correct" (vertical) or "incorrect" (horizontal)
       based on the aspect ratio threshold
    
    Args:
        points: Array of points representing the contour of the plank
        width: Width of the bounding box
        height: Height of the bounding box
        
    Returns:
        tuple: (orientation, angle, aspect_ratio)
            - orientation: String indicating "correct" or "incorrect" orientation
            - angle: Normalized angle of the plank in degrees
            - aspect_ratio: Width to height ratio of the plank
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
        # print("Orientation: correct")
        orientation = "correct"
    else:
        # print("Orientation: incorrect")
        orientation = "incorrect" 

    return orientation, angle, aspect_ratio



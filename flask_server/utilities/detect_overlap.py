# OVERLAP_THRESHOLD = 0.20

def check_overlap_from_boxes(ltrb1, ltrb2):
    """
    Determines if two bounding boxes overlap and calculates the percentage of overlap.
    
    This function uses the "Intersection over Area" approach to quantify overlap:
    1. Finds the intersection rectangle between two bounding boxes (if any)
    2. Calculates the area of this intersection
    3. Computes what percentage of each original box is covered by the intersection
    
    Args:
        ltrb1: First bounding box in format [left, top, right, bottom]
        ltrb2: Second bounding box in format [left, top, right, bottom]
        
    Returns:
        tuple: (is_overlapping, overlap_percent_of_box1, overlap_percent_of_box2)
            - is_overlapping: Boolean indicating if boxes overlap
            - overlap_percent_of_box1: Percentage of box1's area covered by the intersection
            - overlap_percent_of_box2: Percentage of box2's area covered by the intersection
    """
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
        
        return True, overlap_percent1, overlap_percent2
    return False, 0, 0


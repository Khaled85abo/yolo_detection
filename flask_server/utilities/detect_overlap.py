
# OVERLAP_THRESHOLD = 0.20

def _check_overlap_from_boxes(ltrb1, ltrb2):
    """Helper function to check overlap and return overlap percentages"""
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


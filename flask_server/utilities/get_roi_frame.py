
def get_roi_frame(frame_width, frame_height , roi):

    roi_width = int(frame_width * (roi["width_end"] - roi["width_start"]))
    roi_height = int(frame_height * (roi["height_end"] - roi["height_start"]))
    if roi_width % 2 != 0:
        roi_width += 1  # Make it even
    if roi_height % 2 != 0:
        roi_height += 1  # Make it even
    
    print(f"  Full dimensions: {frame_width}x{frame_height}")
    print(f"  ROI dimensions: {roi_width}x{roi_height}")
    
    return roi_width, roi_height





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






def main():
    try:
        from stream_server_flask import StreamServer
        print("Starting stream server...")
        server = StreamServer()
        server.add_camera('camera2')
        server_thread = server.run_threaded()
        print("Stream server started successfully")
    except Exception as e:
        print(f"Failed to start stream server: {e}")
        return

    global output_path
    print("Initializing camera...")
    picam2 = Picamera2()

    print("Configuring camera settings...")
    full_width, full_height = 640, 480
    ROI_start = 0.40
    ROI_end = 0.60
    
    # Ensure ROI width is even
    roi_width = int(full_width * (ROI_end - ROI_start))
    if roi_width % 2 != 0:
        roi_width += 1  # Make it even
    
    print(f"Full dimensions: {full_width}x{full_height}")
    print(f"ROI width: {roi_width}")
    
    config = picam2.create_video_configuration(
        main={"size": (full_width, full_height), "format": "RGB888"},
        controls={"FrameDurationLimits": (33333, 33333)}  # ~30fps
    )
    picam2.configure(config)

    picam2.set_controls({"AeEnable": True})

    print("Starting camera...")
    picam2.start()

    custom_fps = 10



    try:
        frame_count = 0
        while True:
            frame_count += 1
            print(f"\n--- Frame {frame_count} ---")
            
            loop_start = time.time()
            
            # Capture frame
            capture_start = time.time()
            frame = picam2.capture_array()
            capture_end = time.time()
            
            # Color conversion
            color_conv_start = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            color_conv_end = time.time()
            
            # Crop frame to ROI
            roi_x_start = int(full_width * ROI_start)
            roi_x_end = int(full_width * ROI_end)
            cropped_frame = frame[:, roi_x_start:roi_x_end]
            
            # Calculate padding to maintain aspect ratio
            target_width = 640
            target_height = 480
            
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
            if current_width != roi_width or current_height != full_height:
                print(f"Warning: Frame dimensions ({current_width}x{current_height}) "
                      f"don't match expected dimensions ({roi_width}x{full_height})")
            
            

            
            server.update_frame('camera2', frame)
            
            
            loop_end = time.time()
            
            print(
                f"[main loop] Frame {frame_count} total: {(loop_end - loop_start):.3f}s | "
                f"capture: {(capture_end - capture_start):.3f}s | "
                f"color_conv: {(color_conv_end - color_conv_start):.3f}s | "
            )

    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        print("Cleaning up...")
        picam2.stop()
        # Give the server thread a moment to clean up
        time.sleep(0.5)
        print("Done!")

if __name__ == "__main__":
    main()
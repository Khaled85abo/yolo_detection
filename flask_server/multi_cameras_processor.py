from ultralytics import YOLO
import time
import numpy as np
from camera_stream_cls import CameraStream
from utilities.process_frame import process_frame
from utilities.draw_boxes import draw_boxes_and_orientations
from utilities.detect_stop import create_tracker, STOP_THRESHOLD_FRAMES, MOVEMENT_THRESHOLD, stop_detection_memory

# Default frame dimensions
frame_width = 640
frame_height = 480
custom_fps = 10

# Initialize YOLO model
model = YOLO('/home/rise/enter/train_yolo11n/weights/best_yolo11n.pt')


def main():
    print("Initializing camera processing client...")
    active_cameras = 0
    # Import StreamServer to get camera registry and update frames
    from server_websocket import StreamServer
    from utilities.get_roi_frame import get_roi_frame
    server = StreamServer()
    
    # Start the server in a separate thread
    server_thread = server.run_threaded(host='0.0.0.0', port=5000)
    print(f"Server started on http://0.0.0.0:5000")
    time.sleep(2)  # Give the server time to start
    
    # Initialize camera objects
    camera_objects = {}
    camera_trackers = {}  # Dictionary to store trackers for each camera
    
    # Initialize video outputs
    try:
        from video_saving_cls import OutputVideo
        out_cls = {}
    except Exception as e:
        print(f"Failed to initialize OutputVideo: {str(e)}")
        return
    
    # Main processing loop
    try:
        frame_count = 0
        while True:
            # Check for changes in camera registry
            current_cameras = server.camera_configs
            
            # Handle new cameras
            for camera_id, camera_info in current_cameras.items():
                if camera_id not in camera_objects:
                    print(f"New camera detected: {camera_id}")
                    try:
                        # Initialize camera stream
                        camera_obj = CameraStream(camera_info)
                        print(f"Initialized camera stream: {camera_id} at {camera_info['url']}")
                        camera_obj.start()
                        validation_timeout = 5
                        validation_start = time.time()
                        camera_active = False
                        
                        while time.time() - validation_start < validation_timeout:
                            time.sleep(0.5)
                            test_frame = camera_obj.capture_array()
                            if test_frame is not None and test_frame.size > 0 and np.mean(test_frame) > 5.0:
                                camera_active = True
                                active_cameras += 1
                                print(f"Camera {camera_id} is active")
                                break
                            else:
                                print(f"Waiting for camera {camera_id} to become active...")
                        
                        if camera_active:
                            # Calculate ROI dimensions
                            roi = camera_info.get('roi', {
                                "width_start": 0.4,
                                "width_end": 0.6,
                                "height_start": 0.0,
                                "height_end": 1.0
                            })
                            
                            frame_width, frame_height = camera_info.get('frame_size', (640, 480))
                            roi_frame_width, roi_frame_height = get_roi_frame(frame_width, frame_height, roi)
                            
                            # Create a new tracker for this camera
                            camera_trackers[camera_id] = create_tracker()
                            
                            # Store camera with its configuration
                            camera_info["roi_frame_width"] = roi_frame_width
                            camera_info["roi_frame_height"] = roi_frame_height
                            camera_objects[camera_id] = {"camera": camera_obj, "config": camera_info}
                            
                            # Set up video recording if enabled
                            recording_config = camera_info.get("recording", {"enabled": True, "format": "mp4", "quality": "medium", "fps": 10})
                            
                            if recording_config.get("enabled", True):
                                # Set quality parameters based on configuration
                                quality = recording_config.get("quality", "medium")
                                if quality == "high":
                                    bitrate = 2000000  # 2 Mbps
                                elif quality == "medium":
                                    bitrate = 1000000  # 1 Mbps
                                else:  # low
                                    bitrate = 500000   # 500 Kbps
                                    
                                # Use the fps from the config file
                                out = OutputVideo(
                                    fps=recording_config.get("fps", 10), 
                                    frame_width=roi_frame_width, 
                                    frame_height=roi_frame_height,
                                    bitrate=bitrate
                                )
                                out.create_writer(name=camera_id, subfolder='pi')
                                out_cls[camera_id] = out
                        else:
                            camera_obj.stop()
                            server.camera_registry.unregister_camera(camera_id)
                            print(f"Camera {camera_id} failed validation - stopping camera and unregistering")
                    except Exception as e:
                        print(f"Failed to initialize camera {camera_id}: {e}")
            
            # Handle removed cameras
            for camera_id in list(camera_objects.keys()):
                if camera_id not in current_cameras:
                    print(f"Camera removed: {camera_id}")
                    # Stop the camera
                    camera_objects[camera_id]["camera"].stop()
                    # Release video writer if exists
                    if camera_id in out_cls:
                        out_cls[camera_id].release(writer_key=camera_id)
                        del out_cls[camera_id]
                    # Remove from our tracking
                    del camera_objects[camera_id]
                    if camera_id in camera_trackers:
                        del camera_trackers[camera_id]
                    active_cameras -= 1
            
            frame_count += 1
            print(f"\n--- Frame {frame_count} ---")
            
            if active_cameras == 0:
                print("No active cameras, waiting for cameras to be added...")
                time.sleep(1)
                continue
            
            loop_start = time.time()
            
            # Process each camera
            for camera_id, camera_data in camera_objects.items():
                camera = camera_data["camera"]
                config = camera_data["config"]
                
                # Skip if processing is disabled for this camera
                processing_config = config.get("processing", {"enabled": True})
                if not processing_config.get("enabled", True):
                    print(f"Processing disabled for camera {camera_id}, skipping")
                    continue
                
                # Capture frame
                capture_start = time.time()
                frame = camera.capture_array()
                capture_end = time.time()
                
                # Skip if frame is None or empty
                if frame is None or frame.size == 0:
                    print(f"No frame from camera {camera_id}, skipping")
                    continue
                
                # Check if frame is just a blank/black frame (common when connection fails)
                if np.mean(frame) < 5.0:  # Very dark frame is likely blank
                    print(f"Camera {camera_id} returned blank frame, skipping")
                    continue
                
                # Calculate ROI coordinates
                roi = config["roi"]
                frame_width, frame_height = config.get('frame_size', (640, 480))
                roi_x_start = int(frame_width * roi["width_start"])
                roi_x_end = int(frame_width * roi["width_end"])
                roi_y_start = int(frame_height * roi["height_start"])
                roi_y_end = int(frame_height * roi["height_end"])
                
                # Extract ROI
                cropped_frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                
                # Process frame with camera-specific settings
                process_start = time.time()
                conf_threshold = processing_config.get("confidence_threshold", 0.5)
                iou_threshold = processing_config.get("iou_threshold", 0.45)
                
                # Pass camera-specific parameters to process_frame
                tracked_objects, orientations, roi_bounds = process_frame(
                    cropped_frame, 
                    model, 
                    camera_trackers[camera_id],  # Use the camera-specific tracker
                    server
                )
                process_end = time.time()
                
                # Print summary
                objects_in_roi = len(tracked_objects)
                print(f"Camera {camera_id} summary: {objects_in_roi} tracked objects")
                
                # Draw results
                draw_start = time.time()
                processed_frame = draw_boxes_and_orientations(cropped_frame, tracked_objects, orientations, roi_bounds)
                draw_end = time.time()
                
                # Update server with processed frame
                server.update_frame(camera_id, processed_frame)
                
                # Write frame to video file if recording is enabled
                if camera_id in out_cls:
                    out_write_start = time.time()
                    out_cls[camera_id].write_frame(processed_frame, writer_key=camera_id)
                    out_write_end = time.time()
                    write_time = out_write_end - out_write_start
                else:
                    write_time = 0
                
                print(
                    f"[{camera_id}] Frame {frame_count} total: {(time.time() - capture_start):.3f}s | "
                    f"capture: {(capture_end - capture_start):.3f}s | "
                    f"process_frame: {(process_end - process_start):.3f}s | "
                    f"draw: {(draw_end - draw_start):.3f}s | "
                    f"write: {write_time:.3f}s"
                )
            
            # Calculate sleep time to maintain target FPS
            loop_end = time.time()
            processing_time = loop_end - loop_start
            target_frame_time = 1.0 / custom_fps
            
            # If processing was faster than target frame time, sleep to maintain FPS
            # If processing was slower, continue immediately to catch up
            sleep_time = max(0, target_frame_time - processing_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            print(f"Total frame processing time: {(loop_end - loop_start):.3f}s, Sleep time: {sleep_time:.3f}s")
            
    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        print("Cleaning up...")
        # Stop all cameras
        for camera_id, camera_data in camera_objects.items():
            camera_data["camera"].stop()
            if camera_id in out_cls:
                out_cls[camera_id].release(writer_key=camera_id)
        
        # Allow time for the server to close
        time.sleep(0.5)
        print("Done!")

if __name__ == "__main__":
    main() 
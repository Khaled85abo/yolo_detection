from ultralytics import YOLO
import cv2
import time
import threading
import numpy as np
import requests
from urllib.parse import urlparse

from utilities.process_frame import process_frame
from utilities.draw_boxes import draw_boxes_and_orientations
from utilities.detect_stop import tracker

# Default frame dimensions
frame_width = 640
frame_height = 480
custom_fps = 10

# Initialize YOLO model
model = YOLO('/home/rise/enter/train_yolo11n/weights/best_yolo11n.pt')

class CameraStream:
    def __init__(self, camera_config):
        self.config = camera_config
        self.url = camera_config['url']
        self.stream_type = camera_config.get('stream_type', 'mjpeg')
        self.frame = None
        self.running = False
        self.thread = None
        self.connection_config = camera_config.get('connection', {
            'timeout': 5,
            'retry_interval': 1,
            'max_retries': 3
        })
        self.frame_size = camera_config.get('frame_size', (640, 480))
        self.fps = camera_config.get('fps', 30)
        self.retry_count = 0
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _update_frame(self):
        if self.stream_type == 'mjpeg':
            self._update_mjpeg_stream()
        elif self.stream_type == 'jpeg':
            self._update_jpeg_stream()
        elif self.stream_type == 'rtsp':
            self._update_rtsp_stream()
        else:
            print(f"Unknown stream type: {self.stream_type}, defaulting to MJPEG")
            self._update_mjpeg_stream()
    
    def _update_mjpeg_stream(self):
        cap = cv2.VideoCapture(self.url)
        
        # Set buffer size based on FPS to avoid lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while self.running:
            success, frame = cap.read()
            if success:
                self.frame = frame
                self.retry_count = 0
                # Control frame rate
                time.sleep(1.0 / self.fps)
            else:
                print(f"Failed to read from MJPEG stream: {self.url}")
                self.retry_count += 1
                if self.retry_count > self.connection_config['max_retries']:
                    print(f"Max retries exceeded for {self.url}, reconnecting...")
                    cap.release()
                    time.sleep(self.connection_config['retry_interval'])
                    cap = cv2.VideoCapture(self.url)
                    self.retry_count = 0
                else:
                    time.sleep(self.connection_config['retry_interval'])
        
        cap.release()
    
    def _update_jpeg_stream(self):
        while self.running:
            try:
                response = requests.get(
                    self.url, 
                    stream=True, 
                    timeout=self.connection_config['timeout']
                )
                
                if response.status_code == 200:
                    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None and frame.size > 0:
                        self.frame = frame
                        self.retry_count = 0
                    else:
                        print(f"Received empty frame from {self.url}")
                        self.retry_count += 1
                else:
                    print(f"Failed to get image from {self.url}, status: {response.status_code}")
                    self.retry_count += 1
            except Exception as e:
                print(f"Error fetching from {self.url}: {e}")
                self.retry_count += 1
            
            # Handle retries
            if self.retry_count > self.connection_config['max_retries']:
                print(f"Max retries exceeded for {self.url}, waiting longer...")
                time.sleep(self.connection_config['retry_interval'] * 2)
                self.retry_count = 0
            
            # Control frame rate
            time.sleep(1.0 / self.fps)
    
    def _update_rtsp_stream(self):
        # RTSP handling is similar to MJPEG but might need different settings
        cap = cv2.VideoCapture(self.url)
        
        # RTSP-specific settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
        
        while self.running:
            success, frame = cap.read()
            if success:
                self.frame = frame
                self.retry_count = 0
            else:
                print(f"Failed to read from RTSP stream: {self.url}")
                self.retry_count += 1
                if self.retry_count > self.connection_config['max_retries']:
                    print(f"Max retries exceeded for {self.url}, reconnecting...")
                    cap.release()
                    time.sleep(self.connection_config['retry_interval'])
                    cap = cv2.VideoCapture(self.url)
                    self.retry_count = 0
                else:
                    time.sleep(self.connection_config['retry_interval'])
        
        cap.release()
                
    def capture_array(self):
        if self.frame is None:
            return None
        
        # Resize frame if needed
        if self.frame.shape[1] != self.frame_size[0] or self.frame.shape[0] != self.frame_size[1]:
            return cv2.resize(self.frame, self.frame_size)
            
        return self.frame.copy()

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
    
    # Get registered cameras from the server
    camera_registry = server.camera_registry.get_cameras()
    if not camera_registry:
        print("No cameras registered with the server")
        return
        
    print(f"Found {len(camera_registry)} cameras: {list(camera_registry.keys())}")
    
    # Initialize camera objects
    camera_objects = {}
    for camera_id, camera_info in camera_registry.items():
        # Initialize camera stream
        this_camera_active = False
        try:
            camera_obj = CameraStream(camera_info)
            print(f"Initialized camera stream: {camera_id} at {camera_info['url']}")
            camera_obj.start()
            validation_timeout = 3
            validation_start= time.time()
            while time.time() - validation_start < validation_timeout:
                time.sleep(0.4)
                test_frame = camera_obj.capture_array()
                if test_frame is not None:
                    this_camera_active = True
                    active_cameras += 1
                    print(f"Camera {camera_id} is active")
                    break
                else:
                    print(f"Camera {camera_id} is not active")
        except Exception as e:
            print(f"Failed to initialize camera {camera_id}: {e}")
            continue
            
        # Calculate ROI dimensions
        roi = camera_info.get('roi', {
            "width_start": 0.4,
            "width_end": 0.6,
            "height_start": 0.0,
            "height_end": 1.0
        })
        
        frame_width, frame_height = camera_info.get('frame_size', (640, 480))
        
        roi_frame_width, roi_frame_height = get_roi_frame(frame_width, frame_height, roi)
            
        print(f"Camera {camera_id}:")
        print(f"  Stream type: {camera_info.get('stream_type', 'mjpeg')}")
        print(f"  Target FPS: {camera_info.get('fps', 30)}")
        
        # Store camera with its configuration
        camera_info["roi_frame_width"] = roi_frame_width
        camera_info["roi_frame_height"] = roi_frame_height
        if this_camera_active:
            camera_objects[camera_id] = {"camera": camera_obj, "config": camera_info}
        else:
            print(f"Camera {camera_id} failed validation - stopping camera")
            camera_obj.stop()
    # Start all cameras
    for camera_id, camera_data in camera_objects.items():
        print(f"Starting camera {camera_id}...")
        camera_data["camera"].start()
    
    # Initialize video outputs
    try:
        from video_saving_class import OutputVideo
        out_cls = {}
        for camera_id, camera_data in camera_objects.items():
            config = camera_data["config"]
            recording_config = config.get("recording", {"enabled": True, "format": "mp4", "quality": "medium", "fps": 10})
            
            if not recording_config.get("enabled", True):
                print(f"Recording disabled for camera {camera_id}")
                continue
                
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
                frame_width=config["roi_frame_width"], 
                frame_height=config["roi_frame_height"],
                bitrate=bitrate
            )
            out.create_writer(name=camera_id, subfolder='pi')
            out_cls[camera_id] = out
    except Exception as e:
        print(f"Failed to initialize OutputVideo: {str(e)}")
        return
    
    # Main processing loop
    try:
        frame_count = 0
        while True and active_cameras > 0:
            frame_count += 1
            print(f"\n--- Frame {frame_count} ---")
            
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
                    tracker, 
                    server,
                    # conf_threshold=conf_threshold,
                    # iou_threshold=iou_threshold,
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
            
            loop_end = time.time()
            print(f"Total frame processing time: {(loop_end - loop_start):.3f}s")
            
            # Control overall processing rate
            elapsed = loop_end - loop_start
            target_time = 1.0 / custom_fps
            if elapsed < target_time:
                time.sleep(target_time - elapsed)

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
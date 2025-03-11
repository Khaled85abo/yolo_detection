# Remove eventlet from the top - it's interfering with camera processing
# import eventlet
# eventlet.monkey_patch()


from ultralytics import YOLO
from picamera2 import Picamera2
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

class RemoteCamera:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.running = False
        self.thread = None
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _update_frame(self):
        # For MJPEG streams
        if 'mjpg' in self.url or 'mjpeg' in self.url:
            cap = cv2.VideoCapture(self.url)
            while self.running:
                success, frame = cap.read()
                if success:
                    self.frame = frame
                else:
                    print(f"Failed to read from stream: {self.url}")
                    time.sleep(1)  # Wait before retry
            cap.release()
        else:
            # For single JPEG streams (like ESP32-CAM)
            while self.running:
                try:
                    response = requests.get(self.url, stream=True, timeout=5)
                    if response.status_code == 200:
                        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                        self.frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    else:
                        print(f"Failed to get image from {self.url}, status: {response.status_code}")
                except Exception as e:
                    print(f"Error fetching from {self.url}: {e}")
                time.sleep(0.1)  # Adjust based on desired frame rate
                
    def capture_array(self):
        return self.frame if self.frame is not None else np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

def main():
    print("Initializing camera processing client...")
    
    # Import StreamServer to get camera registry and update frames
    from server_websocket import StreamServer
    server = StreamServer()
    
    # Get registered cameras from the server
    camera_registry = server.camera_registry.get_cameras()
    if not camera_registry:
        print("No cameras registered with the server")
        return
        
    print(f"Found {len(camera_registry)} cameras: {list(camera_registry.keys())}")
    
    # Initialize camera objects based on type
    camera_objects = {}
    for camera_id, camera_info in camera_registry.items():
        camera_type = camera_info.get('type', 'remote')
        
        if camera_type == 'picamera' and camera_info.get('local', True):
            # Initialize local Pi camera
            try:
                picam = Picamera2(camera_num=camera_info.get('camera_num', 0))
                
                config = picam.create_video_configuration(
                    main={"size": (frame_width, frame_height), "format": "RGB888"},
                    controls={"FrameDurationLimits": (33333, 33333)}  # ~30fps
                )
                picam.configure(config)
                picam.set_controls({"AeEnable": True})
                camera_obj = picam
                print(f"Initialized local Pi camera: {camera_id}")
            except Exception as e:
                print(f"Failed to initialize Pi camera {camera_id}: {e}")
                continue
            
        elif camera_type == 'remote' or not camera_info.get('local', True):
            # Initialize remote camera
            try:
                camera_obj = RemoteCamera(camera_info['url'])
                print(f"Initialized remote camera: {camera_id} at {camera_info['url']}")
            except Exception as e:
                print(f"Failed to initialize remote camera {camera_id}: {e}")
                continue
        else:
            print(f"Unknown camera type: {camera_type}")
            continue
            
        # Calculate ROI dimensions
        roi = camera_info.get('roi', {
            "width_start": 0.4,
            "width_end": 0.6,
            "height_start": 0.0,
            "height_end": 1.0
        })
        
        roi_width = int(frame_width * (roi["width_end"] - roi["width_start"]))
        roi_height = int(frame_height * (roi["height_end"] - roi["height_start"]))
        if roi_width % 2 != 0:
            roi_width += 1  # Make it even
        if roi_height % 2 != 0:
            roi_height += 1  # Make it even
            
        print(f"Camera {camera_id}:")
        print(f"  Full dimensions: {frame_width}x{frame_height}")
        print(f"  ROI dimensions: {roi_width}x{roi_height}")
        
        # Store camera with its configuration
        camera_info["roi_width"] = roi_width
        camera_info["roi_height"] = roi_height
        camera_objects[camera_id] = {"camera": camera_obj, "config": camera_info}
    
    # Start all cameras
    for camera_id, camera_data in camera_objects.items():
        print(f"Starting camera {camera_id}...")
        camera_data["camera"].start()
    
    # Initialize video outputs
    try:
        from video_saving_cls import OutputVideo
        out_cls = {}
        for camera_id, camera_data in camera_objects.items():
            config = camera_data["config"]
            out = OutputVideo(fps=custom_fps, frame_width=config["roi_width"], frame_height=config["roi_height"])
            out.create_writer(name=camera_id, subfolder='pi')
            out_cls[camera_id] = out
    except Exception as e:
        print(f"Failed to initialize OutputVideo: {str(e)}")
        return
    
    try:
        frame_count = 0
        while True:
            frame_count += 1
            print(f"\n--- Frame {frame_count} ---")
            
            loop_start = time.time()
            
            # Process each camera
            for camera_id, camera_data in camera_objects.items():
                camera = camera_data["camera"]
                config = camera_data["config"]
                
                # Capture frame
                capture_start = time.time()
                frame = camera.capture_array()
                capture_end = time.time()
                
                # Skip if frame is None or empty
                if frame is None or frame.size == 0:
                    print(f"No frame from camera {camera_id}, skipping")
                    continue
                
                # Ensure frame has correct dimensions
                if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
                    frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Calculate ROI coordinates
                roi = config["roi"]
                roi_x_start = int(frame_width * roi["width_start"])
                roi_x_end = int(frame_width * roi["width_end"])
                roi_y_start = int(frame_height * roi["height_start"])
                roi_y_end = int(frame_height * roi["height_end"])
                
                # Extract ROI
                cropped_frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                
                # Process frame
                process_start = time.time()
                tracked_objects, orientations, roi_bounds = process_frame(cropped_frame, model, tracker, server)
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
                
                # Write frame to video file
                out_write_start = time.time()
                out_cls[camera_id].write_frame(processed_frame, writer_key=camera_id)
                out_write_end = time.time()
                
                print(
                    f"[{camera_id}] Frame {frame_count} total: {(time.time() - capture_start):.3f}s | "
                    f"capture: {(capture_end - capture_start):.3f}s | "
                    f"process_frame: {(process_end - process_start):.3f}s | "
                    f"draw: {(draw_end - draw_start):.3f}s | "
                    f"write: {(out_write_end - out_write_start):.3f}s"
                )
            
            loop_end = time.time()
            print(f"Total frame processing time: {(loop_end - loop_start):.3f}s")

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
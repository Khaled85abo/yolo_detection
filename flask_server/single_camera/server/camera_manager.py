import cv2
import threading
from typing import Dict, Tuple, Optional
import logging

class CameraManager:
    def __init__(self):
        self.cameras: Dict[str, Optional[cv2.Mat]] = {}
        self.frame_locks: Dict[str, threading.Lock] = {}
        self.frame_sizes: Dict[str, Tuple[int, int]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_camera(self, camera_id: str, frame_size: Tuple[int, int] = (640, 480)):
        """Add a new camera feed"""
        self.cameras[camera_id] = None
        self.frame_locks[camera_id] = threading.Lock()
        self.frame_sizes[camera_id] = frame_size
        self.logger.info(f"Added camera: {camera_id} with frame size {frame_size}")
    
    def update_frame(self, camera_id: str, frame):
        """Update the frame for a specific camera"""
        if camera_id not in self.frame_locks:
            self.logger.error(f"Attempted to update non-existent camera: {camera_id}")
            return False
            
        with self.frame_locks[camera_id]:
            # Resize frame for streaming to reduce bandwidth
            stream_frame = cv2.resize(frame, self.frame_sizes[camera_id])
            self.cameras[camera_id] = stream_frame.copy()
        return True
    
    def get_frame(self, camera_id: str):
        """Get the current frame for a camera"""
        if camera_id not in self.cameras:
            return None
            
        with self.frame_locks[camera_id]:
            if self.cameras[camera_id] is None:
                return None
            return self.cameras[camera_id].copy()
    
    def generate_frames(self, camera_id: str):
        """Generator function for streaming frames"""
        frame_skip = 0  # Counter to potentially skip frames
        
        while True:
            # Skip some frames to reduce CPU/network load
            frame_skip += 1
            if frame_skip % 2 != 0:  # Process every other frame
                threading.Event().wait(0.01)
                continue
                
            frame = self.get_frame(camera_id)
            if frame is not None:
                try:
                    # Reduce JPEG quality for faster streaming
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    self.logger.error(f"Error encoding frame: {e}")
                    continue
                    
            # Slightly longer delay to prevent overwhelming the network
            threading.Event().wait(0.05)
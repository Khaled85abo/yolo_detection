"""Base class for camera processing clients"""
import time
import threading
import requests
import json

class CameraClient:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.client_id = f"client_{int(time.time())}"
        self.cameras = {}  # Cameras this client is processing
        self.running = False
        self.process_thread = None
        
    def register_with_server(self):
        """Register this client with the server"""
        try:
            response = requests.post(
                f"{self.server_url}/api/clients",
                json={"id": self.client_id, "capabilities": self.get_capabilities()}
            )
            if response.status_code == 200 or response.status_code == 201:
                print(f"Client registered with server: {self.client_id}")
                return True
            else:
                print(f"Failed to register client: {response.text}")
                return False
        except Exception as e:
            print(f"Error registering client: {e}")
            return False
            
    def get_capabilities(self):
        """Return capabilities of this client - override in subclasses"""
        return {
            "can_process_local": True,
            "can_process_remote": True,
            "supported_camera_types": ["picamera", "remote"]
        }
        
    def get_assigned_cameras(self):
        """Get cameras assigned to this client from server"""
        try:
            response = requests.get(
                f"{self.server_url}/api/clients/{self.client_id}/cameras"
            )
            if response.status_code == 200:
                assigned_cameras = response.json()
                print(f"Assigned cameras: {list(assigned_cameras.keys())}")
                return assigned_cameras
            else:
                print(f"Failed to get assigned cameras: {response.text}")
                return {}
        except Exception as e:
            print(f"Error getting assigned cameras: {e}")
            return {}
            
    def initialize_cameras(self, camera_configs):
        """Initialize cameras based on configuration"""
        # Override in subclasses
        pass
        
    def process_frame(self, camera_id, frame):
        """Process a frame from a camera"""
        # Override in subclasses
        pass
        
    def start(self):
        """Start processing cameras"""
        if self.running:
            return
            
        self.running = True
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def stop(self):
        """Stop processing cameras"""
        self.running = False
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
            
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            # Process each camera
            for camera_id, camera in self.cameras.items():
                try:
                    # Get frame
                    frame = camera.get_frame()
                    
                    # Process frame
                    processed_frame = self.process_frame(camera_id, frame)
                    
                    # Send processed frame to server
                    self.send_frame_to_server(camera_id, processed_frame)
                except Exception as e:
                    print(f"Error processing camera {camera_id}: {e}")
                    
            # Sleep to control processing rate
            time.sleep(0.1)
            
    def send_frame_to_server(self, camera_id, frame):
        """Send processed frame to server"""
        # Implementation depends on how you want to send frames
        # Could use HTTP, WebSockets, or direct function call if server is local
        pass

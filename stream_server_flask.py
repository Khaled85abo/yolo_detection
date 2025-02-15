# def main():
#     # Import the StreamServer at the top of main to avoid circular imports
#     from stream_server import StreamServer
    
#     // ... existing code ...
    
#     # Initialize and start the streaming server
#     print("Starting streaming server...")
#     server = StreamServer()
#     server.add_camera('camera1')  # Add our camera
#     server.run_threaded()
    
#     try:
#         frame_count = 0
#         while True:
#             // ... existing code ...
            
#             # Update the frame for streaming
#             server.update_frame('camera1', frame)
            
#             # Write original frame to video file
#             out_write_start = time.time()
#             out.write(frame)
#             out_write_end = time.time()
            
#             // ... existing code ...


# server1 = StreamServer()
# server2 = StreamServer()
# print(server1 is server2)  # Will print True


from flask import Flask, Response
import cv2
import threading
from queue import Queue
import logging


# Singleton class to ensure only one instance of StreamServer is created, even if multiple instances are created in different files.
class StreamServer:
    _instance = None
    _lock = threading.Lock()

# The combination of __new__ and _instance ensures we only ever create one StreamServer instance.
    def __new__(cls):
        with cls._lock:                      # 1. Thread-safe lock
            if cls._instance is None:        # 2. Check if instance exists
                # 3. Create new instance if none exists
                cls._instance = super(StreamServer, cls).__new__(cls)
                
                # 4. Initialize the instance attributes
                cls._instance.app = Flask(__name__)
                cls._instance.cameras = {}
                cls._instance.frame_locks = {}
                
                # 5. Set up Flask routes
                cls._instance.app.route('/')(cls._instance.index)
                cls._instance.app.route('/video_feed/<camera_id>')(cls._instance.video_feed)
                
            return cls._instance             # 6. Return existing or new instance

    def __init__(self):
        # Skip initialization if already done
        pass
    
    def add_camera(self, camera_id):
        """Add a new camera feed"""
        self.cameras[camera_id] = None
        self.frame_locks[camera_id] = threading.Lock()
    
    def update_frame(self, camera_id, frame):
        """Update the frame for a specific camera"""
        with self.frame_locks[camera_id]:
            # Resize frame for streaming to reduce bandwidth
            stream_frame = cv2.resize(frame, (640, 480))
            self.cameras[camera_id] = stream_frame.copy()
    
    def generate_frames(self, camera_id):
        """Generator function for streaming frames"""
        while True:
            with self.frame_locks[camera_id]:
                frame = self.cameras[camera_id]
                if frame is not None:
                    try:
                        # Reduce JPEG quality for faster streaming
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                        _, buffer = cv2.imencode('.jpg', frame, encode_param)
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    except Exception as e:
                        print(f"Error in generate_frames: {e}")
                        continue
            threading.Event().wait(0.03)  # Small delay to prevent overwhelming the network
    
    def video_feed(self, camera_id):
        """Route for streaming video"""
        return Response(self.generate_frames(camera_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def index(self):
        """Route for main page"""
        camera_feeds = ""
        for camera_id in self.cameras.keys():
            camera_feeds += f'<div><h2>Camera {camera_id}</h2>'
            camera_feeds += f'<img src="/video_feed/{camera_id}" width="640" height="480" /></div>'
        
        return f"""
        <html>
            <body>
                <h1>Plank Detection Streams</h1>
                {camera_feeds}
                <p>Status: Streaming</p>
            </body>
        </html>
        """
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the Flask server"""
        # Disable debug mode and reduce logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        self.app.run(host=host, port=port, debug=False, use_reloader=False)
    
    def run_threaded(self, host='0.0.0.0', port=5000):
        """Run the Flask server in a separate thread"""
        server_thread = threading.Thread(
            target=self.run,
            args=(host, port)
        )
        server_thread.daemon = True
        server_thread.start()
        return server_thread
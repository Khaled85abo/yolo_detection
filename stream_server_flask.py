from flask import Flask, Response
import cv2
import threading
from queue import Queue
import logging

class StreamServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.cameras = {}  # Dictionary to store camera feeds
        self.frame_locks = {}  # Dictionary to store frame locks
        
        # Route definitions
        self.app.route('/')(self.index)
        self.app.route('/video_feed/<camera_id>')(self.video_feed)
    
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
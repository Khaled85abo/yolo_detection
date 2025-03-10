from flask import Flask, Response
import cv2
from picamera2 import Picamera2
import threading
import time
import argparse
import numpy as np

app = Flask(__name__)

class PiCameraStream:
    def __init__(self, camera_num=0, width=640, height=480, fps=30):
        self.camera_num = camera_num
        self.width = width
        self.height = height
        self.fps = fps
        self.picam = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        
    def start(self):
        if self.running:
            return
            
        self.running = True
        self.picam = Picamera2(camera_num=self.camera_num)
        
        # Configure camera with frame rate limits
        # For maximum speed, set a very high frame rate limit
        config = self.picam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps}
        )
        self.picam.configure(config)
        self.picam.set_controls({"AeEnable": True})
        self.picam.start()
        
        # Start capture thread
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        print(f"Camera {self.camera_num} started at {self.width}x{self.height} @ {self.fps}fps")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.picam:
            self.picam.stop()
            self.picam = None
            
    def _capture_loop(self):
        while self.running:
            try:
                frame = self.picam.capture_array()
                with self.lock:
                    self.frame = frame
            except Exception as e:
                print(f"Error capturing frame: {e}")
            # No sleep needed - camera hardware handles timing
            
    def get_frame(self):
        with self.lock:
            if self.frame is None:
                # Return a blank frame if no frame is available
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return self.frame.copy()

# Global camera instance
camera = None

def generate_frames():
    while True:
        # Get frame from camera
        frame = camera.get_frame()
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_bytes = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # No sleep needed - we want to stream as fast as frames are available

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Pi Camera Stream</title>
      </head>
      <body>
        <h1>Pi Camera Stream</h1>
        <img src="/stream" width="640" height="480" />
      </body>
    </html>
    """

@app.route('/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pi Camera Stream Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--camera', type=int, default=0, help='Camera number to use')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--fps', type=int, default=0, help='Target FPS (0 for maximum)')
    args = parser.parse_args()
    
    # For maximum speed, use fps=0 or a very high number
    fps = args.fps if args.fps > 0 else 120  # Use 120 as a high default when 0 is specified
    
    # Initialize camera
    camera = PiCameraStream(
        camera_num=args.camera,
        width=args.width,
        height=args.height,
        fps=fps
    )
    camera.start()
    
    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    finally:
        camera.stop()
        print("Camera stopped")

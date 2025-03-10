"""
Pi Camera Streaming Server
Streams MJPEG frames via Flask at minimal latency.

Usage:
  python pi_stream_server.py [--port 8000] [--camera 0] [--width 640] [--height 480] [--fps 0] [--quality 80]

Key points for minimal latency:
  - No sleeping in the capture loop or the Flask generator.
  - If --fps=0, we set a high FrameRate (e.g., 120) so hardware runs as fast as possible.
  - JPEG quality can be tuned for performance vs. image size.
  - The consumer of this stream must also read frames quickly to avoid buffering.
"""

import argparse
import threading

import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)

class PiCameraStreamer:
    """
    Continuously captures frames from a Picamera2 device in a background thread.
    Frames are stored in self.frame and read via get_frame().
    """
    def __init__(self, camera_num=0, width=640, height=480, fps=20):
        """
        :param camera_num: Index of the camera (if multiple cameras).
        :param width: Desired capture width.
        :param height: Desired capture height.
        :param fps: Target FPS. If set very high, camera will run at hardware limit.
        """
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
        """Initializes the camera and starts the background capture thread."""
        if self.running:
            return
        self.running = True

        self.picam = Picamera2(camera_num=self.camera_num)
        # For maximum speed, set a high frame rate limit if fps is large
        config = self.picam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps}
        )
        self.picam.configure(config)
        self.picam.set_controls({"AeEnable": True})  # Auto-exposure enabled
        self.picam.start()

        # Start the thread that continuously captures frames
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"Camera {self.camera_num} started at {self.width}x{self.height} @ {self.fps}fps")

    def stop(self):
        """Stops the background thread and the camera."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.picam:
            self.picam.stop()
            self.picam = None
        print(f"Camera {self.camera_num} stopped.")

    def _capture_loop(self):
        """
        Continuously capture frames from the camera into self.frame.
        No additional sleep needed — the camera hardware and FPS setting manage timing.
        """
        while self.running:
            try:
                new_frame = self.picam.capture_array()
                with self.lock:
                    self.frame = new_frame
            except Exception as e:
                print(f"Error capturing frame from camera {self.camera_num}: {e}")

    def get_frame(self):
        """
        Safely returns the latest frame captured, or a blank frame if none available.
        :return: A copy of the current frame (numpy array).
        """
        with self.lock:
            if self.frame is None:
                # Return a blank frame if no frame has been captured yet
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return self.frame.copy()

# Global camera instance, set up when __main__ is run
camera = None
jpeg_quality = 80

def generate_frames():
    """Generator function that encodes frames as JPEG and yields them in an MJPEG stream."""
    while True:
        frame = camera.get_frame()

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        frame_bytes = buffer.tobytes()

        # Yield frame in multipart/x-mixed-replace format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # No sleep here: we push frames as quickly as they are available.

@app.route('/')
def index():
    """Simple HTML page to view the camera stream."""
    return f"""
    <html>
      <head>
        <title>Pi Camera Stream</title>
      </head>
      <body>
        <h1>Pi Camera Stream</h1>
        <img src="/stream" width="{camera.width}" height="{camera.height}" />
      </body>
    </html>
    """

@app.route('/stream')
def stream():
    """Route that provides the MJPEG stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    """Parses arguments and starts the Flask app with the Pi camera streamer."""
    parser = argparse.ArgumentParser(description='Pi Camera Stream Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (if multiple connected)')
    parser.add_argument('--width', type=int, default=640, help='Frame capture width')
    parser.add_argument('--height', type=int, default=480, help='Frame capture height')
    parser.add_argument('--fps', type=int, default=20, help='Target FPS (0 for maximum camera speed)')
    parser.add_argument('--quality', type=int, default=80, help='JPEG quality (0-100, higher=better)')
    args = parser.parse_args()

    # Set a high default FPS if user chooses 0 for "no limit"
    fps = args.fps if args.fps > 0 else 120

    # Make the global references
    global camera
    global jpeg_quality
    jpeg_quality = max(1, min(args.quality, 100))  # Clamp between 1 and 100

    # Initialize and start the camera streamer
    camera = PiCameraStreamer(
        camera_num=args.camera,
        width=args.width,
        height=args.height,
        fps=fps
    )
    camera.start()

    # Run the Flask app
    try:
        print(f"Starting Flask server on port {args.port}...")
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    finally:
        camera.stop()

if __name__ == '__main__':
    main()

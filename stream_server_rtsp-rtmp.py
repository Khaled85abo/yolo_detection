# The advantages of this implementation:
# Supports both RTSP and RTMP protocols
# Very low latency (typically <100ms)
# Compatible with many video players (VLC, ffplay, etc.)
# Supports multiple camera feeds
# Hardware acceleration support through FFmpeg
# Better bandwidth efficiency with H.264 encoding
# More reliable for 24/7 streaming
# You can view the streams using:
# VLC: Open Network Stream -> rtsp://localhost:8554/camera1
# FFplay: ffplay rtsp://localhost:8554/camera1
# OBS Studio: Add Media Source -> rtmp://localhost:1935/camera1
# The streams can also be accessed over the network by replacing localhost with your server's IP address.
# Note: The implementation uses TCP for RTSP transport which is more reliable but slightly higher latency. You can modify the FFmpeg parameters to use UDP if needed.

# To use this server, you'll need to:
# Install MediaMTX (formerly rtsp-simple-server):
# Download from: https://github.com/bluenviron/mediamtx/releases
# Add it to your system PATH

# # Ubuntu/Debian
# sudo apt-get install ffmpeg

# # macOS
# brew install ffmpeg

# # Windows
# # Download from https://ffmpeg.org/download.html


# def main():
#     # Initialize the RTSP server
#     server = RTSPServer()
#     server.add_camera('camera1')
#     server.run_threaded()

#     # In your main loop:
#     while True:
#         # ... get your frame ...
#         server.update_frame('camera1', frame)

import subprocess
import threading
import cv2
import numpy as np
import time
import logging
from typing import Dict, Optional
import os

class RTSPServer:
    def __init__(self):
        self.logger = logging.getLogger("RTSPServer")
        self.logger.setLevel(logging.INFO)
        
        self.cameras: Dict[str, Dict] = {}
        self.running = False
        self.mediamtx_process: Optional[subprocess.Popen] = None
        
        # Default RTSP/RTMP ports
        self.rtsp_port = 8554
        self.rtmp_port = 1935
        
        # Configure MediaMTX
        self._create_mediamtx_config()

    def _create_mediamtx_config(self):
        """Create MediaMTX configuration file"""
        config = f"""
paths:
  all:
    readUser: admin
    readPass: admin
    publishUser: admin
    publishPass: admin

rtspAddress: 0.0.0.0:{self.rtsp_port}
rtmpAddress: 0.0.0.0:{self.rtmp_port}
rtsp: yes
rtmp: yes
metrics: no
pprof: no
"""
        with open("mediamtx.yml", "w") as f:
            f.write(config)

    def add_camera(self, camera_id: str):
        """Add a new camera feed"""
        if camera_id in self.cameras:
            return
        
        self.cameras[camera_id] = {
            'frame': None,
            'lock': threading.Lock(),
            'process': None,
            'running': False
        }
        
        # Start streaming process for this camera
        self._start_camera_stream(camera_id)
        self.logger.info(f"Added camera: {camera_id}")

    def update_frame(self, camera_id: str, frame: np.ndarray):
        """Update the frame for a specific camera"""
        if camera_id in self.cameras:
            with self.cameras[camera_id]['lock']:
                self.cameras[camera_id]['frame'] = frame.copy()

    def _start_camera_stream(self, camera_id: str):
        """Start FFmpeg process for streaming camera feed"""
        if camera_id not in self.cameras:
            return
        
        camera_data = self.cameras[camera_id]
        camera_data['running'] = True
        
        def stream_thread():
            # FFmpeg command for RTSP/RTMP streaming
            rtsp_url = f"rtsp://localhost:{self.rtsp_port}/{camera_id}"
            rtmp_url = f"rtmp://localhost:{self.rtmp_port}/{camera_id}"
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', '640x480',  # Input resolution
                '-r', '30',  # Input framerate
                '-i', '-',  # Input from pipe
                # RTSP output
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-f', 'rtsp',
                '-rtsp_transport', 'tcp',
                rtsp_url,
                # RTMP output
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-f', 'flv',
                rtmp_url
            ]
            
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                camera_data['process'] = process
                
                while camera_data['running']:
                    with camera_data['lock']:
                        frame = camera_data['frame']
                    
                    if frame is not None:
                        # Resize frame to match FFmpeg input size
                        frame = cv2.resize(frame, (640, 480))
                        try:
                            process.stdin.write(frame.tobytes())
                        except (BrokenPipeError, IOError):
                            break
                    
                    time.sleep(1/30)  # Limit to 30 FPS
                
            except Exception as e:
                self.logger.error(f"Streaming error for camera {camera_id}: {e}")
            finally:
                if process:
                    process.terminate()
                    process.wait()
                camera_data['process'] = None
        
        thread = threading.Thread(target=stream_thread, daemon=True)
        thread.start()

    def _stop_camera_stream(self, camera_id: str):
        """Stop streaming for a specific camera"""
        if camera_id in self.cameras:
            self.cameras[camera_id]['running'] = False
            process = self.cameras[camera_id]['process']
            if process:
                process.terminate()
                process.wait()
            self.cameras[camera_id]['process'] = None

    def start(self):
        """Start the RTSP/RTMP server"""
        if self.running:
            return
        
        try:
            # Start MediaMTX server
            self.mediamtx_process = subprocess.Popen(
                ['mediamtx', 'mediamtx.yml'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.running = True
            self.logger.info(f"RTSP server started on port {self.rtsp_port}")
            self.logger.info(f"RTMP server started on port {self.rtmp_port}")
            
            # Start all camera streams
            for camera_id in self.cameras:
                self._start_camera_stream(camera_id)
                
        except FileNotFoundError:
            self.logger.error("MediaMTX not found. Please install it first.")
            raise

    def stop(self):
        """Stop the RTSP/RTMP server"""
        if not self.running:
            return
        
        # Stop all camera streams
        for camera_id in self.cameras:
            self._stop_camera_stream(camera_id)
        
        # Stop MediaMTX server
        if self.mediamtx_process:
            self.mediamtx_process.terminate()
            self.mediamtx_process.wait()
            self.mediamtx_process = None
        
        self.running = False
        self.logger.info("RTSP/RTMP server stopped")

    def run_threaded(self):
        """Run the RTSP/RTMP server in a separate thread"""
        server_thread = threading.Thread(target=self.start, daemon=True)
        server_thread.start()
        return server_thread

    def get_stream_urls(self, camera_id: str) -> Dict[str, str]:
        """Get streaming URLs for a camera"""
        return {
            'rtsp': f'rtsp://localhost:{self.rtsp_port}/{camera_id}',
            'rtmp': f'rtmp://localhost:{self.rtmp_port}/{camera_id}'
        }
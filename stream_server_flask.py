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


from flask import Flask, Response, jsonify, request
import cv2
import threading
from queue import Queue
import logging
from typing import Dict, List


class PlankStatus:
    def __init__(self):
        self.overlap: List[tuple] = []
        self.stop: List[int] = []
        self.incorrect: List[int] = []
        self.conveyor_status: str = "running"

class StreamServer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StreamServer, cls).__new__(cls)
                cls._instance.app = Flask(__name__)
                cls._instance.cameras = {}
                cls._instance.frame_locks = {}
                cls._instance.plank_status = PlankStatus()
                
                # Add new routes for status and control
                cls._instance.app.route('/')(cls._instance.index)
                cls._instance.app.route('/video_feed/<camera_id>')(cls._instance.video_feed)
                cls._instance.app.route('/api/status', methods=['GET'])(cls._instance.get_status)
                cls._instance.app.route('/api/control', methods=['POST'])(cls._instance.control_conveyor)
                
            return cls._instance

    def __init__(self):
        # Skip initialization if already done
        pass
    
    def add_camera(self, camera_id, frame_size=(640, 480)):
        """Add a new camera feed"""
        self.cameras[camera_id] = None
        self.frame_locks[camera_id] = threading.Lock()
        self.frame_size = frame_size

    def update_frame(self, camera_id, frame):
        """Update the frame for a specific camera"""
        with self.frame_locks[camera_id]:
            # Resize frame for streaming to reduce bandwidth
            stream_frame = cv2.resize(frame, self.frame_size)
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
    
    def update_status(self, overlapped=None, stopped=None, incorrect=None):
        """Update the status of planks"""
        if overlapped is not None:
            self.plank_status.overlap = overlapped
        if stopped is not None:
            self.plank_status.stop = stopped
        if incorrect is not None:
            self.plank_status.incorrect = incorrect

    def get_status(self):
        """API endpoint to get current status"""
        return jsonify({
            'overlap': False if len(self.plank_status.overlap) == 0 else True,
            'stop': False if len(self.plank_status.stop) == 0 else True,
            'incorrect': False if len(self.plank_status.incorrect) == 0 else True,
            'conveyor_status': self.plank_status.conveyor_status
        })

    def control_conveyor(self):
        """API endpoint to control conveyor"""
        action = request.json.get('action')
        if action in ['stop', 'start']:
            self.plank_status.conveyor_status = 'stopped' if action == 'stop' else 'running'
            # Here you would add actual conveyor control logic
            return jsonify({'status': 'success', 'conveyor_status': self.plank_status.conveyor_status})
        return jsonify({'status': 'error', 'message': 'Invalid action'})

    def index(self):
        """Enhanced main page with status and controls"""
        camera_feeds = ""
        for camera_id in self.cameras.keys():
            camera_feeds += f'<div><h2>Camera {camera_id}</h2>'
            camera_feeds += f'<img src="/video_feed/{camera_id}" width="{self.frame_size[0]}" height="{self.frame_size[1]}" /></div>'
        
        return f"""
        <html>
            <head>
                <style>
                    .status {{ padding: 10px; margin: 10px; border: 1px solid #ccc; }}
                    .warning {{ color: red; }}
                    .controls {{ margin: 20px; }}
                    button {{ padding: 10px; margin: 5px; }}
                </style>
            </head>
            <body>
                <h1>Plank Detection System</h1>
                {camera_feeds}
                <div class="status" id="status">
                    Loading status...
                </div>
                <div class="controls">
                    <button onclick="controlConveyor('stop')" style="background-color: #ff4444;">Stop Conveyor</button>
                    <button onclick="controlConveyor('start')" style="background-color: #44ff44;">Start Conveyor</button>
                </div>
                <div id="warnings">
                    <div id="stopped" class="warning inactive">
                        <h3>Stopped Plank</h3>
                        <button onclick="acknowledge('stopped')">Acknowledge</button>
                    </div>
                </div>
                <script>
                    function updateStatus() {{
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(data => {{
                                console.log(data);
                                document.getElementById('status').innerHTML = `
                                    <p>Conveyor Status: <strong>${{data.conveyor_status}}</strong></p>
                                    <p>Overlapped Planks: <strong>${{data.overlap ? 'Yes' : 'No'}}</strong></p>
                                    <p>Stopped Planks: <strong>${{data.stop ? 'Yes' : 'No'}}</strong></p>
                                    <p>Incorrect Planks: <strong>${{data.incorrect ? 'Yes' : 'No'}}</strong></p>
                                `;
                            }});
                    }}

                    function controlConveyor(action) {{
                        fetch('/api/control', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ action: action }})
                        }})
                        .then(response => response.json())
                        .then(data => updateStatus());
                    }}

                    // Update status every second
                    setInterval(updateStatus, 1000);
                </script>
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
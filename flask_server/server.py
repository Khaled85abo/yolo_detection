from flask import Flask, Response, jsonify, request, render_template
from flask_socketio import SocketIO, emit
import cv2
import threading
from queue import Queue
import logging
from typing import Dict, List
# import eventlet
import os

# Initialize eventlet for better async performance
# eventlet.monkey_patch()

class PlankStatus:
    def __init__(self):
        self.overlap: List[tuple] = []
        self.stop: List[int] = []
        self.incorrect: List[int] = []
        self.conveyor_stop: bool = False

class StreamServer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StreamServer, cls).__new__(cls)
                
                # Set template folder relative to the server.py file
                template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
                static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
                
                cls._instance.app = Flask(__name__, 
                                        template_folder=template_dir,
                                        static_folder=static_dir)
                
                cls._instance.socketio = SocketIO(
                    cls._instance.app, 
                    cors_allowed_origins="*", 
                    async_mode='eventlet',
                    ping_timeout=30,  # Increased timeout
                    ping_interval=15,
                    logger=True,      # Enable logging for debugging
                    engineio_logger=True
                )
                
                cls._instance.cameras = {}
                cls._instance.frame_locks = {}
                cls._instance.plank_status = PlankStatus()
                cls._instance.frame_size = (640, 480)  # Default frame size
                
                # Add routes
                cls._instance.app.route('/')(cls._instance.index)
                cls._instance.app.route('/video_feed/<camera_id>')(cls._instance.video_feed)
                cls._instance.app.route('/api/status', methods=['GET'])(cls._instance.get_status)
                cls._instance.app.route('/api/control', methods=['POST'])(cls._instance.control_conveyor)
                
                # Add socketio routes
                cls._instance.socketio.on_event('connect', cls._instance.on_connect)
                cls._instance.socketio.on_event('disconnect', cls._instance.on_disconnect)
                cls._instance.socketio.on_event('control_conveyor', cls._instance.on_control_conveyor)
                cls._instance.socketio.on_event('update_conveyor_stop', cls._instance.update_conveyor_stop)
                cls._instance.socketio.on_event('status_update', cls._instance.emit_status)

            return cls._instance

    def __init__(self):
        # Skip initialization if already done
        pass

    def on_connect(self):
        """Handle client connection"""
        print("Client connected")
        # Emit initial status immediately after connection
        self.emit_status()

    def on_disconnect(self):
        print("Client disconnected")

    def on_control_conveyor(self, data):
        """Send conveyor status to ESP32"""
        print("on_control_conveyor received:", data)
        if isinstance(data, dict) and 'state' in data:
            state = data['state']
            self.plank_status.conveyor_stop = state
            self.socketio.emit('update_conveyor_stop', {'state': state})
            # Also emit updated status to all clients
            self.emit_status()

    def emit_status(self):
        """Emit status to all connected clients"""
        try:
            # Create data directly instead of using jsonify response
            status_data = {
                'overlap': False if len(self.plank_status.overlap) == 0 else True,
                'stop': False if len(self.plank_status.stop) == 0 else True,
                'incorrect': False if len(self.plank_status.incorrect) == 0 else True,
                'conveyor_stop': self.plank_status.conveyor_stop
            }
            print(f"Emitting status update: {status_data}")
            self.socketio.emit('status_update', status_data)
        except Exception as e:
            print(f"Error emitting status: {e}")

    def update_conveyor_stop(self, data):
        """Update conveyor status from ESP32"""
        try:
            print("Conveyor update received:", data)
            if isinstance(data, dict) and 'state' in data:
                self.plank_status.conveyor_stop = data['state']
                self.emit_status()  # Emit the updated status to all clients
                return True
            else:
                print("Invalid data format for update_conveyor_stop")
                return False
        except Exception as e:
            print(f"Error in update_conveyor_stop: {e}")
            return False

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
        """Generator function for streaming frames with better resource management"""
        frame_skip = 0  # Counter to potentially skip frames
        
        while True:
            # Skip some frames to reduce CPU/network load if needed
            frame_skip += 1
            if frame_skip % 2 != 0:  # Process every other frame
                threading.Event().wait(0.01)
                continue
                
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
                        
            # Slightly longer delay to prevent overwhelming the network
            threading.Event().wait(0.05)
    
    def video_feed(self, camera_id):
        """Route for streaming video"""
        return Response(self.generate_frames(camera_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def update_status(self, overlapped=None, stopped=None, incorrect=None):
        """Update the status of planks"""
        print("Updating status:", overlapped, stopped, incorrect)
        
        # Store previous state to check for changes
        prev_overlap = self.plank_status.overlap.copy()
        prev_stop = self.plank_status.stop.copy()
        prev_incorrect = self.plank_status.incorrect.copy()
        
        # Update the state
        if overlapped is not None:
            self.plank_status.overlap = overlapped
        if stopped is not None:
            self.plank_status.stop = stopped
        if incorrect is not None:
            self.plank_status.incorrect = incorrect
        
        # Only emit if state has changed
        if (prev_overlap != self.plank_status.overlap or 
            prev_stop != self.plank_status.stop or 
            prev_incorrect != self.plank_status.incorrect):
            print("State changed, emitting status update")
            self.emit_status()

    def get_status(self):
        """API endpoint to get current status"""
        print("Getting status:", self.plank_status.overlap, self.plank_status.stop, self.plank_status.incorrect)
        return jsonify({
            'overlap': False if len(self.plank_status.overlap) == 0 else True,
            'stop': False if len(self.plank_status.stop) == 0 else True,
            'incorrect': False if len(self.plank_status.incorrect) == 0 else True,
            'conveyor_stop': self.plank_status.conveyor_stop
        })

    def control_conveyor(self):
        """API endpoint to control conveyor"""
        state = request.json.get('state')
        self.plank_status.conveyor_stop = state
        # Here you would add actual conveyor control logic
        return jsonify({'status': 'success', 'conveyor_stop': self.plank_status.conveyor_stop})

    def index(self):
        """Render the template-based index page"""
        # Pass camera IDs and frame size to the template
        camera_ids = list(self.cameras.keys())
        
        return render_template('index.html', 
                              camera_ids=camera_ids,
                              frame_size=self.frame_size)
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the Flask server"""
        # Disable debug mode and reduce logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        self.socketio.run(self.app, host=host, port=port, debug=False, use_reloader=False, log_output=False)
    
    def run_threaded(self, host='0.0.0.0', port=5000):
        """Run the Flask server with SocketIO in a separate thread with better configuration"""
        server_thread = threading.Thread(
            target=self.socketio.run,
            args=(self.app,),
            kwargs={
                'host': host, 
                'port': port, 
                'debug': False, 
                'use_reloader': False,
                'log_output': False  # Reduce logging to console
            }
        )
        server_thread.daemon = True
        server_thread.start()
        return server_thread

# # Create a singleton instance
# stream_server = StreamServer()

# # Function to get the server instance (for importing from other modules)
# def get_server():
#     return stream_server

# # If running directly, start the server
# if __name__ == "__main__":
#     server = get_server()
#     server.add_camera('camera1')  # Add a default camera
#     server.run(host='0.0.0.0', port=5000)

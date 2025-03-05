from flask import Flask, Response, jsonify, request, render_template
import cv2
import threading
from queue import Queue
import logging
from typing import Dict, List
import os
from flask import Blueprint, current_app
from simple_websocket import Server as WebSocketServer, ConnectionClosed

# Import camera configuration
try:
    from camera_config import CAMERAS as DEFAULT_CAMERAS
except ImportError:
    DEFAULT_CAMERAS = {}

class PlankStatus:
    def __init__(self):
        self.overlap: List[tuple] = []
        self.stop: List[int] = []
        self.incorrect: List[int] = []
        self.conveyor_stop: bool = False

class CameraRegistry:
    def __init__(self):
        self.cameras = {}  # Dictionary of registered cameras
        self.lock = threading.Lock()
        
    def register_camera(self, camera_id, camera_info):
        """Register a camera with the server"""
        with self.lock:
            self.cameras[camera_id] = camera_info
            print(f"Camera registered: {camera_id} - {camera_info['url']}")
            return True
            
    def unregister_camera(self, camera_id):
        """Remove a camera from the registry"""
        with self.lock:
            if camera_id in self.cameras:
                del self.cameras[camera_id]
                print(f"Camera unregistered: {camera_id}")
                return True
            return False
            
    def get_cameras(self):
        """Get a list of all registered cameras"""
        with self.lock:
            return {k: v for k, v in self.cameras.items()}

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
                
                cls._instance.cameras = {}
                cls._instance.frame_locks = {}
                cls._instance.plank_status = PlankStatus()
                cls._instance.frame_size = (640, 480)  # Default frame size
                cls._instance.clients = []  # List to store websocket connections
                cls._instance.camera_registry = CameraRegistry()  # Add camera registry
                
                # Add rules configuration
                cls._instance.rules = {
                    'overlap': 'ignore',
                    'stop': 'ignore',
                    'incorrect': 'ignore'
                }
                
                # Add routes
                cls._instance.app.route('/')(cls._instance.index)
                cls._instance.app.route('/video_feed/<camera_id>')(cls._instance.video_feed)
                cls._instance.app.route('/api/status', methods=['GET'])(cls._instance.get_status)
                cls._instance.app.route('/api/control_conveyor', methods=['POST'])(cls._instance.control_conveyor)
                cls._instance.app.route('/api/cameras', methods=['GET'])(cls._instance.get_cameras)
                cls._instance.app.route('/api/cameras', methods=['POST'])(cls._instance.register_camera)
                cls._instance.app.route('/api/cameras/<camera_id>', methods=['DELETE'])(cls._instance.delete_camera)
                
                # Add websocket route
                cls._instance.app.route('/ws', websocket=True)(cls._instance.websocket_route)
                
                # Load default cameras from configuration
                for camera_id, camera_info in DEFAULT_CAMERAS.items():
                    cls._instance.camera_registry.register_camera(camera_id, camera_info)
                    frame_size = camera_info.get('frame_size', cls._instance.frame_size)
                    cls._instance.add_camera(camera_id, frame_size)

            return cls._instance

    def __init__(self):
        # Skip initialization if already done
        if hasattr(self, 'app'):
            return
            
        # The initialization is already done in __new__
        # No need to duplicate the route and rule definitions here
        pass

    def websocket_route(self):
        """Handle websocket connections"""
        try:
            # Explicitly print debugging info
            print("WebSocket connection attempt received")
            ws = WebSocketServer(request.environ)
            print("WebSocket connection established successfully")
            self.clients.append(ws)
            print(f"Client connected. Total clients: {len(self.clients)}")
            
            # Emit initial status immediately after connection
            self.emit_status()
            
            try:
                while True:
                    message = ws.receive()
                    if message:
                        print(f"Received message: {message}")
                        self.handle_websocket_message(ws, message)
            except ConnectionClosed:
                print("Client disconnected")
            finally:
                if ws in self.clients:
                    self.clients.remove(ws)
                    print(f"Client removed. Remaining clients: {len(self.clients)}")
            return ''
        except Exception as e:
            print(f"Error in websocket_route: {str(e)}")
            import traceback
            traceback.print_exc()
            return str(e), 500  # Return error with 500 status code
        
    def handle_websocket_message(self, ws, message):
        """Process incoming websocket messages"""
        import json
        try:
            data = json.loads(message)
            event_type = data.get('event')
            payload = data.get('data', {})
            
            # command from the UI to control the conveyor
            if event_type == 'control_conveyor':
                self.on_control_conveyor(payload)
            # command from the ESP32 to update the conveyor status
            elif event_type == 'update_conveyor_stop':
                self.update_conveyor_stop(payload)
            # Handle rule updates from the client
            elif event_type == 'update_rules':
                self.update_rules(payload)
            # Handle ping messages to keep connection alive
            elif event_type == 'ping':
                pass  # Just acknowledge receipt, no action needed
            # Add camera registration via websocket
            elif event_type == 'register_camera':
                camera_id = payload.get('id')
                if camera_id and self.camera_registry.register_camera(camera_id, payload):
                    # Also add to streaming system
                    if 'frame_size' in payload:
                        frame_size = tuple(payload['frame_size'])
                    else:
                        frame_size = self.frame_size
                    self.add_camera(camera_id, frame_size)
                    # Notify all clients about the new camera
                    self.send_to_all_clients({
                        'event': 'camera_added', 
                        'data': {'id': camera_id, 'info': payload}
                    })
            # Add camera unregistration via websocket
            elif event_type == 'unregister_camera':
                camera_id = payload.get('id')
                if camera_id and self.camera_registry.unregister_camera(camera_id):
                    # Also remove from streaming system
                    with self.frame_locks.get(camera_id, threading.Lock()):
                        if camera_id in self.cameras:
                            del self.cameras[camera_id]
                        if camera_id in self.frame_locks:
                            del self.frame_locks[camera_id]
                    # Notify all clients about the removed camera
                    self.send_to_all_clients({
                        'event': 'camera_removed', 
                        'data': {'id': camera_id}
                    })
            # Add more event handlers as needed
            
        except json.JSONDecodeError:
            print(f"Invalid JSON message: {message}")
        except Exception as e:
            print(f"Error handling websocket message: {e}")

    def on_control_conveyor(self, data):
        """Send conveyor status to ESP32"""
        print("on_control_conveyor received:", data)
        if isinstance(data, dict) and 'state' in data:
            state = data['state']
            self.send_to_all_clients({'event': 'control_conveyor', 'data': {'state': state}})

    def control_conveyor(self):
        """API endpoint to control conveyor
        This is the command from the UI to control the conveyor
        The command is sent to the ESP32 and the ESP32 updates the conveyor status
        """
        state = request.json.get('state')
        # Emit the command to the ESP32
        self.send_to_all_clients({'event': 'control_conveyor', 'data': {'state': state}})
        # Here you would add actual conveyor control logic
        return jsonify({"message": "Conveyor control command received"})

    def emit_status(self):
        """Emit status to all connected clients"""
        try:
            # Create data directly
            status_data = {
                'overlap': False if len(self.plank_status.overlap) == 0 else True,
                'stop': False if len(self.plank_status.stop) == 0 else True,
                'incorrect': False if len(self.plank_status.incorrect) == 0 else True,
                'conveyor_stop': self.plank_status.conveyor_stop
            }
            print(f"Emitting status update: {status_data}")
            self.send_to_all_clients({'event': 'status_update', 'data': status_data})
        except Exception as e:
            print(f"Error emitting status: {e}")
            
    def send_to_all_clients(self, data):
        """Send data to all connected websocket clients"""
        import json
        message = json.dumps(data)
        disconnected_clients = []
        
        for client in self.clients:
            try:
                client.send(message)
            except ConnectionClosed:
                disconnected_clients.append(client)
            except Exception as e:
                print(f"Error sending to client: {e}")
                disconnected_clients.append(client)
                
        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.clients:
                self.clients.remove(client)

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
            # Apply rules after status update
            self.apply_rules()

    def get_status(self):
        """API endpoint to get current status"""
        print("Getting status:", self.plank_status.overlap, self.plank_status.stop, self.plank_status.incorrect)
        return jsonify({
            'overlap': False if len(self.plank_status.overlap) == 0 else True,
            'stop': False if len(self.plank_status.stop) == 0 else True,
            'incorrect': False if len(self.plank_status.incorrect) == 0 else True,
            'conveyor_stop': self.plank_status.conveyor_stop
        })

    def update_rules(self, rules_data):
        """Update the rules configuration"""
        try:
            print(f"Updating rules: {rules_data}")
            # Validate the rules data
            if not isinstance(rules_data, dict):
                print("Invalid rules data format")
                return False
                
            # Update the rules
            for key in ['overlap', 'stop', 'incorrect']:
                if key in rules_data and rules_data[key] in ['ignore', 'stop_conveyor', 'alert']:
                    self.rules[key] = rules_data[key]
            
            print(f"Rules updated: {self.rules}")
            return True
        except Exception as e:
            print(f"Error updating rules: {e}")
            return False
            
    def apply_rules(self):
        """Apply the configured rules based on current status"""
        try:
            status = {
                'overlap': len(self.plank_status.overlap) > 0,
                'stop': len(self.plank_status.stop) > 0,
                'incorrect': len(self.plank_status.incorrect) > 0
            }
            
            # Check each condition and apply the configured action
            actions_taken = []
            
            for condition, is_active in status.items():
                if is_active and self.rules[condition] == 'stop_conveyor' and not self.plank_status.conveyor_stop:
                    # Stop the conveyor
                    self.plank_status.conveyor_stop = True
                    self.send_to_all_clients({
                        'event': 'control_conveyor', 
                        'data': {'state': True}
                    })
                    actions_taken.append(f"Stopped conveyor due to {condition}")
            
            if actions_taken:
                print(f"Rules applied: {', '.join(actions_taken)}")
                self.emit_status()  # Update all clients with new status
                
            return True
        except Exception as e:
            print(f"Error applying rules: {e}")
            return False

    def index(self):
        """Render the template-based index page"""
        # Pass camera registry to the template
        camera_ids = list(self.cameras.keys())
        registered_cameras = self.camera_registry.get_cameras()
        
        return render_template('index.html', 
                              camera_ids=camera_ids,
                              cameras=registered_cameras,
                              frame_size=self.frame_size)
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the Flask server"""
        # Disable debug mode and reduce logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        
        # Add CORS headers for WebSocket compatibility
        @self.app.after_request
        def add_cors_headers(response):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
            response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
            return response
        
        self.app.run(host=host, port=port, debug=False, use_reloader=False)
    
    def run_threaded(self, host='0.0.0.0', port=5000):
        """Run the Flask server with SocketIO in a separate thread with better configuration"""
        server_thread = threading.Thread(
            target=self.app.run,
            args=(host, port),
            kwargs={
                'debug': False, 
                'use_reloader': False,
            }
        )
        server_thread.daemon = True
        server_thread.start()
        return server_thread

    def get_cameras(self):
        """API endpoint to get all registered cameras"""
        return jsonify(self.camera_registry.get_cameras())
        
    def register_camera(self):
        """API endpoint to register a new camera"""
        data = request.json
        if not data or 'id' not in data or 'url' not in data:
            return jsonify({"error": "Missing required fields"}), 400
            
        camera_id = data['id']
        
        # Register with the camera registry
        success = self.camera_registry.register_camera(camera_id, data)
        if success:
            # Also add to the streaming system if frame_size is provided
            if 'frame_size' in data:
                frame_size = tuple(data['frame_size'])
            else:
                frame_size = self.frame_size
                
            self.add_camera(camera_id, frame_size)
            return jsonify({"message": f"Camera {camera_id} registered successfully"}), 201
        else:
            return jsonify({"error": "Failed to register camera"}), 500
            
    def delete_camera(self, camera_id):
        """API endpoint to unregister a camera"""
        success = self.camera_registry.unregister_camera(camera_id)
        if success:
            # Also remove from streaming system
            with self.frame_locks.get(camera_id, threading.Lock()):
                if camera_id in self.cameras:
                    del self.cameras[camera_id]
                if camera_id in self.frame_locks:
                    del self.frame_locks[camera_id]
            return jsonify({"message": f"Camera {camera_id} unregistered successfully"})
        else:
            return jsonify({"error": "Camera not found"}), 404

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

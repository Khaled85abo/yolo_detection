from flask import Flask, Response, jsonify, request, render_template
import cv2
import threading
from queue import Queue
import logging
from typing import Dict, List
import os

from flask import Blueprint, current_app
from simple_websocket import Server as WebSocketServer, ConnectionClosed



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
                
                cls._instance.cameras = {}
                cls._instance.frame_locks = {}
                cls._instance.plank_status = PlankStatus()
                cls._instance.frame_sizes = {}  # Default frame size
                cls._instance.clients = []  # List to store websocket connections
                
                # Add rules configuration
                cls._instance.rules = {
                    'overlap': 'ignore',
                    'stop': 'ignore',
                    'incorrect': 'ignore'
                }
                cls._instance.rules_options = ['ignore', 'stop_conveyor', 'alert']
                
                # Add routes
                cls._instance.app.route('/')(cls._instance.index)
                cls._instance.app.route('/video_feed/<camera_id>')(cls._instance.video_feed)
                cls._instance.app.route('/api/status', methods=['GET'])(cls._instance.get_status)
                cls._instance.app.route('/api/control_conveyor', methods=['POST'])(cls._instance.control_conveyor)
                cls._instance.app.route('/api/rules', methods=['POST'])(cls._instance.update_rules)
                
                # Add websocket route
                # cls._instance.app.route('/ws')(cls._instance.websocket_route)
                cls._instance.app.route('/ws', websocket=True)(cls._instance.websocket_route)

            return cls._instance

    def __init__(self):
        # Skip initialization if already done
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
            self.emit_rules()
            
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

    # def take_action(self):
    #     """
    #     This method will be called after status update to take action according to the rules
    #     This is the command from the ESP32 to take action according to the rules
    #     actions can be to show a warning, stop the conveyor or do nothing
    #     """
    #     print("take_action received:")


    #     if isinstance(data, dict) and 'action' in data:
    #         action = data['action']
    #         self.send_to_all_clients({'event': 'take_action', 'data': {'action': action}})

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

    def emit_rules(self):
        """Emit the rules to all connected clients"""
        try:
            self.send_to_all_clients({'event': 'rules_update', 'data': {'rules': self.rules, 'rules_options': self.rules_options}})
        except Exception as e:
            print(f"Error emitting rules: {e}")


            
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
        self.frame_sizes[camera_id] = frame_size

    def update_frame(self, camera_id, frame):
        """Update the frame for a specific camera"""
        with self.frame_locks[camera_id]:
            # Resize frame for streaming to reduce bandwidth
            stream_frame = cv2.resize(frame, self.frame_sizes[camera_id])
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
        # Track which detection types need rule application
        active_detections = []
        
        # Handle overlap detection
        if overlapped is not None:
            sorted_overlapped = sorted(overlapped)
            if sorted_overlapped != self.plank_status.overlap:
                print("Overlap status changed:", self.plank_status.overlap, "->", sorted_overlapped)
                self.plank_status.overlap = sorted_overlapped
                # Only add to active detections if there's an active overlap
                if len(sorted_overlapped) > 0:
                    active_detections.append('overlap')
        
        # Handle stop detection
        if stopped is not None:
            sorted_stopped = sorted(stopped)
            if sorted_stopped != self.plank_status.stop:
                print("Stop status changed:", self.plank_status.stop, "->", sorted_stopped)
                self.plank_status.stop = sorted_stopped
                # Only add to active detections if there's an active stop
                if len(sorted_stopped) > 0:
                    active_detections.append('stop')
        
        # Handle incorrect detection
        if incorrect is not None:
            sorted_incorrect = sorted(incorrect)
            if sorted_incorrect != self.plank_status.incorrect:
                print("Incorrect status changed:", self.plank_status.incorrect, "->", sorted_incorrect)
                self.plank_status.incorrect = sorted_incorrect
                # Only add to active detections if there's an active incorrect
                if len(sorted_incorrect) > 0:
                    active_detections.append('incorrect')
        
        # Emit the current status to all clients
        self.emit_status()
        
        # Apply rules for all active detections at once
        if active_detections:
            self.apply_rules_for_detections(active_detections)
    
    def apply_rules_for_detections(self, detection_types):
        """Apply rules for multiple detection types at once"""
        try:
            # Initialize actions
            actions_taken = {
                "stop_conveyor": False,
                "alert": [],
                "ignore": []
            }
            
            # Process each detection type
            for detection_type in detection_types:
                rule_action = self.rules[detection_type]
                
                if rule_action == 'stop_conveyor' and not self.plank_status.conveyor_stop:
                    actions_taken["stop_conveyor"] = True
                    print(f"Rule applied: stopping conveyor due to {detection_type}")
                elif rule_action == 'alert':
                    actions_taken["alert"].append(detection_type)
                    print(f"Rule applied: alert for {detection_type}")
                elif rule_action == 'ignore':
                    actions_taken["ignore"].append(detection_type)
                    print(f"Rule applied: ignoring {detection_type}")
            
            # Only send notification if we're taking action
            if actions_taken["stop_conveyor"] or actions_taken["alert"]:
                self.send_to_all_clients({
                    'event': 'rules_applied', 
                    'data': actions_taken
                })
                
        except Exception as e:
            print(f"Error applying rules for detections {detection_types}: {e}")

    def get_status(self):
        """API endpoint to get current status"""
        print("Getting status:", self.plank_status.overlap, self.plank_status.stop, self.plank_status.incorrect)
        return jsonify({
            'overlap': False if len(self.plank_status.overlap) == 0 else True,
            'stop': False if len(self.plank_status.stop) == 0 else True,
            'incorrect': False if len(self.plank_status.incorrect) == 0 else True,
            'conveyor_stop': self.plank_status.conveyor_stop
        })

    def update_rules(self, rules_data = None):
        """Update the rules configuration"""
        if rules_data is None:
            rules_data = request.json
        try:
            print(f"Updating rules: {rules_data}")
            # Validate the rules data
            if not isinstance(rules_data, dict):
                print("Invalid rules data format")
                return False

            # Update the rules
            for key in self.rules.keys():
                if key in rules_data and rules_data[key] in self.rules_options:
                    self.rules[key] = rules_data[key]
            
            print(f"Rules updated: {self.rules}")
            self.emit_rules()
            return True
        except Exception as e:
            print(f"Error updating rules: {e}")
            return False
            

    def index(self):
        """Render the template-based index page"""
        # Pass camera IDs and frame size to the template
        camera_ids = list(self.cameras.keys())
        
        return render_template('index.html', 
                              camera_ids=camera_ids,
                              frame_sizes=self.frame_sizes)
    
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
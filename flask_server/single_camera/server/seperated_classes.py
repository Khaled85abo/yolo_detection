import os
import cv2
import json
import logging
import threading
from queue import Queue
from typing import Dict, List, Tuple, Optional

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    render_template,
    Blueprint,
    current_app
)
from simple_websocket import Server as WebSocketServer, ConnectionClosed

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

DEFAULT_FRAME_SIZE = (640, 480)
RULE_OPTIONS = ['ignore', 'stop_conveyor', 'alert']

# Set up the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Business Logic Classes
# -----------------------------------------------------------------------------

class PlankStatus:
    """
    Holds the status of planks on the conveyor.
    """
    def __init__(self) -> None:
        self.overlap: List[Tuple] = []
        self.stop: List[int] = []
        self.incorrect: List[int] = []
        self.conveyor_stop: bool = False


class StreamServerCore:
    """
    Core logic for the server: camera management, statuses, rule application.
    Note: This class has no Flask references; it's pure business logic.
    """
    def __init__(self) -> None:
        self.plank_status = PlankStatus()
        
        # camera_id -> current frame
        self.cameras: Dict[str, Optional[cv2.Mat]] = {}
        
        # camera_id -> Lock for thread-safe frame access
        self.frame_locks: Dict[str, threading.Lock] = {}
        
        # camera_id -> (width, height)
        self.frame_sizes: Dict[str, Tuple[int, int]] = {}

        # Store rules in memory
        self.rules = {
            'overlap': 'ignore',
            'stop': 'ignore',
            'incorrect': 'ignore'
        }

    def add_camera(self, camera_id: str, frame_size: Tuple[int, int] = DEFAULT_FRAME_SIZE) -> None:
        """
        Register a new camera in the system.
        """
        self.cameras[camera_id] = None
        self.frame_locks[camera_id] = threading.Lock()
        self.frame_sizes[camera_id] = frame_size

    def update_frame(self, camera_id: str, frame: cv2.Mat) -> None:
        """
        Update the current frame for the specified camera.
        """
        if camera_id not in self.cameras:
            logger.warning(f"Camera {camera_id} not registered.")
            return

        with self.frame_locks[camera_id]:
            resized = cv2.resize(frame, self.frame_sizes[camera_id])
            self.cameras[camera_id] = resized.copy()

    def get_frame(self, camera_id: str) -> Optional[cv2.Mat]:
        """
        Retrieve the current frame for the specified camera.
        """
        with self.frame_locks[camera_id]:
            return self.cameras[camera_id]

    def get_status_dict(self) -> dict:
        """
        Return the status of the planks in a serializable dictionary form.
        """
        return {
            'overlap': bool(self.plank_status.overlap),
            'stop': bool(self.plank_status.stop),
            'incorrect': bool(self.plank_status.incorrect),
            'conveyor_stop': self.plank_status.conveyor_stop
        }

    def update_plank_status(self,
                            overlapped: Optional[List[Tuple]] = None,
                            stopped: Optional[List[int]] = None,
                            incorrect: Optional[List[int]] = None) -> bool:
        """
        Update the plank status. Returns True if a status changed, False otherwise.
        """
        status_changed = False
        active_detections = []

        if overlapped is not None:
            new_overlap = sorted(overlapped)
            if new_overlap != self.plank_status.overlap:
                logger.info(f"Overlap changed from {self.plank_status.overlap} to {new_overlap}")
                self.plank_status.overlap = new_overlap
                status_changed = True
                if new_overlap:
                    active_detections.append('overlap')

        if stopped is not None:
            new_stopped = sorted(stopped)
            if new_stopped != self.plank_status.stop:
                logger.info(f"Stop changed from {self.plank_status.stop} to {new_stopped}")
                self.plank_status.stop = new_stopped
                status_changed = True
                if new_stopped:
                    active_detections.append('stop')

        if incorrect is not None:
            new_incorrect = sorted(incorrect)
            if new_incorrect != self.plank_status.incorrect:
                logger.info(f"Incorrect changed from {self.plank_status.incorrect} to {new_incorrect}")
                self.plank_status.incorrect = new_incorrect
                status_changed = True
                if new_incorrect:
                    active_detections.append('incorrect')

        return status_changed, active_detections

    def apply_rules(self, detection_types: List[str]) -> dict:
        """
        Apply the configured rules for the given detection types.
        Returns a dictionary of actions taken (for broadcasting, logs, etc.).
        """
        actions_taken = {
            "stop_conveyor": False,
            "alert": [],
            "ignore": []
        }

        for detection_type in detection_types:
            rule_action = self.rules.get(detection_type, 'ignore')
            if rule_action == 'stop_conveyor' and not self.plank_status.conveyor_stop:
                actions_taken["stop_conveyor"] = True
                logger.info(f"Applying rule: stopping conveyor (due to {detection_type}).")
            elif rule_action == 'alert':
                actions_taken["alert"].append(detection_type)
                logger.info(f"Applying rule: alert for {detection_type}.")
            else:
                actions_taken["ignore"].append(detection_type)

        return actions_taken

    def update_conveyor_stop(self, stop_state: bool) -> None:
        """
        Update the conveyor's stop state.
        """
        self.plank_status.conveyor_stop = stop_state

    def update_rules(self, new_rules: dict) -> None:
        """
        Update the server’s rules. Expects something like:
        {
            "overlap": "ignore",
            "stop": "stop_conveyor",
            "incorrect": "alert"
        }
        """
        for key, value in new_rules.items():
            if key in self.rules and value in RULE_OPTIONS:
                self.rules[key] = value


# -----------------------------------------------------------------------------
# Flask Blueprints
# -----------------------------------------------------------------------------

main_bp = Blueprint('main_bp', __name__, template_folder='templates', static_folder='static')
api_bp = Blueprint('api_bp', __name__)
video_bp = Blueprint('video_bp', __name__)


@main_bp.route('/')
def index():
    """
    Renders the main page with all available camera feeds.
    """
    server_core: StreamServerCore = current_app.config['SERVER_CORE']
    camera_ids = list(server_core.cameras.keys())
    frame_sizes = server_core.frame_sizes
    return render_template('index.html', camera_ids=camera_ids, frame_sizes=frame_sizes)


@video_bp.route('/video_feed/<string:camera_id>')
def video_feed(camera_id: str):
    """
    Provides an MJPEG stream for the specified camera.
    """
    server_core: StreamServerCore = current_app.config['SERVER_CORE']

    def generate_frames():
        frame_skip = 0
        while True:
            frame_skip += 1
            # Skip frames if needed to reduce CPU usage
            if frame_skip % 2 != 0:
                threading.Event().wait(0.01)
                continue

            frame = server_core.get_frame(camera_id)
            if frame is not None:
                try:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() +
                           b'\r\n')
                except Exception as e:
                    logger.exception(f"Error in generate_frames for {camera_id}: {e}")
                    continue

            threading.Event().wait(0.05)

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@api_bp.route('/status', methods=['GET'])
def get_status():
    """
    Retrieves the current status of the planks and conveyor.
    """
    server_core: StreamServerCore = current_app.config['SERVER_CORE']
    return jsonify(server_core.get_status_dict())


@api_bp.route('/control_conveyor', methods=['POST'])
def control_conveyor():
    """
    Endpoint for manual control of the conveyor (from the UI).
    This sends a message to the WebSocket clients (ESP32, etc.)
    """
    server_core: StreamServerCore = current_app.config['SERVER_CORE']
    state = request.json.get('state', False)
    # You could broadcast to all WS clients here, e.g.:
    # broadcast_ws({'event': 'control_conveyor', 'data': {'state': state}})
    logger.info(f"UI requested to set conveyor state to: {state}")
    return jsonify({"message": "Conveyor control command received"}), 200


@api_bp.route('/rules', methods=['POST'])
def update_rules():
    """
    Endpoint to update the system's rules.
    """
    server_core: StreamServerCore = current_app.config['SERVER_CORE']
    try:
        new_rules = request.json
        server_core.update_rules(new_rules)
        # Optionally broadcast to WS clients, e.g.:
        # broadcast_ws({'event': 'rules_update', 'data': {...}})
        return jsonify({"message": "Rules updated", "rules": server_core.rules}), 200
    except Exception as e:
        logger.exception(f"Error updating rules: {e}")
        return jsonify({"error": str(e)}), 400


# -----------------------------------------------------------------------------
# WebSocket Handling
# -----------------------------------------------------------------------------

ws_bp = Blueprint('ws_bp', __name__)

@ws_bp.route('/ws', websocket=True)
def websocket_route():
    """
    WebSocket endpoint that handles real-time communication.
    """
    server_core: StreamServerCore = current_app.config['SERVER_CORE']
    ws = None
    clients: List[WebSocketServer] = current_app.config['WS_CLIENTS']

    try:
        logger.info("Attempting WebSocket handshake...")
        ws = WebSocketServer(request.environ)
        logger.info("WebSocket connection established.")
        clients.append(ws)
        logger.info(f"Client connected. Total clients: {len(clients)}")

        # Upon connection, you might send initial state:
        send_to_client(ws, {
            'event': 'status_update',
            'data': server_core.get_status_dict()
        })
        send_to_client(ws, {
            'event': 'rules_update',
            'data': {'rules': server_core.rules, 'rules_options': RULE_OPTIONS}
        })

        # Listen for incoming messages
        while True:
            message = ws.receive()
            if message:
                handle_websocket_message(server_core, ws, message)

    except ConnectionClosed:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.exception(f"Error in websocket_route: {e}")
    finally:
        if ws and ws in clients:
            clients.remove(ws)
            logger.info(f"Client removed. Remaining clients: {len(clients)}")

    return ''


def handle_websocket_message(server_core: StreamServerCore, ws: WebSocketServer, message: str) -> None:
    """
    Parse and handle incoming WebSocket messages.
    """
    try:
        data = json.loads(message)
        event_type = data.get('event')
        payload = data.get('data', {})
        logger.info(f"Received WS message: {event_type} -> {payload}")

        if event_type == 'control_conveyor':
            # This would come from UI to control the conveyor
            # You can forward it to other clients
            broadcast_ws({
                'event': 'control_conveyor',
                'data': {'state': payload.get('state')}
            })
        elif event_type == 'update_conveyor_stop':
            # From the ESP32, for instance
            state = payload.get('state', False)
            server_core.update_conveyor_stop(state)
            # Broadcast new status to everyone
            broadcast_ws({
                'event': 'status_update',
                'data': server_core.get_status_dict()
            })
        elif event_type == 'update_rules':
            server_core.update_rules(payload)
            broadcast_ws({
                'event': 'rules_update',
                'data': {'rules': server_core.rules, 'rules_options': RULE_OPTIONS}
            })
        elif event_type == 'ping':
            # Optional keep-alive
            pass
        else:
            logger.warning(f"Unhandled WebSocket event: {event_type}")

    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON received: {message}")
    except Exception as e:
        logger.exception(f"Error in handle_websocket_message: {e}")


def send_to_client(ws: WebSocketServer, data: dict) -> None:
    """
    Send data to a single WebSocket client. Safe send with exception handling.
    """
    try:
        ws.send(json.dumps(data))
    except ConnectionClosed:
        logger.info("Connection closed while sending to client.")
    except Exception as e:
        logger.exception(f"Error sending to WS client: {e}")


def broadcast_ws(data: dict) -> None:
    """
    Broadcast data to all connected WebSocket clients.
    """
    clients: List[WebSocketServer] = current_app.config['WS_CLIENTS']
    disconnected = []
    message = json.dumps(data)

    for client in clients:
        try:
            client.send(message)
        except ConnectionClosed:
            disconnected.append(client)
        except Exception as e:
            logger.exception(f"Error broadcasting WS message: {e}")
            disconnected.append(client)

    for client in disconnected:
        if client in clients:
            clients.remove(client)

# -----------------------------------------------------------------------------
# Flask App Factory
# -----------------------------------------------------------------------------

def create_app() -> Flask:
    """
    Application factory that creates and configures a Flask app.
    """
    # Derive template & static folder from this file’s location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(base_dir, 'templates')
    static_dir = os.path.join(base_dir, 'static')

    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

    # Create a core "business-logic" object
    server_core = StreamServerCore()

    # Store references in Flask config so we can access them in blueprints
    app.config['SERVER_CORE'] = server_core
    app.config['WS_CLIENTS'] = []

    # Register blueprints
    app.register_blueprint(main_bp, url_prefix='/')
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(video_bp, url_prefix='/')
    app.register_blueprint(ws_bp, url_prefix='/')

    # Example: Add a default camera or configure more as needed
    server_core.add_camera('camera1', (640, 480))

    # Add cross-origin headers if needed
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        return response

    return app


def run_app(host='0.0.0.0', port=5000):
    """
    Convenience function to create and run the Flask app in one step.
    """
    app = create_app()
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    run_app()

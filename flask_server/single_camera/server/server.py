# ... existing code ...

# Split into multiple files:
# 1. models.py - For data models like PlankStatus
# 2. camera_manager.py - For camera handling
# 3. websocket_handler.py - For WebSocket communication
# 4. rule_engine.py - For rule processing
# 5. routes/ - Directory for route handlers

from flask import Flask, Response, jsonify, request, render_template
import cv2
import threading
from queue import Queue
import logging
from typing import Dict, List, Optional, Any
import os
import json
from simple_websocket import Server as WebSocketServer, ConnectionClosed

# Import refactored modules
from .models import PlankStatus
from .camera_manager import CameraManager
from .websocket_handler import WebSocketHandler
from .rule_engine import RuleEngine
from .routes import register_routes

class StreamServer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StreamServer, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize the server components"""
        # Set template folder relative to the server.py file
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        
        self.app = Flask(__name__, 
                        template_folder=template_dir,
                        static_folder=static_dir)
        
        # Initialize components
        self.plank_status = PlankStatus()
        self.camera_manager = CameraManager()
        self.rule_engine = RuleEngine()
        self.websocket_handler = WebSocketHandler(self)
        
        # Register routes
        register_routes(self)
        
        # Configure logging
        self._configure_logging()

    def __init__(self):
        # Skip initialization if already done
        pass
    
    def _configure_logging(self):
        """Configure application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        # Reduce Flask's default logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

    # ... existing code ...
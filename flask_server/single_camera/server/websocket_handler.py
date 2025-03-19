from typing import List, Dict, Any
import json
import logging
from simple_websocket import Server as WebSocketServer, ConnectionClosed
from flask import request

class WebSocketHandler:
    def __init__(self, server):
        self.server = server
        self.clients = []
        self.logger = logging.getLogger(__name__)
    
    def handle_connection(self):
        """Handle new websocket connections"""
        try:
            self.logger.info("WebSocket connection attempt received")
            ws = WebSocketServer(request.environ)
            self.logger.info("WebSocket connection established successfully")
            self.clients.append(ws)
            self.logger.info(f"Client connected. Total clients: {len(self.clients)}")
            
            # Emit initial status immediately after connection
            self.emit_status()
            self.emit_rules()
            
            try:
                while True:
                    message = ws.receive()
                    if message:
                        self.logger.debug(f"Received message: {message}")
                        self.handle_message(ws, message)
            except ConnectionClosed:
                self.logger.info("Client disconnected")
            finally:
                if ws in self.clients:
                    self.clients.remove(ws)
                    self.logger.info(f"Client removed. Remaining clients: {len(self.clients)}")
            return ''
        except Exception as e:
            self.logger.error(f"Error in websocket_route: {str(e)}", exc_info=True)
            return str(e), 500
    
    def handle_message(self, ws, message):
        """Process incoming websocket messages"""
        try:
            data = json.loads(message)
            event_type = data.get('event')
            payload = data.get('data', {})
            
            # Handle different event types
            if event_type == 'control_conveyor':
                self.on_control_conveyor(payload)
            elif event_type == 'update_conveyor_stop':
                self.update_conveyor_stop(payload)
            elif event_type == 'update_rules':
                self.server.rule_engine.update_rules(payload)
                self.emit_rules()
            elif event_type == 'ping':
                pass  # Just acknowledge receipt
            else:
                self.logger.warning(f"Unknown event type: {event_type}")
            
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON message: {message}")
        except Exception as e:
            self.logger.error(f"Error handling websocket message: {e}")
    
    # ... other methods ...
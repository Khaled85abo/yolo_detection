import websockets
import asyncio
import json
import time
from enum import Enum, auto
from typing import Optional

class PlankStatus(Enum):
    NORMAL = auto()
    STOPPED = auto()
    OVERLAPPED = auto()
    INCORRECT = auto()

class WarningLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2

class ESP32Controller:
    def __init__(self, websocket_url='ws://192.168.1.202/:81'):  # ESP32's IP
        self.websocket_url = websocket_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.warning_states = {
            "stopped": False,
            "overlapped": False,
            "incorrect": False
        }
        # Start connection in background
        asyncio.create_task(self.maintain_connection())

    async def maintain_connection(self):
        while True:
            try:
                if not self.connected:
                    print(f"Connecting to ESP32 at {self.websocket_url}")
                    self.websocket = await websockets.connect(self.websocket_url)
                    self.connected = True
                    print("Connected to ESP32")
                    # Start listening for messages from ESP32
                    asyncio.create_task(self.listen_for_messages())
            except Exception as e:
                print(f"Connection failed: {e}")
                self.connected = False
                await asyncio.sleep(5)  # Wait before retrying

    async def listen_for_messages(self):
        while True:
            try:
                if self.websocket:
                    message = await self.websocket.recv()
                    await self.handle_esp32_message(json.loads(message))
            except Exception as e:
                print(f"Error receiving message: {e}")
                self.connected = False
                break

    async def handle_esp32_message(self, message):
        """Handle messages received from ESP32"""
        try:
            if message.get("type") == "action":
                action = message.get("action")
                if action == "acknowledge_warning":
                    warning_id = message.get("warning_id")
                    print(f"Warning {warning_id} acknowledged by operator")
                elif action == "stop_conveyor":
                    print("Conveyor stop requested by operator")
                elif action == "ignore_warning":
                    warning_id = message.get("warning_id")
                    print(f"Warning {warning_id} ignored by operator")
        except Exception as e:
            print(f"Error handling message: {e}")

    async def send_warning(self, status: PlankStatus, level: WarningLevel, details: dict):
        if not self.connected:
            return

        message = {
            "type": "warning",
            "status": status.name,
            "level": level.value,
            "details": details,
            "timestamp": time.time()
        }

        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            print(f"Failed to send warning to ESP32: {e}")
            self.connected = False

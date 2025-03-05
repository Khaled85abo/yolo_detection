"""
Camera configuration file for the plank detection system.
This file defines all cameras in the system with detailed configuration options.
"""

CAMERAS = {
    "camera1": {
        "id": "camera1",
        "position": "front",
        "url": "http://192.168.1.10:8000/stream",  # Pi camera streamed from python script
        "stream_type": "mjpeg",  # mjpeg, jpeg, or rtsp
        "roi": {
            "width_start": 0.40,
            "width_end": 0.60,
            "height_start": 0.0,
            "height_end": 1.0
        },
        "frame_size": (640, 480),
        "fps": 10,
        "processing": {
            "enabled": True,
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            "max_objects": 10
        },
        "recording": {
            "fps": 10,
            "enabled": True,
            "format": "mp4",
            "quality": "medium"  # low, medium, high
        },
        "connection": {
            "timeout": 5,
            "retry_interval": 1,
            "max_retries": 3
        }
    },
    "camera2": {
        "id": "camera2",
        "position": "side",
        "url": "http://192.168.1.100/stream",  # ESP32-CAM stream
        "stream_type": "jpeg",  # ESP32-CAM often uses single JPEG images
        "roi": {
            "width_start": 0.3,
            "width_end": 0.7,
            "height_start": 0.0,
            "height_end": 1.0
        },
        "frame_size": (640, 480),
        "fps": 10,  # ESP32-CAM might have lower FPS
        "processing": {
            "enabled": True,
            "confidence_threshold": 0.4,  # Lower threshold for this camera
            "iou_threshold": 0.45,
            "max_objects": 10
        },
        "recording": {
            "fps": 10,
            "enabled": True,
            "format": "mp4",
            "quality": "low"  # Lower quality for this camera
        },
        "connection": {
            "timeout": 10,  # Longer timeout for potentially slower device
            "retry_interval": 2,
            "max_retries": 5
        }
    }
}

from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

# Add these to the global variables section
STOP_THRESHOLD_FRAMES = 5  # Number of frames to consider as a stop
MOVEMENT_THRESHOLD = 10    # Pixel distance to consider as movement
stop_detection_memory = defaultdict(lambda: {
    "positions": [],
    "stop_frames": 0,
    "is_stopped": False
})

# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=30,              # Maximum number of frames to keep dead tracks
    n_init=2,                # Number of frames for track initialization
    nms_max_overlap=0.7,     # NMS threshold for suppressing overlapping detections
    max_iou_distance=0.7,    # Maximum IOU distance for matching
    max_cosine_distance=0.3, # Maximum cosine distance for feature matching
    nn_budget=100,           # Maximum size of the appearance descriptors gallery
    embedder="mobilenet",    # Feature extractor
    half=True,              # Use half precision for better speed
    bgr=False,
    embedder_gpu=True
)
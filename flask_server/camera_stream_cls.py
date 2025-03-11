import threading
import time
import requests
import cv2
import numpy as np

class CameraStream:
    def __init__(self, camera_config):
        # Basic configuration
        self.config = camera_config
        self.url = camera_config['url']
        self.stream_type = camera_config.get('stream_type', 'mjpeg')
        self.frame_size = camera_config.get('frame_size', (640, 480))
        self.fps = camera_config.get('fps', 30)

        # Connection error/retry handling
        self.connection_config = camera_config.get('connection', {
            'timeout': 5,
            'retry_interval': 1,
            'max_retries': 3
        })
        self.retry_count = 0

        # Thread control
        self.running = False
        self.thread = None

        # Frame storage
        # If you truly only want the single most recent frame, you can store just one:
        self.frame = None  
        
        # Or if you want a small queue, set max_queue_size > 1 and store multiple frames
        self.frame_queue = []
        self.max_queue_size = camera_config.get('max_queue_size', 1)

        # Synchronization lock to protect self.frame / self.frame_queue
        self.lock = threading.Lock()

    def start(self):
        """Start the camera reading thread."""
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the camera reading thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _update_frame(self):
        """
        Dispatches to the correct streaming method based on self.stream_type.
        Handles mjpeg, jpeg, or rtsp.
        """
        if self.stream_type == 'mjpeg':
            self._update_mjpeg_stream()
        elif self.stream_type == 'jpeg':
            self._update_jpeg_stream()
        elif self.stream_type == 'rtsp':
            self._update_rtsp_stream()
        else:
            print(f"Unknown stream type: {self.stream_type}, defaulting to MJPEG")
            self._update_mjpeg_stream()

    def _update_mjpeg_stream(self):
        """
        Attempts to open the MJPEG stream with OpenCV first.
        If that fails, fallback to requests-based chunk parsing.
        """
        try:
            print(f"Connecting to MJPEG stream at {self.url}")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            cap = cv2.VideoCapture(self.url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Attempt to minimize internal buffering

            if not cap.isOpened():
                print(f"OpenCV failed to open {self.url}, trying requests library...")
                cap.release()
                self._update_mjpeg_stream_requests(headers)
                return

            # If OpenCV connection was successful, read frames in a loop
            while self.running:
                # Read one frame
                success, frame = cap.read()
                if not success:
                    print(f"Failed to read from MJPEG stream: {self.url}")
                    self._retry_opencv_reconnect(cap)
                    continue

                # ---- Flush older frames in the buffer to minimize latency ----
                # We'll keep reading until no more frames are immediately available.
                flushed_frame = None
                while True:
                    success2, frame2 = cap.read()
                    if not success2:
                        break
                    flushed_frame = frame2
                # If we did flush more frames, the last one is the newest
                if flushed_frame is not None:
                    frame = flushed_frame

                # Optionally resize
                if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                    frame = cv2.resize(frame, self.frame_size)

                # Store frame in a thread‐safe manner
                with self.lock:
                    self.frame = frame
                    self._add_to_queue(frame)

                self.retry_count = 0  # Reset retry count on success

            cap.release()

        except Exception as e:
            print(f"Exception in MJPEG stream handler for {self.url}: {e}")
            import traceback
            traceback.print_exc()

    def _update_mjpeg_stream_requests(self, headers):
        """
        Fallback method if OpenCV fails.
        Parses the MJPEG stream chunk by chunk with the requests library.
        """
        while self.running:
            try:
                response = requests.get(
                    self.url,
                    stream=True,
                    timeout=self.connection_config['timeout'],
                    headers=headers
                )
                if response.status_code != 200:
                    print(f"Failed to connect to {self.url}, status: {response.status_code}")
                    time.sleep(self.connection_config['retry_interval'])
                    continue

                # Parse multipart/x-mixed-replace
                bytes_data = bytes()
                for chunk in response.iter_content(chunk_size=1024):
                    if not self.running:
                        break

                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')  # JPEG start
                    b = bytes_data.find(b'\xff\xd9')  # JPEG end

                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        # Decode
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None and frame.size > 0:
                            if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                                frame = cv2.resize(frame, self.frame_size)
                            with self.lock:
                                self.frame = frame
                                self._add_to_queue(frame)
                        self.retry_count = 0

            except Exception as e:
                print(f"Error in requests-based MJPEG stream for {self.url}: {e}")
                self.retry_count += 1
                if self.retry_count > self.connection_config['max_retries']:
                    print(f"Max retries exceeded for {self.url}, waiting longer...")
                    time.sleep(self.connection_config['retry_interval'] * 2)
                    self.retry_count = 0
                else:
                    time.sleep(self.connection_config['retry_interval'])

    def _retry_opencv_reconnect(self, cap):
        """Handles reconnection attempts for OpenCV if read() fails."""
        self.retry_count += 1
        if self.retry_count > self.connection_config['max_retries']:
            print(f"Max retries exceeded for {self.url}, reconnecting...")
            cap.release()
            time.sleep(self.connection_config['retry_interval'])
            cap.open(self.url)
            self.retry_count = 0
        else:
            time.sleep(self.connection_config['retry_interval'])

    def _update_jpeg_stream(self):
        """For single JPEG snapshot endpoints (not MJPEG)."""
        while self.running:
            try:
                response = requests.get(
                    self.url,
                    stream=True,
                    timeout=self.connection_config['timeout']
                )
                if response.status_code == 200:
                    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None and frame.size > 0:
                        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                            frame = cv2.resize(frame, self.frame_size)
                        with self.lock:
                            self.frame = frame
                            self._add_to_queue(frame)
                        self.retry_count = 0
                    else:
                        print(f"Received empty frame from {self.url}")
                        self.retry_count += 1
                else:
                    print(f"Failed to get image from {self.url}, status: {response.status_code}")
                    self.retry_count += 1

            except Exception as e:
                print(f"Error fetching from {self.url}: {e}")
                self.retry_count += 1

            if self.retry_count > self.connection_config['max_retries']:
                print(f"Max retries exceeded for {self.url}, waiting longer...")
                time.sleep(self.connection_config['retry_interval'] * 2)
                self.retry_count = 0
            else:
                # For single JPEG endpoints, we might do another request soon
                time.sleep(self.connection_config['retry_interval'])

    def _update_rtsp_stream(self):
        """Attempts RTSP with OpenCV, flushing older frames on each loop."""
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.running:
            success, frame = cap.read()
            if not success:
                print(f"Failed to read from RTSP stream: {self.url}")
                self._retry_opencv_reconnect(cap)
                continue

            # Flush older frames in buffer
            flushed_frame = None
            while True:
                success2, frame2 = cap.read()
                if not success2:
                    break
                flushed_frame = frame2
            if flushed_frame is not None:
                frame = flushed_frame

            with self.lock:
                self.frame = frame
                self._add_to_queue(frame)
            self.retry_count = 0

        cap.release()

    def _add_to_queue(self, frame):
        """
        Add a frame to the queue. Keep only up to self.max_queue_size frames.
        If you only care about the newest frame, set max_queue_size=1.
        """
        if self.max_queue_size < 1:
            return  # queue disabled, do nothing

        self.frame_queue.append(frame.copy())
        while len(self.frame_queue) > self.max_queue_size:
            self.frame_queue.pop(0)

    def capture_array(self):
        """
        Returns the most recent frame.
        If using a queue, we discard older frames here and keep only the newest.
        """
        with self.lock:
            if not self.frame_queue:
                # If queue is empty, fallback to self.frame (the single newest)
                return self.frame.copy() if self.frame is not None else None

            # If queue has frames, pop the newest
            newest = self.frame_queue.pop()
            # Optionally clear the queue entirely, so next call also sees the newest
            self.frame_queue.clear()

        if newest is not None and newest.size > 0:
            return newest
        return None
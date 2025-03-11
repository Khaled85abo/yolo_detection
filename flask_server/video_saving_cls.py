import cv2
import os
from datetime import datetime
import threading


class OutputVideo:
    _instance = None
    _lock = threading.Lock()
    base_directory = 'videos_output'
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, fps=10, frame_width=640, frame_height=480, bitrate=None):
        with self._lock:
            # Skip initialization if already done
            if not self._initialized:
                self.fps = fps
                self.frame_width = frame_width
                self.frame_height = frame_height
                self.bitrate = bitrate
                self.video_writers = {}
                
                # Create base directory if it doesn't exist
                if not os.path.exists(self.base_directory):
                    os.makedirs(self.base_directory)
                    print(f"Created base directory: {self.base_directory}")
                    
                self._initialized = True

    def create_writer(self, name, subfolder=None, include_timestamp=True, suffix=None):
        """
        Create a new video writer with a specific name
        
        Args:
            name (str): Base name for the video
            subfolder (str, optional): Subfolder within base_directory
            include_timestamp (bool): Whether to include timestamp in filename
            suffix (str, optional): Optional suffix to add before file extension
            
        Returns:
            str: The key used to identify this writer
        """
        # Create full directory path
        with self._lock:
            output_dir = self.base_directory
            if subfolder:
                output_dir = os.path.join(output_dir, subfolder)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if include_timestamp else ''
            suffix = f"_{suffix}" if suffix else ''
            filename = f"{name}{suffix}_{timestamp}.mp4" if timestamp else f"{name}{suffix}.mp4"
            
            output_path = os.path.join(output_dir, filename)
        
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    output_path, 
                    fourcc, 
                    self.fps, 
                    (self.frame_width, self.frame_height)
                )
                
                # Set bitrate if specified
                if self.bitrate is not None and hasattr(writer, 'set'):
                    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, self.bitrate)
                
                if writer.isOpened():
                    writer_key = f"{name}{suffix}"
                    self.video_writers[writer_key] = {
                        'writer': writer,
                        'path': output_path
                    }
                    print(f"Successfully created video writer: {output_path}")
                    return writer_key
                else:
                    raise Exception(f"Failed to open VideoWriter for {output_path}")
                
            except Exception as e:
                print(f"Error creating video writer for {output_path}: {str(e)}")
                return None

    def write_frame(self, frame, writer_key=None):
        """
        Write frame to specified writer(s)
        
        Args:
            frame: The frame to write
            writer_key (str or list, optional): Specific writer(s) to use. If None, writes to all.
        """
        if frame is None:
            return
        
        if writer_key is None:
            # Write to all writers
            for writer_info in self.video_writers.values():
                writer_info['writer'].write(frame)
        elif isinstance(writer_key, (list, tuple)):
            # Write to multiple specific writers
            for key in writer_key:
                if key in self.video_writers:
                    self.video_writers[key]['writer'].write(frame)
        else:
            # Write to a single specific writer
            if writer_key in self.video_writers:
                self.video_writers[writer_key]['writer'].write(frame)

    def release(self, writer_key=None):
        """
        Release specified writer(s)
        
        Args:
            writer_key (str or list, optional): Specific writer(s) to release. If None, releases all.
        """
        with self._lock:
            if writer_key is None:
                # Release all writers
                for writer_info in self.video_writers.values():
                    writer_info['writer'].release()
                self.video_writers.clear()
            elif isinstance(writer_key, (list, tuple)):
                # Release specific writers
                for key in writer_key:
                    if key in self.video_writers:
                        self.video_writers[key]['writer'].release()
                        del self.video_writers[key]
            else:
                # Release a single specific writer
                if writer_key in self.video_writers:
                    self.video_writers[writer_key]['writer'].release()
                    del self.video_writers[writer_key]

    def __del__(self):
        """Ensure all writers are released when the object is destroyed"""
        self.release()


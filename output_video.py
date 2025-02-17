# use it in multiple files like this:
# # Example usage in any file:
# from output_video import VideoOutputHandler

# # Create a video handler
# video_handler = VideoOutputHandler(
#     base_directory='videos_output',
#     fps=10,
#     target_width=640,
#     target_height=480
# )

# # Create different writers for different purposes
# main_writer = video_handler.create_writer(
#     name='camera1',
#     subfolder='raw_footage',
#     include_timestamp=True
# )

# roi_writer = video_handler.create_writer(
#     name='camera1',
#     subfolder='roi_footage',
#     suffix='roi',
#     include_timestamp=True
# )

# # Write frames to specific writers
# video_handler.write_frame(full_frame, main_writer)
# video_handler.write_frame(roi_frame, roi_writer)

# # Or write the same frame to multiple writers
# video_handler.write_frame(frame, [main_writer, roi_writer])

# # Release specific writers when done
# video_handler.release(main_writer)
# video_handler.release(roi_writer)

# # Or release all writers
# video_handler.release()


# # File 1: camera1_processing.py
# from output_video import VideoOutputHandler

# video_handler = VideoOutputHandler(base_directory='videos_output')
# camera1_writer = video_handler.create_writer(
#     name='camera1',
#     subfolder='camera1_footage'
# )

# # File 2: camera2_processing.py
# from output_video import VideoOutputHandler

# video_handler = VideoOutputHandler()  # Will use same instance
# camera2_writer = video_handler.create_writer(
#     name='camera2',
#     subfolder='camera2_footage'
# )

# # File 3: analysis.py
# from output_video import VideoOutputHandler

# video_handler = VideoOutputHandler()  # Will use same instance
# analysis_writer = video_handler.create_writer(
#     name='analysis',
#     subfolder='processed'
# )

import cv2
import os
from datetime import datetime
import threading
class OutputVideo:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, base_directory='videos_output', fps=10, target_width=640, target_height=480):
        with self._lock:
            # Skip initialization if already done
            if not self._initialized:
                self.base_directory = base_directory
                self.fps = fps
                self.target_width = target_width
                self.target_height = target_height
                self.video_writers = {}
                
                # Create base directory if it doesn't exist
                if not os.path.exists(base_directory):
                    os.makedirs(base_directory)
                    print(f"Created base directory: {base_directory}")
                    
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
                (self.target_width, self.target_height)
            )
            
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




# Single video output


# import cv2
# import os

# class VideoOutputHandler:
#     def __init__(self, output_path, fps=10, target_width=640, target_height=480):
#         self.output_path = output_path
#         self.fps = fps
#         self.target_width = target_width
#         self.target_height = target_height
#         self.out = None
#         self.out_roi = None
#         self.initialize_writers()

#     def initialize_writers(self):
#         """Initialize video writers with error handling"""
#         # Create output directory if it doesn't exist
#         output_dir = os.path.dirname(self.output_path)
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#             print(f"Created output directory: {output_dir}")

#         # Test with a simple output path first
#         test_output = 'test_output.mp4'
#         print(f"Testing VideoWriter with simple path: {test_output}")
        
#         try:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             test_out = cv2.VideoWriter(test_output, fourcc, self.fps, 
#                                      (self.target_width, self.target_height))
            
#             if test_out.isOpened():
#                 print("VideoWriter opened successfully with test path")
#                 test_out.release()
                
#                 # Initialize actual video writers
#                 output_avi = self.output_path.rsplit('.', 1)[0] + '.mp4'
#                 output_avi_roi = self.output_path.rsplit('.', 1)[0] + '_roi.mp4'
                
#                 self.out = cv2.VideoWriter(output_avi, fourcc, self.fps, 
#                                          (self.target_width, self.target_height))
#                 self.out_roi = cv2.VideoWriter(output_avi_roi, fourcc, self.fps, 
#                                              (self.target_width, self.target_height))
                
#                 if self.out.isOpened():
#                     self.output_path = output_avi
#                     print(f"Successfully initialized VideoWriter: {self.output_path}")
#                 else:
#                     raise Exception("Failed to open VideoWriter with actual path")
#             else:
#                 raise Exception("Failed to open VideoWriter with test path")
                
#         except Exception as e:
#             print(f"Failed to initialize VideoWriter: {str(e)}")
#             print("Trying with absolute path...")

#     def write_frame(self, frame, roi_frame=None):
#         """Write frames to video files"""
#         if self.out is not None and frame is not None:
#             self.out.write(frame)
#         if self.out_roi is not None and roi_frame is not None:
#             self.out_roi.write(roi_frame)

#     def release(self):
#         """Release video writers"""
#         if self.out is not None:
#             self.out.release()
#         if self.out_roi is not None:
#             self.out_roi.release()
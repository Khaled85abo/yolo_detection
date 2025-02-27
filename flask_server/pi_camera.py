# Remove eventlet from the top - it's interfering with camera processing
# import eventlet
# eventlet.monkey_patch()


from ultralytics import YOLO
from picamera2 import Picamera2
import time

from utilities.process_frame import process_frame
from utilities.draw_boxes import draw_boxes_and_orientations
from utilities.detect_stop import tracker
#   
target_width = 640
target_height = 480
custom_fps = 10 

# ROI parameters
ROI_start = 0.40
ROI_end = 0.60


# Initialize YOLO model
model = YOLO('/home/rise/enter/train_yolo11n/weights/best_yolo11n.pt')


def main():
    print("Initializing camera...")
    picam2 = Picamera2()

    print("Configuring camera settings...")
    



    # Ensure ROI width is even
    roi_width = int(target_width * (ROI_end - ROI_start))
    if roi_width % 2 != 0:
        roi_width += 1  # Make it even
    
    print(f"Full dimensions: {target_width}x{target_height}")
    print(f"ROI width: {roi_width}")
    
    config = picam2.create_video_configuration(
        main={"size": (target_width, target_height), "format": "RGB888"},
        controls={"FrameDurationLimits": (33333, 33333)}  # ~30fps
    )
    picam2.configure(config)

    picam2.set_controls({"AeEnable": True})

    print("Starting camera...")
    picam2.start()

    try:
        # Import StreamServer from your local server module
        from server import StreamServer
        print("Starting stream server...")
        server = StreamServer()
        server.add_camera('camera1', frame_size=(roi_width, target_height))
        server_thread = server.run_threaded()
        print("Stream server started successfully")
    except Exception as e:
        print(f"Failed to start stream server: {e}")
        return

    # Initialize video output
    try:
        from video_output_class import OutputVideo
        out_cls = OutputVideo( fps=custom_fps, target_width=roi_width, target_height=target_height)
        out_cls.create_writer(name='camera1', subfolder='pi')

    except Exception as e:
        print(f"Failed to initialize OutputVideo: {str(e)}")
        return



    try:
        
        frame_count = 0
        while True:
            frame_count += 1
            print(f"\n--- Frame {frame_count} ---")
            
            loop_start = time.time()
            
            # Capture frame
            capture_start = time.time()
            frame = picam2.capture_array()
            capture_end = time.time()
            
            roi_x_start = int(target_width * ROI_start)
            roi_x_end = int(target_width * ROI_end)
            
            # Extract ROI
            cropped_frame = frame[:, roi_x_start:roi_x_end]
            
            # Process frame
            process_start = time.time()
            tracked_objects, orientations, roi_bounds = process_frame(cropped_frame, model, tracker, server)
            process_end = time.time()
            
            # Print summary
            objects_in_roi = len(tracked_objects)
            print(f"Summary: {objects_in_roi} tracked objects")
            
            # Draw results
            draw_start = time.time()
            frame = draw_boxes_and_orientations(cropped_frame, tracked_objects, orientations, roi_bounds)
            draw_end = time.time()
            
            server.update_frame('camera1', frame)
            
            # Write original frame to video file
            out_write_start = time.time()
            out_cls.write_frame(frame, writer_key='camera1')
            out_write_end = time.time()
            
            loop_end = time.time()
            
            print(
                f"[main loop] Frame {frame_count} total: {(loop_end - loop_start):.3f}s | "
                f"capture: {(capture_end - capture_start):.3f}s | "
                f"process_frame: {(process_end - process_start):.3f}s | "
                f"draw: {(draw_end - draw_start):.3f}s | "
                f"write: {(out_write_end - out_write_start):.3f}s"
            )

    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        print("Cleaning up...")
        picam2.stop()
        out_cls.release(writer_key='camera1')
        # allow time for the server to close
        time.sleep(0.5)
        print("Done!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import time

# Set up logging
log_dir = "/var/log/multi_cameras"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "multi_cameras.log")

logger = logging.getLogger("multi_cameras")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Add console output
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

# Import the main function
try:
    from multi_cameras_processor import main
    logger.info("Starting multi-cameras service")
    
    # Run with automatic restart on failure
    while True:
        try:
            main()
        except Exception as e:
            logger.error(f"Service crashed: {str(e)}", exc_info=True)
            logger.info("Restarting service in 10 seconds...")
            time.sleep(10)
            
except Exception as e:
    logger.critical(f"Failed to start service: {str(e)}", exc_info=True)
    sys.exit(1)
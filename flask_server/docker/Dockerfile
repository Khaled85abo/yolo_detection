FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV with video support
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for models (optional but recommended)
RUN mkdir -p /app/models

# Copy application code
COPY . .

# Create directory for video output
RUN mkdir -p /app/videos_output

# Expose ports
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
# Add environment variable for model path (optional)
ENV MODEL_PATH=/app/models/your_trained_model.pt

# Run the application
CMD ["python", "multi_cameras_processor.py"]


# docker build -t multi-cameras .
# docker run --network host --restart unless-stopped multi-cameras


# project/
# ├── docker/
# │   └── Dockerfile
# ├── models/
# │   └── trained_model.pt
# ├── multi_cameras_processor.py
# ├── server_websocket.py
# ├── video_saving_class.py
# ├── requirements.txt
# └── videos_output/    (created by Docker)
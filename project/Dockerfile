# Lightweight Python base image
FROM python:3.10-slim AS base

WORKDIR /app

# Install system libraries (needed for OpenCV + imshow)
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and weights
COPY main.py best.pt /app/

# Default command
CMD ["python3", "main.py"]

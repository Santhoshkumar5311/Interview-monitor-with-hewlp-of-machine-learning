# Multi-stage build for Interview Monitor
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional system packages for audio
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyQt5 for GUI
RUN apt-get update && apt-get install -y \
    python3-pyqt5 \
    python3-pyqt5.qtchart \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY *.py /app/
COPY *.md /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/credentials /app/data

# Set permissions
RUN chmod +x /app/*.py

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Expose ports
EXPOSE 8000 8080

# Default command
CMD ["python", "src/interview_monitor.py"]

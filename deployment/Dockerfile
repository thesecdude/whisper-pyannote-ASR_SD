# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and pyproject.toml first for better layer caching
COPY requirements.txt pyproject.toml MANIFEST.in ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Install additional dependencies for ASR_SD benchmarks
# (pyannote.audio and other dependencies not in main requirements.txt)
RUN pip install --no-cache-dir \
    pyannote.audio \
    soundfile \
    librosa

# Environment variables for CPU performance
# Not setting defaults here - will use all available cores unless overridden
# You can override these via docker-compose.yml or at runtime

# Create directories for data and results
RUN mkdir -p /app/data /app/results

# Default command - runs whisper help
CMD ["whisper", "--help"]

# Docker Usage Guide for Whisper

This guide explains how to use Docker with this Whisper repository.

## Prerequisites

- Docker installed on your system
- Docker Compose installed (usually comes with Docker Desktop)

## Quick Start

All Docker files are located in the `deployment/` directory. Navigate there first:

```bash
cd deployment
```

### 1. Build the Docker Image

```bash
docker build -t whisper:latest -f Dockerfile ..
```

Or using docker-compose:

```bash
docker-compose build
```

### 2. Run with Docker Compose

#### Option A: Interactive Container (Recommended for Development)

Start an interactive container that stays running:

```bash
docker-compose up -d whisper
```

Then execute commands inside the container:

```bash
# Transcribe a single audio file
docker-compose exec whisper whisper /app/data/audio.wav --model base --output_dir /app/results

# Run Python scripts
docker-compose exec whisper python ASR_SD/core/whisper_pyannote.py --help

# Access interactive shell
docker-compose exec whisper bash
```

#### Option B: One-off Commands

```bash
# Transcribe audio files
docker-compose run --rm whisper whisper /app/data/audio.wav --model turbo

# Run benchmark
docker-compose run --rm whisper python ASR_SD/benchmarks/benchmark_whisper_cpu.py /app/data/audio.wav
```

### 3. Using Docker Directly

```bash
# From the deployment directory
cd deployment

# Run transcription
docker run --rm -v $(pwd)/../data:/app/data -v $(pwd)/../results:/app/results \
  whisper:latest whisper /app/data/audio.wav --model base --output_dir /app/results

# Interactive shell
docker run --rm -it -v $(pwd)/../data:/app/data -v $(pwd)/../results:/app/results \
  whisper:latest bash
```

## Directory Structure

The Docker setup expects the following structure:

```
whisper/
├── deployment/          # Docker configuration files
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── docker-compose.yml
│   └── DOCKER_USAGE.md
├── data/               # Place your input audio files here
├── results/            # Output files will be saved here
├── ASR_SD/             # ASR and Speaker Diarization code
└── whisper/            # Core Whisper code
```

The `data/` and `results/` directories at the project root are automatically mounted as volumes.

## Running ASR_SD Benchmarks

### Pyannote Benchmark (Requires HuggingFace Token)

First, set your HuggingFace token:

```bash
export HF_TOKEN=your_huggingface_token_here
```

Then run:

```bash
docker-compose run --rm -e HF_TOKEN=$HF_TOKEN whisper \
  python ASR_SD/benchmarks/benchmark_pyannote_cpu.py \
  /app/data/audio.wav \
  --hf-token $HF_TOKEN \
  --output /app/results/pyannote_benchmark.json
```

### Whisper Benchmark

```bash
docker-compose run --rm whisper \
  python ASR_SD/benchmarks/benchmark_whisper_cpu.py \
  /app/data/audio.wav \
  --output /app/results/whisper_benchmark.json
```

### Combined Whisper + Pyannote

```bash
docker-compose run --rm -e HF_TOKEN=$HF_TOKEN whisper \
  python ASR_SD/core/whisper_pyannote.py \
  /app/data/audio.wav \
  --hf-token $HF_TOKEN \
  --output /app/results/combined_output.json
```

## Environment Variables

You can customize the Docker environment using these variables:

- `HF_TOKEN` - Your HuggingFace token for pyannote models
- `OMP_NUM_THREADS` - Number of OpenMP threads (default: 4)
- `MKL_NUM_THREADS` - Number of MKL threads (default: 4)

Create a `.env` file in the `deployment/` directory:

```bash
HF_TOKEN=your_token_here
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

Docker Compose will automatically load these variables.

## GPU Support (Optional)

To enable GPU support, modify the `docker-compose.yml`:

```yaml
services:
  whisper:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

And use the NVIDIA base image in `Dockerfile`:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

## Tips

1. **First Run**: The first time you transcribe, Whisper will download the model weights. This may take a few minutes.

2. **Model Storage**: Models are cached inside the container. To persist them:
   ```bash
   docker run -v ~/.cache/whisper:/root/.cache/whisper ...
   ```

3. **Cleanup**:
   ```bash
   # Stop all containers
   docker-compose down

   # Remove image
   docker rmi whisper:latest
   ```

4. **View Logs**:
   ```bash
   docker-compose logs -f whisper
   ```

## Example Workflow

```bash
# 1. Place audio files in data directory (from project root)
cp ~/my_audio.wav ./data/

# 2. Navigate to deployment directory
cd deployment

# 3. Build image
docker-compose build

# 4. Start container
docker-compose up -d ASR

# 5. 
  #Transcribe with Whisper CLI 
    docker-compose exec whisper whisper /app/data/my_audio.wav --model base --output_dir /app/results

  #Transcribe + Speaker D. 
    docker-compose exec whisper python ASR_SD/core/whisper_pyannote.py \
        /app/data/my_audio.wav \
        --hf-token YOUR_TOKEN \
        --output /app/results/output.json

  #Transcribe + Speaker D. in parallel execution
    docker-compose exec whisper python ASR_SD/core/whisper_pyannote.py \
        /app/data/my_audio.wav \
        --hf-token YOUR_TOKEN \
        --output /app/results/output.json

  #Benchmarking performance
    docker-compose exec whisper python ASR_SD/benchmarks/benchmark_whisper_cpu.py \
        /app/data/my_audio.wav \
        --model large-v3 \
        --output /app/results/whisper_benchmark.json

    docker-compose exec whisper python ASR_SD/benchmarks/benchmark_pyannote_cpu.py \
        /app/data/my_audio.wav \
        --hf-token YOUR_TOKEN \
        --output /app/results/pyannote_benchmark.json

# 6. Check results (from project root)
ls -la ../results/

# 7. Clean up
docker-compose down
```

## Troubleshooting

### Issue: "Permission denied" errors
Solution: Ensure the data and results directories have proper permissions:
```bash
chmod -R 755 ./data ./results
```

### Issue: Container exits immediately
Solution: Use the default docker-compose configuration which keeps the container running with `tail -f /dev/null`

### Issue: Out of memory
Solution: Increase Docker memory limits in Docker Desktop settings or use smaller Whisper models (tiny, base)

### Issue: HuggingFace authentication errors
Solution: Ensure your HF_TOKEN is valid and you've accepted the pyannote model terms at https://huggingface.co/pyannote/speaker-diarization-3.1

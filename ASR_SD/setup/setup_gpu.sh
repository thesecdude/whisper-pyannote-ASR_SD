#!/bin/bash
# Whisper + Pyannote Diarization - GPU Setup Script
# This script sets up the complete transcription system on a GPU-enabled machine

set -e  # Exit on error

echo "=========================================="
echo "Whisper + Pyannote GPU Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running on Linux (typical for GPU servers)
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_warning "This script is optimized for Linux GPU servers"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    print_status "NVIDIA GPU detected"
else
    print_error "NVIDIA GPU not detected. Install NVIDIA drivers first."
    exit 1
fi

# Check Python version
echo ""
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Python version: $PYTHON_VERSION"

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.8 or higher"
    exit 1
fi

# Check CUDA
echo ""
echo "Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_status "CUDA version: $CUDA_VERSION"
else
    print_warning "CUDA compiler not found in PATH"
fi

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
python3 -m venv whisper_env
source whisper_env/bin/activate
print_status "Virtual environment created and activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
print_warning "Installing PyTorch for CUDA 12.1 (adjust if needed)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch CUDA support..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Install Whisper
echo ""
echo "Installing OpenAI Whisper..."
pip install -U openai-whisper

# Install pyannote.audio
echo ""
echo "Installing pyannote.audio for speaker diarization..."
pip install pyannote.audio

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install tiktoken numba more-itertools

# Download model files (optional - speeds up first run)
echo ""
read -p "Pre-download Whisper turbo model? (Recommended, ~1.5GB) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Downloading Whisper turbo model..."
    python3 -c "import whisper; whisper.load_model('turbo')"
    print_status "Model downloaded"
fi

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p results
mkdir -p logs
print_status "Directories created"

# Get HuggingFace token
echo ""
echo "=========================================="
echo "HuggingFace Token Setup"
echo "=========================================="
echo "Pyannote requires a HuggingFace token for speaker diarization."
echo ""
echo "Steps to get your token:"
echo "1. Go to: https://huggingface.co/settings/tokens"
echo "2. Create a token with 'read' access"
echo "3. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "4. Accept terms at: https://huggingface.co/pyannote/segmentation-3.0"
echo ""
read -p "Enter your HuggingFace token (or press Enter to skip): " HF_TOKEN

if [ ! -z "$HF_TOKEN" ]; then
    echo "export HF_TOKEN='$HF_TOKEN'" >> ~/.bashrc
    export HF_TOKEN="$HF_TOKEN"
    print_status "HuggingFace token saved to ~/.bashrc"
else
    print_warning "Skipped HuggingFace token setup. You'll need to provide it when running transcriptions."
fi

# Create requirements.txt for reference
echo ""
echo "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Whisper + Pyannote Diarization Requirements
# Install with: pip install -r requirements.txt

# PyTorch (install separately with CUDA support)
# torch>=2.0.0
# torchvision>=0.15.0
# torchaudio>=2.0.0

# Whisper
openai-whisper>=20231117

# Pyannote for speaker diarization
pyannote.audio>=3.1.0

# Additional dependencies
tiktoken>=0.5.0
numba>=0.60.0
more-itertools>=10.0.0
EOF
print_status "requirements.txt created"

# Create monitoring script
echo ""
echo "Creating GPU monitoring script..."
cat > monitor_gpu.sh << 'EOF'
#!/bin/bash
# GPU Monitoring Script
echo "=========================================="
echo "GPU Resource Monitor"
echo "=========================================="
echo ""
watch -n 2 'nvidia-smi; echo ""; ps aux | grep whisper | grep -v grep | awk "{print \$2, \$3, \$4, \$11}"'
EOF
chmod +x monitor_gpu.sh
print_status "GPU monitoring script created: ./monitor_gpu.sh"

# Create example usage script
echo ""
echo "Creating example usage script..."
cat > run_transcription.sh << 'EOF'
#!/bin/bash
# Example Transcription Script

# Activate virtual environment
source whisper_env/bin/activate

# Set HuggingFace token (if not in environment)
# export HF_TOKEN='your_token_here'

# Example: Transcribe with turbo model on GPU
python whisper_diarization.py \
    "path/to/your/audio.mp3" \
    --whisper-model turbo \
    --output "output/transcript" \
    --hf-token "$HF_TOKEN" \
    --device cuda

# For multilingual (English-Hindi code-switching)
# Add: --multilingual

# For specific number of speakers
# Add: --num-speakers 2
EOF
chmod +x run_transcription.sh
print_status "Example script created: ./run_transcription.sh"

# Print system information
echo ""
echo "=========================================="
echo "System Information Summary"
echo "=========================================="
python3 << 'EOF'
import torch
import whisper
import platform

print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  - Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  - Compute Capability: {props.major}.{props.minor}")
print(f"Whisper: {whisper.__version__}")
try:
    import pyannote.audio
    print(f"Pyannote: Available")
except:
    print(f"Pyannote: Not installed")
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
print_status "Virtual environment created: ./whisper_env"
print_status "Activate with: source whisper_env/bin/activate"
echo ""
echo "Next steps:"
echo "1. Copy your whisper_diarization.py script to this directory"
echo "2. Activate the environment: source whisper_env/bin/activate"
echo "3. Run transcription: python whisper_diarization.py [audio_file] --device cuda"
echo ""
echo "Useful commands:"
echo "  - Monitor GPU: ./monitor_gpu.sh"
echo "  - Example run: ./run_transcription.sh"
echo "  - Check GPU: nvidia-smi"
echo ""
print_warning "Remember to set your HuggingFace token before running!"
echo ""

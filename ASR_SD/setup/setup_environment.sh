#!/bin/bash
################################################################################
# Setup script for Whisper + Pyannote transcription pipeline
# This script installs all dependencies needed to run complete_pipeline.py
################################################################################

set -e  # Exit on error

echo "================================================================================================"
echo "Whisper + Pyannote Transcription Pipeline - Environment Setup"
echo "================================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}WARNING: Not running in a virtual environment!${NC}"
    echo "It's recommended to create and activate a virtual environment first:"
    echo ""
    echo "  python -m venv venv"
    echo "  source venv/bin/activate  # On Linux/Mac"
    echo "  # OR"
    echo "  venv\\Scripts\\activate     # On Windows"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Running in virtual environment: $VIRTUAL_ENV${NC}"
    echo ""
fi

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python version: $PYTHON_VERSION"

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
    echo -e "${RED}ERROR: Python 3.8 or higher required${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python version compatible${NC}"
echo ""

# Detect GPU and CUDA availability
echo "================================================================================================"
echo "Detecting GPU and CUDA..."
echo "================================================================================================"

HAS_NVIDIA_GPU=false
CUDA_AVAILABLE=false

# Check for nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader
    HAS_NVIDIA_GPU=true

    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
        echo -e "${GREEN}✓ CUDA version: $CUDA_VERSION${NC}"
        CUDA_AVAILABLE=true
    else
        echo -e "${YELLOW}! CUDA toolkit not found in PATH${NC}"
        echo "  PyTorch will be installed with CUDA support, but nvcc won't be available"
    fi
else
    echo -e "${YELLOW}! No NVIDIA GPU detected (nvidia-smi not found)${NC}"
    echo "  Will install CPU-only versions"
fi

echo ""

# Upgrade pip, setuptools, wheel
echo "================================================================================================"
echo "Upgrading pip, setuptools, and wheel..."
echo "================================================================================================"

python -m pip install --upgrade pip setuptools wheel

echo -e "${GREEN}✓ Package managers updated${NC}"
echo ""

# Install PyTorch with appropriate CUDA support
echo "================================================================================================"
echo "Installing PyTorch..."
echo "================================================================================================"

if [[ $HAS_NVIDIA_GPU == true ]]; then
    echo "Installing PyTorch with CUDA support..."
    # For CUDA 11.8 (most compatible with recent GPUs)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Verify CUDA is available in PyTorch
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
else
    echo "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Install OpenAI Whisper
echo "================================================================================================"
echo "Installing OpenAI Whisper..."
echo "================================================================================================"

pip install -U openai-whisper

echo -e "${GREEN}✓ Whisper installed${NC}"
echo ""

# Install ffmpeg-python (for audio processing)
echo "================================================================================================"
echo "Installing ffmpeg-python..."
echo "================================================================================================"

pip install ffmpeg-python

# Check if ffmpeg binary is available
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}WARNING: ffmpeg binary not found in PATH${NC}"
    echo "You need to install ffmpeg separately:"
    echo ""
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  CentOS/RHEL:   sudo yum install ffmpeg"
    echo "  macOS:         brew install ffmpeg"
    echo "  Windows:       Download from https://ffmpeg.org/download.html"
    echo ""
else
    FFMPEG_VERSION=$(ffmpeg -version | head -n1)
    echo -e "${GREEN}✓ ffmpeg binary found: $FFMPEG_VERSION${NC}"
fi

echo ""

# Install Pyannote Audio
echo "================================================================================================"
echo "Installing Pyannote Audio..."
echo "================================================================================================"

pip install pyannote.audio

echo -e "${GREEN}✓ Pyannote Audio installed${NC}"
echo ""

# Install system monitoring tools
echo "================================================================================================"
echo "Installing monitoring tools (psutil, GPUtil)..."
echo "================================================================================================"

pip install psutil
pip install gputil

echo -e "${GREEN}✓ Monitoring tools installed${NC}"
echo ""

# Install Anthropic SDK for Claude (LLM refinement)
echo "================================================================================================"
echo "Installing Anthropic SDK for Claude..."
echo "================================================================================================"

pip install anthropic

echo -e "${GREEN}✓ Anthropic SDK installed${NC}"
echo ""

# Install Google Cloud Vertex AI SDK (for Vertex AI Claude)
echo "================================================================================================"
echo "Installing Google Cloud Vertex AI SDK..."
echo "================================================================================================"

pip install google-cloud-aiplatform

echo -e "${GREEN}✓ Google Cloud Vertex AI SDK installed${NC}"
echo ""

# Install additional utilities
echo "================================================================================================"
echo "Installing additional utilities..."
echo "================================================================================================"

pip install tqdm  # Progress bars
pip install pydub  # Audio manipulation

echo -e "${GREEN}✓ Additional utilities installed${NC}"
echo ""

# Create requirements.txt for future reference
echo "================================================================================================"
echo "Creating requirements.txt..."
echo "================================================================================================"

cat > requirements.txt << 'EOF'
# Core dependencies for Whisper + Pyannote transcription pipeline

# PyTorch (install separately based on CUDA version)
# For CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU only:  pip install torch torchvision torchaudio

# Speech recognition and diarization
openai-whisper>=20231117
pyannote.audio>=3.1.0

# Audio processing
ffmpeg-python>=0.2.0
pydub>=0.25.1

# LLM integration
anthropic>=0.18.0
google-cloud-aiplatform>=1.38.0

# System monitoring
psutil>=5.9.0
gputil>=1.4.0

# Utilities
tqdm>=4.65.0
EOF

echo -e "${GREEN}✓ requirements.txt created${NC}"
echo ""

# Print summary
echo "================================================================================================"
echo "Installation Summary"
echo "================================================================================================"
echo ""

python << 'PYEOF'
import sys

print("Installed packages:")
print("-" * 80)

packages_to_check = [
    ("torch", "PyTorch"),
    ("whisper", "OpenAI Whisper"),
    ("pyannote.audio", "Pyannote Audio"),
    ("anthropic", "Anthropic SDK"),
    ("google.cloud.aiplatform", "Google Vertex AI"),
    ("psutil", "psutil (CPU/RAM monitoring)"),
    ("GPUtil", "GPUtil (GPU monitoring)"),
    ("ffmpeg", "ffmpeg-python"),
]

all_installed = True

for module_name, display_name in packages_to_check:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ {display_name:30s} {version}")
    except ImportError:
        print(f"✗ {display_name:30s} NOT INSTALLED")
        all_installed = False

print("-" * 80)

if all_installed:
    print("\n✓ All packages installed successfully!")
else:
    print("\n! Some packages failed to install")
    sys.exit(1)

# Check CUDA availability
print("\nGPU/CUDA Status:")
print("-" * 80)

try:
    import torch
    print(f"PyTorch version:    {torch.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:       {torch.version.cuda}")
        print(f"cuDNN version:      {torch.backends.cudnn.version()}")
        print(f"GPU count:          {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:              {torch.cuda.get_device_name(i)}")
    else:
        print("! Running in CPU mode - GPU acceleration not available")
except Exception as e:
    print(f"Error checking CUDA: {e}")

print("-" * 80)
PYEOF

echo ""
echo "================================================================================================"
echo "Setup Complete!"
echo "================================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Set up authentication:"
echo "   - HuggingFace token: Visit https://huggingface.co/settings/tokens"
echo "     Accept pyannote/speaker-diarization-3.1 terms at:"
echo "     https://huggingface.co/pyannote/speaker-diarization-3.1"
echo ""
echo "   - For LLM refinement with Vertex AI Claude:"
echo "     export ANTHROPIC_VERTEX_PROJECT_ID='your-project-id'"
echo "     export ANTHROPIC_VERTEX_REGION='your-region'"
echo "     gcloud auth application-default login"
echo ""
echo "2. Test the installation:"
echo "   python complete_pipeline.py --help"
echo ""
echo "3. Run a test transcription:"
echo "   python complete_pipeline.py audio.mp3 \\"
echo "     --output results/test \\"
echo "     --hf-token YOUR_HF_TOKEN \\"
echo "     --num-speakers 2 \\"
echo "     --mode parallel"
echo ""
echo "================================================================================================"

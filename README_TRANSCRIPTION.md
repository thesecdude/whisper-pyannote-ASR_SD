# Whisper Transcription with Speaker Diarization

Complete guide for transcribing audio files with speaker identification, optional LLM refinement, and parallel processing.

## Quick Start

### Basic Usage
```bash
python whisper_diarization.py "audio.mp3" \
  --whisper-model turbo \
  --hf-token "$HF_TOKEN"
```

### Multilingual (English-Hindi)
```bash
python whisper_diarization.py "audio.mp3" \
  --whisper-model turbo \
  --hf-token "$HF_TOKEN" \
  --multilingual
```

### Parallel Processing (Faster)
```bash
python parallel_whisper_pyannote.py "audio.mp3" \
  --hf-token "$HF_TOKEN" \
  --num-speakers 3
```

## Features

### 1. LLM Refinement (Enabled by Default)

Automatically improves transcripts using Claude AI via Vertex AI:
- Identifies speakers by name from conversation context
- Fixes grammar and spelling errors
- Adds proper punctuation
- Translates Hindi to English while preserving meaning
- Removes excessive filler words

**Skip refinement:**
```bash
--no-refine
```

**Cost:** ~$0.10-0.30 per hour (billed to `dev-ai-gamma` project)

### 2. Parallel Processing

Run Whisper and Pyannote simultaneously for ~45% speedup:
```bash
python parallel_whisper_pyannote.py "audio.mp3" \
  --hf-token "$HF_TOKEN"
```

**Time savings:**
- 5-minute audio: 45s → 30s (1.5x faster)
- 1-hour audio: 9m → 5m 40s (1.6x faster)

### 3. Speaker Diarization

Automatic speaker identification with configurable speaker count:
```bash
--num-speakers 4              # Exact count
--min-speakers 2 --max-speakers 5  # Range
```

## Setup

### Prerequisites
```bash
pip install anthropic psutil
```

### Required Tokens

**HuggingFace Token:**
1. Get token: https://huggingface.co/settings/tokens
2. Accept pyannote agreement: https://huggingface.co/pyannote/speaker-diarization-3.1
3. Set token:
```bash
export HF_TOKEN="your_token_here"
```

**Vertex AI (for LLM):**
Already configured via `.zshrc`:
```bash
export CLAUDE_CODE_USE_VERTEX=1
export ANTHROPIC_VERTEX_PROJECT_ID=dev-ai-gamma
```

## Output Files

### With LLM Refinement (Default)
```
audio_transcription_raw.json       # Original Whisper+Pyannote
audio_transcription_raw.txt
audio_transcription_raw.srt

audio_transcription_refined.json   # LLM-improved
audio_transcription_refined.txt
audio_transcription_refined.srt
```

### Without LLM (--no-refine)
```
audio_transcription.json
audio_transcription.txt
audio_transcription.srt
```

## Configuration Options

### Whisper Models
```bash
--whisper-model tiny       # Fastest, lowest accuracy
--whisper-model small      # Balanced
--whisper-model turbo      # Recommended (default)
--whisper-model large-v3   # Highest accuracy, slowest
```

### Claude Models
```bash
--claude-model "claude-3-5-sonnet-v2@20241022"  # Default, best balance
--claude-model "claude-3-haiku@20240307"        # Faster, cheaper
--claude-model "claude-3-opus@20240229"         # Highest quality
```

### Processing Options
```bash
--device cuda              # Use GPU (automatic if available)
--device cpu               # Force CPU
--chunk-size 30            # Segments per LLM API call
--multilingual             # Enable multilingual detection
--language en              # Force specific language
```

## Memory Requirements

### CPU Processing (Current Setup)
| Whisper Model | RAM Required | Processing Speed |
|---------------|--------------|------------------|
| tiny          | 3 GB         | 2-3x realtime   |
| small         | 5 GB         | 1-1.5x realtime |
| turbo         | 9 GB         | 0.8-1.2x realtime|
| large-v3      | 17 GB        | 0.3-0.5x realtime|

### GPU Processing (Recommended for Production)
| Whisper Model | VRAM Required | Processing Speed |
|---------------|---------------|------------------|
| tiny          | 3 GB          | 30-40x realtime |
| small         | 4.5 GB        | 20-30x realtime |
| turbo         | 7.5 GB        | 20-30x realtime |
| large-v3      | 13 GB         | 10-15x realtime |

**Recommended GPU:** NVIDIA RTX 4090 (24 GB) or A5000 (24 GB)

## Advanced Usage

### Parallel Processing with Monitoring
```bash
python run_with_monitoring.py "audio.mp3" \
  --hf-token $HF_TOKEN \
  --num-speakers 3
```

Monitors:
- CPU/memory usage
- Parallel worker activity
- Processing timeline
- Time saved vs sequential

### Merge Consecutive Speaker Segments
```bash
python merge_speaker_segments.py output_refined.srt
```

Creates longer, more natural speaker turns with 80%+ reduction in segment count.

### Compare Sequential vs Parallel
```bash
python compare_sequential_vs_parallel.py "audio.mp3" \
  --hf-token $HF_TOKEN
```

### Batch Processing
```bash
# With LLM refinement
for file in results/*/*.mp3; do
  python whisper_diarization.py "$file" \
    --whisper-model turbo \
    --hf-token "$HF_TOKEN"
done

# Without LLM (faster)
for file in results/*/*.mp3; do
  python whisper_diarization.py "$file" \
    --whisper-model turbo \
    --hf-token "$HF_TOKEN" \
    --no-refine
done
```

## Example Comparison

### Raw Output
```
[SPEAKER_00] um so the the context has passed over now
[UNKNOWN] look,
[SPEAKER_01] haan matlab vo cheez ho gayi na
```

### Refined Output
```
[Ishaan] So, the context has passed over now.
[Sahil] Look.
[Person A] Yes, that thing happened, right?
```

## Troubleshooting

### Missing Dependencies
```bash
pip install anthropic psutil
```

### HuggingFace Token Issues
```bash
export HF_TOKEN="your_token_here"
# Verify pyannote agreement: https://huggingface.co/pyannote/speaker-diarization-3.1
```

### Vertex AI Authentication
```bash
gcloud auth application-default login
gcloud config get-value project  # Should show: dev-ai-gamma
```

### Memory Errors
```bash
# Use smaller model
--whisper-model small

# Or force CPU
--device cpu
```

### Slow Processing
```bash
# Use faster Whisper model
--whisper-model small

# Skip LLM refinement
--no-refine

# Use faster Claude model
--claude-model "claude-3-haiku@20240307"
```

## Cost Analysis

### Vertex AI LLM Costs
| Meeting Length | Estimated Cost |
|----------------|----------------|
| 10 minutes     | $0.02 - $0.05  |
| 30 minutes     | $0.05 - $0.15  |
| 1 hour         | $0.10 - $0.30  |
| 2 hours        | $0.20 - $0.60  |

All costs billed to GCP project: `dev-ai-gamma`

### Monitor Usage
1. Visit: https://console.cloud.google.com/vertex-ai/
2. Select project: `dev-ai-gamma`
3. Navigate to: Vertex AI Studio > Usage

## Performance Tips

### Maximum Speed
```bash
python parallel_whisper_pyannote.py audio.mp3 \
  --whisper-model tiny \
  --device cuda \
  --no-refine
```

### Maximum Accuracy
```bash
python parallel_whisper_pyannote.py audio.mp3 \
  --whisper-model large-v3 \
  --device cuda \
  --chunk-size 10
```

### Low Memory
```bash
python whisper_diarization.py audio.mp3 \
  --whisper-model small \
  --device cpu \
  --no-refine
```

## GPU Deployment

For production deployment on company GPUs:

**Optimal Setup:**
- GPU: NVIDIA RTX 4090 (24 GB) or A5000 (24 GB)
- System RAM: 32 GB
- CPU: 8+ cores
- Storage: NVMe SSD

**Expected Performance:**
- 58-minute audio: ~2-3 minutes
- Daily capacity: 200+ hours per GPU
- Speedup: 20-30x faster than CPU

Use `setup_gpu.sh` for automated GPU environment setup.

## Files Reference

### Core Scripts
- `whisper_diarization.py` - Main sequential processing
- `parallel_whisper_pyannote.py` - Parallel implementation
- `run_with_monitoring.py` - Parallel with resource monitoring
- `compare_sequential_vs_parallel.py` - Performance comparison

### Utilities
- `merge_speaker_segments.py` - Merge consecutive segments
- `merge_json_segments.py` - Merge JSON data
- `refine_existing.py` - Apply LLM to existing transcripts
- `visualize_monitoring.py` - Visualize monitoring data

### Setup Scripts
- `setup_gpu.sh` - GPU environment setup
- `check_progress.sh` - Check processing status
- `monitor_memory.sh` - Memory monitoring
- `test_llm_refinement.sh` - Test LLM integration

## API Usage (Python)

```python
from parallel_whisper_pyannote import ParallelWhisperDiarization

# Initialize pipeline
pipeline = ParallelWhisperDiarization(
    whisper_model="turbo",
    device="cuda",
    hf_token="your_token"
)

# Transcribe
result = pipeline.transcribe(
    "audio.mp3",
    num_speakers=3,
    language="en"
)

# Access results
print(f"Speakers: {result['speakers']}")
print(f"Time saved: {result['timing']['time_saved']:.2f}s")

for segment in result['segments']:
    print(f"[{segment['speaker']}] {segment['text']}")
```

## Support

For issues:
1. Check this guide
2. Verify tokens and authentication
3. Review error messages
4. Check memory/disk space
5. Try with smaller audio file first

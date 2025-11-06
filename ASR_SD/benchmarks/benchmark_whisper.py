#!/usr/bin/env python3
"""
Standalone Whisper transcription benchmark.
Only runs Whisper transcription with word-level timestamps.
No diarization, no merging, no LLM processing.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import whisper


def benchmark_whisper(
    audio_path: str,
    whisper_model: str = "turbo",
    language: Optional[str] = None,
    device: Optional[str] = None,
    output: Optional[str] = None,
    **whisper_kwargs
) -> Dict:
    """
    Run Whisper transcription only.

    Args:
        audio_path: Path to audio file
        whisper_model: Whisper model name
        language: Language code or None for auto-detect
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        output: Optional output path for results
        **whisper_kwargs: Additional Whisper arguments

    Returns:
        Dictionary with transcription results and timing
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("WHISPER-ONLY BENCHMARK")
    print("=" * 80)
    print(f"Audio: {audio_path}")
    print(f"Model: {whisper_model}")
    print(f"Device: {device}")
    print("=" * 80)
    print()

    # Load model
    print(f"Loading Whisper model: {whisper_model}...")
    load_start = time.time()
    model = whisper.load_model(whisper_model, device=device)
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.2f}s")
    print()

    # Transcribe
    print("Running transcription...")
    transcribe_start = time.time()

    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        verbose=False,
        **whisper_kwargs
    )

    transcribe_time = time.time() - transcribe_start
    total_time = time.time() - load_start

    print(f"  Transcription completed in {transcribe_time:.2f}s")
    print()

    # Extract statistics
    num_segments = len(result.get("segments", []))
    num_words = sum(len(seg.get("words", [])) for seg in result.get("segments", []))
    detected_language = result.get("language", "unknown")
    full_text = result.get("text", "")

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Language detected: {detected_language}")
    print(f"Segments: {num_segments}")
    print(f"Words: {num_words}")
    print(f"Text length: {len(full_text)} characters")
    print()
    print(f"Model load time: {load_time:.2f}s")
    print(f"Transcription time: {transcribe_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 80)

    # Prepare output
    output_data = {
        "text": full_text,
        "language": detected_language,
        "segments": result.get("segments", []),
        "timing": {
            "model_load_time": load_time,
            "transcription_time": transcribe_time,
            "total_time": total_time
        },
        "stats": {
            "num_segments": num_segments,
            "num_words": num_words,
            "text_length": len(full_text)
        }
    }

    # Save output if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved results to: {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper transcription (standalone, no diarization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark with default model (turbo)
  python benchmark_whisper.py audio.mp3

  # Benchmark with specific model
  python benchmark_whisper.py audio.mp3 --model large-v3

  # Save results
  python benchmark_whisper.py audio.mp3 --output results/whisper_benchmark.json

  # Specify language
  python benchmark_whisper.py audio.mp3 --language en
"""
    )

    parser.add_argument(
        'audio',
        type=str,
        help='Path to audio file'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='turbo',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'turbo'],
        help='Whisper model to use (default: turbo)'
    )

    parser.add_argument(
        '--language',
        type=str,
        default=None,
        help='Language code (e.g., en, es, fr) or None for auto-detect'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use (default: auto-detect)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for results JSON file'
    )

    args = parser.parse_args()

    # Validate audio file
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return 1

    try:
        benchmark_whisper(
            audio_path=args.audio,
            whisper_model=args.model,
            language=args.language,
            device=args.device,
            output=args.output
        )
        return 0
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

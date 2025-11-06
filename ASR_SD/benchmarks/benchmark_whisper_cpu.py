#!/usr/bin/env python3
"""
Standalone Whisper transcription benchmark - CPU only.
Optimized for multi-core CPUs (e.g., 24 cores).
Only runs Whisper transcription with word-level timestamps.
No diarization, no merging, no LLM processing, no GPU.
"""

import argparse
import json
import time
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import whisper


def set_cpu_threads(num_threads: Optional[int] = None):
    """
    Configure CPU thread usage for optimal performance.

    Args:
        num_threads: Number of threads to use. If None, use all available cores.
    """
    if num_threads is None:
        import multiprocessing
        num_threads = multiprocessing.cpu_count()

    # Set environment variables for CPU threading libraries
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    # Configure PyTorch CPU threading
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)

    return num_threads


def benchmark_whisper_cpu(
    audio_path: str,
    whisper_model: str = "turbo",
    language: Optional[str] = None,
    num_threads: Optional[int] = None,
    output: Optional[str] = None,
    **whisper_kwargs
) -> Dict:
    """
    Run Whisper transcription on CPU only.

    Args:
        audio_path: Path to audio file
        whisper_model: Whisper model name
        language: Language code or None for auto-detect
        num_threads: Number of CPU threads (None for all cores)
        output: Optional output path for results
        **whisper_kwargs: Additional Whisper arguments

    Returns:
        Dictionary with transcription results and timing
    """
    import multiprocessing
    total_cores = multiprocessing.cpu_count()

    # Configure CPU threads
    threads_used = set_cpu_threads(num_threads)

    print("=" * 80)
    print("WHISPER-ONLY BENCHMARK (CPU)")
    print("=" * 80)
    print(f"Audio: {audio_path}")
    print(f"Model: {whisper_model}")
    print(f"Device: CPU only")
    print(f"Total CPU cores: {total_cores}")
    print(f"Threads used: {threads_used}")
    print("=" * 80)
    print()

    # Load model on CPU
    print(f"Loading Whisper model: {whisper_model}...")
    load_start = time.time()
    model = whisper.load_model(whisper_model, device="cpu")
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
    print(f"CPU threads: {threads_used}")
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
        },
        "system": {
            "device": "cpu",
            "total_cores": total_cores,
            "threads_used": threads_used
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
        description="Benchmark Whisper transcription on CPU (no GPU, standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark with all CPU cores (auto-detect)
  python benchmark_whisper_cpu.py audio.mp3

  # Benchmark with specific number of threads
  python benchmark_whisper_cpu.py audio.mp3 --threads 16

  # Benchmark with specific model
  python benchmark_whisper_cpu.py audio.mp3 --model large-v3 --threads 24

  # Save results
  python benchmark_whisper_cpu.py audio.mp3 --output results/whisper_cpu_bench.json

  # Specify language
  python benchmark_whisper_cpu.py audio.mp3 --language en --threads 12
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
        '--threads',
        type=int,
        default=None,
        help='Number of CPU threads to use (default: all available cores)'
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
        benchmark_whisper_cpu(
            audio_path=args.audio,
            whisper_model=args.model,
            language=args.language,
            num_threads=args.threads,
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

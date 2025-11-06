#!/usr/bin/env python3
"""
Standalone Pyannote diarization benchmark - CPU only.
Optimized for multi-core CPUs (e.g., 24 cores).
Only runs speaker diarization.
No transcription, no merging, no LLM processing, no GPU.
"""

import argparse
import json
import time
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from pyannote.audio import Pipeline


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


def benchmark_pyannote_cpu(
    audio_path: str,
    diarization_model: str = "pyannote/speaker-diarization-3.1",
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_threads: Optional[int] = None,
    output: Optional[str] = None
) -> Dict:
    """
    Run Pyannote speaker diarization on CPU only.

    Args:
        audio_path: Path to audio file
        diarization_model: HuggingFace model for diarization
        hf_token: HuggingFace token
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        num_threads: Number of CPU threads (None for all cores)
        output: Optional output path for results

    Returns:
        Dictionary with diarization results and timing
    """
    import multiprocessing
    total_cores = multiprocessing.cpu_count()

    # Configure CPU threads
    threads_used = set_cpu_threads(num_threads)

    print("=" * 80)
    print("PYANNOTE-ONLY BENCHMARK (CPU)")
    print("=" * 80)
    print(f"Audio: {audio_path}")
    print(f"Model: {diarization_model}")
    print(f"Device: CPU only")
    print(f"Total CPU cores: {total_cores}")
    print(f"Threads used: {threads_used}")
    print("=" * 80)
    print()

    # Load model
    print(f"Loading Pyannote model: {diarization_model}...")
    load_start = time.time()

    if hf_token:
        pipeline = Pipeline.from_pretrained(diarization_model, use_auth_token=hf_token)
    else:
        try:
            pipeline = Pipeline.from_pretrained(diarization_model)
        except Exception as e:
            print("\nError: Pyannote models require HuggingFace authentication.")
            print("Please provide a HuggingFace token using --hf-token")
            print("Get your token from: https://huggingface.co/settings/tokens")
            print("Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            raise e

    # Move to CPU (ensure no GPU usage)
    try:
        pipeline.to(torch.device("cpu"))
    except Exception:
        pass

    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.2f}s")
    print()

    # Prepare diarization parameters
    diarization_params = {}
    if num_speakers is not None:
        diarization_params["num_speakers"] = num_speakers
        print(f"Configured for {num_speakers} speakers (exact)")
    elif min_speakers is not None or max_speakers is not None:
        if min_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_params["max_speakers"] = max_speakers
        print(f"Configured for {min_speakers}-{max_speakers} speakers (range)")
    else:
        print("Auto-detecting number of speakers")
    print()

    # Run diarization
    print("Running speaker diarization...")
    diarize_start = time.time()

    diarization = pipeline(audio_path, **diarization_params)

    diarize_time = time.time() - diarize_start
    total_time = time.time() - load_start

    print(f"  Diarization completed in {diarize_time:.2f}s")
    print()

    # Extract statistics
    speakers = set()
    segments = []

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "duration": segment.end - segment.start,
            "speaker": speaker
        })

    num_speakers_detected = len(speakers)
    num_segments = len(segments)
    total_speech_time = sum(seg["duration"] for seg in segments)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Speakers detected: {num_speakers_detected}")
    print(f"Speaker labels: {', '.join(sorted(speakers))}")
    print(f"Speaker segments: {num_segments}")
    print(f"Total speech time: {total_speech_time:.2f}s")
    print()
    print(f"Model load time: {load_time:.2f}s")
    print(f"Diarization time: {diarize_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"CPU threads: {threads_used}")
    print("=" * 80)

    # Prepare output
    output_data = {
        "speakers": sorted(list(speakers)),
        "num_speakers": num_speakers_detected,
        "segments": segments,
        "timing": {
            "model_load_time": load_time,
            "diarization_time": diarize_time,
            "total_time": total_time
        },
        "stats": {
            "num_segments": num_segments,
            "total_speech_time": total_speech_time
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
        description="Benchmark Pyannote speaker diarization on CPU (no GPU, standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark with all CPU cores (auto-detect)
  python benchmark_pyannote_cpu.py audio.mp3 --hf-token YOUR_TOKEN

  # Benchmark with specific number of threads
  python benchmark_pyannote_cpu.py audio.mp3 --hf-token YOUR_TOKEN --threads 16

  # Benchmark with known number of speakers
  python benchmark_pyannote_cpu.py audio.mp3 --hf-token YOUR_TOKEN --num-speakers 3 --threads 24

  # Save results
  python benchmark_pyannote_cpu.py audio.mp3 --hf-token YOUR_TOKEN --output results/pyannote_cpu_bench.json

  # Specify speaker range
  python benchmark_pyannote_cpu.py audio.mp3 --hf-token YOUR_TOKEN --min-speakers 2 --max-speakers 5 --threads 12
"""
    )

    parser.add_argument(
        'audio',
        type=str,
        help='Path to audio file'
    )

    parser.add_argument(
        '--hf-token',
        type=str,
        required=True,
        help='HuggingFace token for Pyannote models'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='pyannote/speaker-diarization-3.1',
        help='Diarization model (default: pyannote/speaker-diarization-3.1)'
    )

    # Speaker detection options
    speaker_group = parser.add_mutually_exclusive_group()
    speaker_group.add_argument(
        '--num-speakers',
        type=int,
        help='Exact number of speakers'
    )

    parser.add_argument(
        '--min-speakers',
        type=int,
        help='Minimum number of speakers (use with --max-speakers)'
    )

    parser.add_argument(
        '--max-speakers',
        type=int,
        help='Maximum number of speakers (use with --min-speakers)'
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

    # Validate arguments
    if args.min_speakers and not args.max_speakers:
        parser.error("--min-speakers requires --max-speakers")
    if args.max_speakers and not args.min_speakers:
        parser.error("--max-speakers requires --min-speakers")

    # Validate audio file
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return 1

    try:
        benchmark_pyannote_cpu(
            audio_path=args.audio,
            diarization_model=args.model,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
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

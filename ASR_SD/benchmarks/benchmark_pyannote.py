import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from pyannote.audio import Pipeline


def benchmark_pyannote(
    audio_path: str,
    diarization_model: str = "pyannote/speaker-diarization-3.1",
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: Optional[str] = None,
    output: Optional[str] = None
) -> Dict:
    """
    Run Pyannote speaker diarization only.
    Args:
        audio_path: Path to audio file
        diarization_model: HuggingFace model for diarization
        hf_token: HuggingFace token
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        output: Optional output path for results
    Returns:
        Dictionary with diarization results and timing
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("PYANNOTE-ONLY BENCHMARK")
    print("=" * 60)
    print(f"Audio: {audio_path}")
    print(f"Model: {diarization_model}")
    print(f"Device: {device}")
    print("=" * 60)
    print()

    # Load model
    print(f"Loading Pyannote model: {diarization_model}...")
    load_start = time.time()

    if hf_token:
        pipeline = Pipeline.from_pretrained(diarization_model, token=hf_token)
    else:
        try:
            pipeline = Pipeline.from_pretrained(diarization_model)
        except Exception as e:
            print("\nError: Pyannote models require HuggingFace authentication.")
            print("Please provide a HuggingFace token using --hf-token")
            raise e

    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        try:
            pipeline.to(torch.device("cuda"))
        except Exception:
            pass

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
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

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Speakers detected: {num_speakers_detected}")
    print(f"Speaker labels: {', '.join(sorted(speakers))}")
    print(f"Speaker segments: {num_segments}")
    print(f"Total speech time: {total_speech_time:.2f}s")
    print()
    print(f"Model load time: {load_time:.2f}s")
    print(f"Diarization time: {diarize_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 60)

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
        description="Benchmark Pyannote speaker diarization (standalone, no transcription)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark with default model
  python benchmark_pyannote.py audio.mp3 --hf-token YOUR_TOKEN

  # Benchmark with known number of speakers
  python benchmark_pyannote.py audio.mp3 --hf-token YOUR_TOKEN --num-speakers 3

  # Save results
  python benchmark_pyannote.py audio.mp3 --hf-token YOUR_TOKEN --output results/pyannote_benchmark.json

  # Specify speaker range
  python benchmark_pyannote.py audio.mp3 --hf-token YOUR_TOKEN --min-speakers 2 --max-speakers 5
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
        benchmark_pyannote(
            audio_path=args.audio,
            diarization_model=args.model,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
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

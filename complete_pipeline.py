#!/usr/bin/env python3
"""
Complete transcription pipeline that:
1. Runs both parallel and sequential processing
2. Merges consecutive speaker segments
3. Sends to LLM for refinement
4. Re-merges speaker segments after LLM processing
"""

import argparse
import json
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:
    print("Warning: psutil not installed. Resource monitoring will be disabled.")
    print("Install with: pip install psutil")
    psutil = None

try:
    import GPUtil
except ImportError:
    print("Warning: GPUtil not installed. GPU monitoring will be disabled.")
    print("Install with: pip install gputil")
    GPUtil = None

# Import our existing modules
from whisper_pyannote import WhisperPyannoteTranscriber
from parallel_whisper_pyannote import ParallelWhisperPyannoteTranscriber
from merge_json_segments import merge_consecutive_speakers


class ResourceMonitor:
    """Monitor CPU, GPU, and RAM usage during processing."""

    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None

        # Storage for metrics
        self.cpu_samples = []
        self.ram_samples = []
        self.gpu_util_samples = []
        self.gpu_mem_samples = []

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            # CPU and RAM
            if psutil:
                self.cpu_samples.append(psutil.cpu_percent(interval=0.5))
                ram = psutil.virtual_memory()
                self.ram_samples.append(ram.percent)

            # GPU
            if GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        # Average across all GPUs
                        avg_util = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
                        avg_mem = sum(gpu.memoryUtil * 100 for gpu in gpus) / len(gpus)
                        self.gpu_util_samples.append(avg_util)
                        self.gpu_mem_samples.append(avg_mem)
                except:
                    pass

            time.sleep(1.0)  # Sample every second

    def start(self):
        """Start monitoring."""
        if not psutil and not GPUtil:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        stats = {}

        if self.cpu_samples:
            stats['cpu'] = {
                'avg': sum(self.cpu_samples) / len(self.cpu_samples),
                'max': max(self.cpu_samples),
                'samples': len(self.cpu_samples)
            }

        if self.ram_samples:
            stats['ram'] = {
                'avg': sum(self.ram_samples) / len(self.ram_samples),
                'max': max(self.ram_samples),
                'samples': len(self.ram_samples)
            }

        if self.gpu_util_samples:
            stats['gpu_util'] = {
                'avg': sum(self.gpu_util_samples) / len(self.gpu_util_samples),
                'max': max(self.gpu_util_samples),
                'samples': len(self.gpu_util_samples)
            }

        if self.gpu_mem_samples:
            stats['gpu_mem'] = {
                'avg': sum(self.gpu_mem_samples) / len(self.gpu_mem_samples),
                'max': max(self.gpu_mem_samples),
                'samples': len(self.gpu_mem_samples)
            }

        return stats


def merge_segments_in_data(data: Dict) -> Dict:
    """
    Merge consecutive speaker segments in transcript data.

    Args:
        data: Transcript data with 'segments' field

    Returns:
        Data with merged segments
    """
    if 'segments' not in data:
        print("Warning: No segments field found in data")
        return data

    original_count = len(data['segments'])
    merged_segments = merge_consecutive_speakers(data['segments'])
    merged_count = len(merged_segments)

    reduction = original_count - merged_count
    print(f"  Merged: {original_count} → {merged_count} segments ({reduction} merged, {reduction/original_count*100:.1f}% reduction)")

    data['segments'] = merged_segments
    return data


def run_pipeline(
    audio_path: str,
    output_base: str,
    hf_token: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    whisper_model: str = "large-v2",
    mode: str = "both",  # "parallel", "sequential", or "both"
    max_tokens: int = 200000,
    skip_llm: bool = False
) -> Dict[str, Path]:
    """
    Run complete transcription pipeline.

    Args:
        audio_path: Path to audio file
        output_base: Base path for output files (without extension)
        hf_token: HuggingFace token for Pyannote
        num_speakers: Exact number of speakers (optional)
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
        whisper_model: Whisper model to use
        mode: Processing mode - "parallel", "sequential", or "both"
        max_tokens: Maximum tokens per LLM chunk
        skip_llm: Skip LLM refinement step

    Returns:
        Dictionary mapping output type to file path
    """
    audio_path = Path(audio_path)
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    resource_stats = {}

    print("="*80)
    print("COMPLETE TRANSCRIPTION PIPELINE")
    print("="*80)
    print(f"Audio: {audio_path}")
    print(f"Output base: {output_base}")
    print(f"Mode: {mode}")
    print(f"Whisper model: {whisper_model}")
    print(f"LLM refinement: {'Disabled' if skip_llm else 'Enabled'}")
    print("="*80)
    print()

    # Step 1: Run Whisper + Pyannote processing
    if mode in ["parallel", "both"]:
        print("\n" + "="*80)
        print("STEP 1a: PARALLEL PROCESSING (Whisper + Pyannote)")
        print("="*80)

        # Start resource monitoring
        monitor = ResourceMonitor()
        monitor.start()

        start_time = time.time()
        transcriber = ParallelWhisperPyannoteTranscriber(
            whisper_model=whisper_model,
            hf_token=hf_token
        )

        result = transcriber.transcribe(
            audio_path=str(audio_path),
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        parallel_time = time.time() - start_time

        # Stop monitoring and get stats
        resource_stats['parallel'] = monitor.stop()

        # Save raw parallel output
        raw_parallel_path = output_base.parent / f"{output_base.stem}_parallel_raw.json"
        with open(raw_parallel_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        results['parallel_raw'] = raw_parallel_path

        print(f"\n✓ Parallel processing completed in {parallel_time:.1f}s")
        print(f"  Saved: {raw_parallel_path}")

    if mode in ["sequential", "both"]:
        print("\n" + "="*80)
        print("STEP 1b: SEQUENTIAL PROCESSING (Whisper + Pyannote)")
        print("="*80)

        # Start resource monitoring
        monitor = ResourceMonitor()
        monitor.start()

        start_time = time.time()
        transcriber = WhisperPyannoteTranscriber(
            whisper_model=whisper_model,
            hf_token=hf_token
        )

        result = transcriber.transcribe(
            audio_path=str(audio_path),
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        sequential_time = time.time() - start_time

        # Stop monitoring and get stats
        resource_stats['sequential'] = monitor.stop()

        # Save raw sequential output
        raw_sequential_path = output_base.parent / f"{output_base.stem}_sequential_raw.json"
        with open(raw_sequential_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        results['sequential_raw'] = raw_sequential_path

        print(f"\n✓ Sequential processing completed in {sequential_time:.1f}s")
        print(f"  Saved: {raw_sequential_path}")

    # Use the result from the preferred mode for further processing
    if mode == "parallel":
        data = result
        processing_mode = "parallel"
    elif mode == "sequential":
        data = result
        processing_mode = "sequential"
    else:  # both - use parallel as it's faster
        data = result
        processing_mode = "parallel (from both)"

    # Step 2: First merge of consecutive speaker segments
    print("\n" + "="*80)
    print("STEP 2: MERGE CONSECUTIVE SPEAKER SEGMENTS (PRE-LLM)")
    print("="*80)

    data = merge_segments_in_data(data)

    # Save merged output
    merged_path = output_base.parent / f"{output_base.stem}_merged.json"
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    results['merged'] = merged_path

    print(f"  Saved: {merged_path}")

    # Step 3: LLM refinement (optional)
    if not skip_llm:
        print("\n" + "="*80)
        print("STEP 3: LLM REFINEMENT")
        print("="*80)

        # Use the appropriate transcriber for refinement
        if processing_mode.startswith("parallel"):
            transcriber = ParallelWhisperPyannoteTranscriber(
                whisper_model=whisper_model,
                hf_token=hf_token
            )
        else:
            transcriber = WhisperPyannoteTranscriber(
                whisper_model=whisper_model,
                hf_token=hf_token
            )

        # Run refinement
        refined_data = transcriber.refine_transcript(
            segments=data['segments'],
            max_tokens=max_tokens
        )

        # Save LLM-refined output
        llm_refined_path = output_base.parent / f"{output_base.stem}_llm_refined.json"
        with open(llm_refined_path, 'w', encoding='utf-8') as f:
            json.dump(refined_data, f, indent=2, ensure_ascii=False)
        results['llm_refined'] = llm_refined_path

        print(f"  Saved: {llm_refined_path}")

        # Step 4: Re-merge after LLM processing
        print("\n" + "="*80)
        print("STEP 4: RE-MERGE SPEAKER SEGMENTS (POST-LLM)")
        print("="*80)

        refined_data = merge_segments_in_data(refined_data)

        # Save final output
        final_path = output_base.parent / f"{output_base.stem}_final.json"
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(refined_data, f, indent=2, ensure_ascii=False)
        results['final'] = final_path

        print(f"  Saved: {final_path}")
    else:
        # If skipping LLM, merged is the final output
        results['final'] = merged_path

    # Print summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)

    # Processing time statistics
    if mode == "both":
        print(f"\nProcessing times:")
        print(f"  Parallel: {parallel_time:.1f}s")
        print(f"  Sequential: {sequential_time:.1f}s")
        print(f"  Time saved: {sequential_time - parallel_time:.1f}s ({(sequential_time - parallel_time)/sequential_time*100:.1f}%)")
    elif mode == "parallel":
        print(f"\nProcessing time: {parallel_time:.1f}s (parallel mode)")
    elif mode == "sequential":
        print(f"\nProcessing time: {sequential_time:.1f}s (sequential mode)")

    # Resource utilization statistics
    if resource_stats:
        print(f"\nResource Utilization:")

        for mode_name, stats in resource_stats.items():
            print(f"\n  {mode_name.upper()} mode:")

            if 'cpu' in stats:
                print(f"    CPU: avg {stats['cpu']['avg']:.1f}%, max {stats['cpu']['max']:.1f}%")

            if 'ram' in stats:
                print(f"    RAM: avg {stats['ram']['avg']:.1f}%, max {stats['ram']['max']:.1f}%")

            if 'gpu_util' in stats:
                print(f"    GPU Utilization: avg {stats['gpu_util']['avg']:.1f}%, max {stats['gpu_util']['max']:.1f}%")

            if 'gpu_mem' in stats:
                print(f"    GPU Memory: avg {stats['gpu_mem']['avg']:.1f}%, max {stats['gpu_mem']['max']:.1f}%")

    print(f"\nOutput files:")
    for output_type, path in results.items():
        print(f"  {output_type}: {path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Complete transcription pipeline with parallel/sequential processing, speaker merging, and LLM refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with parallel processing only
  python complete_pipeline.py audio.mp3 --output results/transcript --hf-token YOUR_TOKEN --num-speakers 3

  # Run both parallel and sequential for comparison
  python complete_pipeline.py audio.mp3 --output results/transcript --hf-token YOUR_TOKEN --mode both

  # Skip LLM refinement
  python complete_pipeline.py audio.mp3 --output results/transcript --hf-token YOUR_TOKEN --skip-llm

  # Use sequential processing only
  python complete_pipeline.py audio.mp3 --output results/transcript --hf-token YOUR_TOKEN --mode sequential
"""
    )

    parser.add_argument(
        'audio_path',
        type=str,
        help='Path to audio file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Base path for output files (without extension)'
    )

    parser.add_argument(
        '--hf-token',
        type=str,
        required=True,
        help='HuggingFace token for Pyannote models'
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

    # Model options
    parser.add_argument(
        '--whisper-model',
        type=str,
        default='large-v2',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help='Whisper model to use (default: large-v2)'
    )

    # Processing mode
    parser.add_argument(
        '--mode',
        type=str,
        default='parallel',
        choices=['parallel', 'sequential', 'both'],
        help='Processing mode: parallel (default), sequential, or both for comparison'
    )

    # LLM options
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=200000,
        help='Maximum tokens per LLM chunk (default: 200000 for ~256k context)'
    )

    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Skip LLM refinement step'
    )

    args = parser.parse_args()

    # Validate speaker arguments
    if args.min_speakers and not args.max_speakers:
        parser.error("--min-speakers requires --max-speakers")
    if args.max_speakers and not args.min_speakers:
        parser.error("--max-speakers requires --min-speakers")

    # Validate audio file
    if not Path(args.audio_path).exists():
        print(f"Error: Audio file not found: {args.audio_path}")
        sys.exit(1)

    # Run pipeline
    try:
        results = run_pipeline(
            audio_path=args.audio_path,
            output_base=args.output,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            whisper_model=args.whisper_model,
            mode=args.mode,
            max_tokens=args.max_tokens,
            skip_llm=args.skip_llm
        )

        print("\n✓ Pipeline completed successfully!")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

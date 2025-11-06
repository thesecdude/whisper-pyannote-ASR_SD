#!/usr/bin/env python3
"""
Compare Sequential vs Parallel Processing Performance

This script runs both the sequential and parallel versions of the
Whisper + Pyannote pipeline and compares their performance.

Usage:
    python compare_sequential_vs_parallel.py <audio_file> [options]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Import both implementations
from whisper_pyannote import WhisperDiarization as SequentialWhisperDiarization
from parallel_whisper_pyannote import ParallelWhisperDiarization


def compare_pipelines(
    audio_path: str,
    whisper_model: str = "turbo",
    diarization_model: str = "pyannote/speaker-diarization-3.1",
    device: str = None,
    hf_token: str = None,
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
    language: str = None,
    multilingual: bool = False
):
    """
    Run both sequential and parallel pipelines and compare results.
    """
    print("="*80)
    print("SEQUENTIAL vs PARALLEL PROCESSING COMPARISON")
    print("="*80)
    print(f"\nAudio file: {audio_path}")
    print(f"Whisper model: {whisper_model}")
    print(f"Diarization model: {diarization_model}")
    print(f"Device: {device or 'auto'}")

    results = {}

    # ========================================================================
    # Test 1: Sequential Processing (Original Implementation)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: SEQUENTIAL PROCESSING")
    print("="*80)

    try:
        seq_pipeline = SequentialWhisperDiarization(
            whisper_model=whisper_model,
            diarization_model=diarization_model,
            device=device,
            hf_token=hf_token
        )

        seq_start = time.time()
        seq_result = seq_pipeline.transcribe(
            audio_path,
            language=language,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            multilingual=multilingual
        )
        seq_total_time = time.time() - seq_start

        results['sequential'] = {
            'result': seq_result,
            'total_time': seq_total_time,
            'num_segments': len(seq_result['segments']),
            'num_speakers': len(seq_result['speakers']),
            'speakers': seq_result['speakers']
        }

        print(f"\n✓ Sequential processing completed in {seq_total_time:.2f}s")
        print(f"  Segments: {len(seq_result['segments'])}")
        print(f"  Speakers: {', '.join(seq_result['speakers'])}")

    except Exception as e:
        print(f"\n✗ Sequential processing failed: {e}")
        results['sequential'] = {'error': str(e)}

    # ========================================================================
    # Test 2: Parallel Processing (New Implementation)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: PARALLEL PROCESSING")
    print("="*80)

    try:
        par_pipeline = ParallelWhisperDiarization(
            whisper_model=whisper_model,
            diarization_model=diarization_model,
            device=device,
            hf_token=hf_token
        )

        par_start = time.time()
        par_result = par_pipeline.transcribe(
            audio_path,
            language=language,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            multilingual=multilingual
        )
        par_total_time = time.time() - par_start

        results['parallel'] = {
            'result': par_result,
            'total_time': par_total_time,
            'num_segments': len(par_result['segments']),
            'num_speakers': len(par_result['speakers']),
            'speakers': par_result['speakers'],
            'timing_breakdown': par_result.get('timing', {})
        }

        print(f"\n✓ Parallel processing completed in {par_total_time:.2f}s")
        print(f"  Segments: {len(par_result['segments'])}")
        print(f"  Speakers: {', '.join(par_result['speakers'])}")

    except Exception as e:
        print(f"\n✗ Parallel processing failed: {e}")
        results['parallel'] = {'error': str(e)}

    # ========================================================================
    # Comparison & Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    if 'error' not in results.get('sequential', {}) and 'error' not in results.get('parallel', {}):
        seq_time = results['sequential']['total_time']
        par_time = results['parallel']['total_time']
        speedup = seq_time / par_time
        time_saved = seq_time - par_time

        print(f"\n  PERFORMANCE:")
        print(f"  Sequential time:    {seq_time:.2f}s")
        print(f"  Parallel time:      {par_time:.2f}s")
        print(f"  Time saved:         {time_saved:.2f}s ({time_saved/seq_time*100:.1f}%)")
        print(f"  Speedup factor:     {speedup:.2f}x")

        # Detailed timing breakdown from parallel
        if 'timing_breakdown' in results['parallel']:
            timing = results['parallel']['timing_breakdown']
            print(f"\n PARALLEL BREAKDOWN:")
            print(f"  Whisper alone:      {timing.get('whisper_time', 0):.2f}s")
            print(f"  Pyannote alone:     {timing.get('diarization_time', 0):.2f}s")
            print(f"  Parallel execution: {timing.get('parallel_time', 0):.2f}s")
            print(f"  Merge time:         {timing.get('merge_time', 0):.2f}s")

        print(f"\n ACCURACY:")
        print(f"  Sequential segments: {results['sequential']['num_segments']}")
        print(f"  Parallel segments:   {results['parallel']['num_segments']}")
        print(f"  Segment difference:  {abs(results['sequential']['num_segments'] - results['parallel']['num_segments'])}")

        print(f"\n SPEAKERS:")
        print(f"  Sequential: {', '.join(results['sequential']['speakers'])}")
        print(f"  Parallel:   {', '.join(results['parallel']['speakers'])}")

        # Check if results are similar
        seg_diff = abs(results['sequential']['num_segments'] - results['parallel']['num_segments'])
        spk_diff = abs(results['sequential']['num_speakers'] - results['parallel']['num_speakers'])

        print(f"\n VALIDATION:")
        if seg_diff <= results['sequential']['num_segments'] * 0.1:  # Within 10%
            print(f"  ✓ Segment counts are similar (diff: {seg_diff})")
        else:
            print(f"  ⚠ Segment counts differ significantly (diff: {seg_diff})")

        if spk_diff == 0:
            print(f"  ✓ Speaker counts match exactly")
        else:
            print(f"  ⚠ Speaker counts differ (diff: {spk_diff})")

        print(f"\n RECOMMENDATION:")
        if speedup >= 1.3:
            print(f"  Use PARALLEL processing for {speedup:.1f}x speedup!")
        elif speedup >= 1.1:
            print(f"  Parallel gives modest {speedup:.1f}x speedup")
        else:
            print(f"  Speedup is minimal, either method works")

    else:
        print("\n  Could not compare - one or both pipelines failed")
        if 'error' in results.get('sequential', {}):
            print(f"  Sequential error: {results['sequential']['error']}")
        if 'error' in results.get('parallel', {}):
            print(f"  Parallel error: {results['parallel']['error']}")

    # ========================================================================
    # Save comparison report
    # ========================================================================
    audio_file = Path(audio_path)
    report_path = audio_file.parent / f"{audio_file.stem}_comparison_report.json"

    # Prepare serializable results
    save_results = {
        'audio_file': str(audio_path),
        'sequential': {
            'total_time': results['sequential'].get('total_time'),
            'num_segments': results['sequential'].get('num_segments'),
            'num_speakers': results['sequential'].get('num_speakers'),
            'speakers': results['sequential'].get('speakers'),
            'error': results['sequential'].get('error')
        },
        'parallel': {
            'total_time': results['parallel'].get('total_time'),
            'num_segments': results['parallel'].get('num_segments'),
            'num_speakers': results['parallel'].get('num_speakers'),
            'speakers': results['parallel'].get('speakers'),
            'timing_breakdown': results['parallel'].get('timing_breakdown'),
            'error': results['parallel'].get('error')
        }
    }

    if 'error' not in results.get('sequential', {}) and 'error' not in results.get('parallel', {}):
        save_results['comparison'] = {
            'speedup': speedup,
            'time_saved': time_saved,
            'time_saved_percent': time_saved/seq_time*100
        }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    print(f"\n Comparison report saved to: {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare sequential vs parallel Whisper + Pyannote processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--whisper-model", type=str, default="turbo",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
                        help="Whisper model size")
    parser.add_argument("--diarization-model", type=str, default="pyannote/speaker-diarization-3.1",
                        help="HuggingFace diarization model")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for pyannote models")
    parser.add_argument("--language", type=str, default=None,
                        help="Language code or None for auto-detect")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Exact number of speakers")
    parser.add_argument("--min-speakers", type=int, default=None,
                        help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="Maximum number of speakers")
    parser.add_argument("--multilingual", action="store_true",
                        help="Enable multilingual mode")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="Device to use for inference")

    args = parser.parse_args()

    compare_pipelines(
        audio_path=args.audio,
        whisper_model=args.whisper_model,
        diarization_model=args.diarization_model,
        device=args.device,
        hf_token=args.hf_token,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        language=args.language,
        multilingual=args.multilingual
    )


if __name__ == "__main__":
    main()

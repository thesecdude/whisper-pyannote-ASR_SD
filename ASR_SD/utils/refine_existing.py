#!/usr/bin/env python3
"""
Refine an existing transcript JSON file using LLM without re-running Whisper+pyannote.

Usage:
    python refine_existing.py input.json --output refined_output

This takes a raw transcript JSON (from whisper_diarization.py) and applies
LLM refinement to it, saving the refined version without redoing transcription.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Import the TranscriptRefiner class from whisper_pyannote (which has the better prompt)
try:
    # Try importing from whisper_pyannote first (newer version)
    from whisper_pyannote import TranscriptRefiner, save_results
    print("Using TranscriptRefiner from whisper_pyannote.py (improved version)")
except ImportError:
    # Fall back to whisper_diarization
    try:
        from whisper_diarization import TranscriptRefiner, save_results
        print("Using TranscriptRefiner from whisper_diarization.py")
    except ImportError:
        print("Error: Could not import TranscriptRefiner. Make sure whisper_pyannote.py or whisper_diarization.py is in the same directory.")
        sys.exit(1)


def load_transcript(json_path: str) -> Dict:
    """Load a transcript JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Refine an existing transcript using LLM (skip Whisper+pyannote)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input_json",
        type=str,
        help="Path to input transcript JSON file (e.g., transcript_raw.json)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (without extension). If not provided, uses input filename with _refined suffix"
    )

    parser.add_argument(
        "--vertex-project-id",
        type=str,
        default=None,
        help="Google Cloud project ID for Vertex AI (uses ANTHROPIC_VERTEX_PROJECT_ID env var if not provided)"
    )

    parser.add_argument(
        "--vertex-region",
        type=str,
        default="us-east5",
        help="Vertex AI region"
    )

    parser.add_argument(
        "--claude-model",
        type=str,
        default="claude-3-5-sonnet-v2@20241022",
        help="Claude model to use for refinement via Vertex AI"
    )

    parser.add_argument(
        "--refinement-prompt",
        type=str,
        default=None,
        help="Custom prompt for LLM refinement (uses default if not provided)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Number of segments to process at once during LLM refinement"
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="all",
        choices=["txt", "json", "srt", "all"],
        help="Output format"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_json):
        print(f"Error: Input file not found: {args.input_json}")
        sys.exit(1)

    # Determine output path
    if args.output is None:
        input_path = Path(args.input_json)
        # Remove _raw suffix if present
        stem = input_path.stem.replace("_raw", "")
        output_path = input_path.parent / stem
    else:
        output_path = args.output

    print(f"\n{'='*60}")
    print("LLM-ONLY REFINEMENT (No Whisper/Pyannote Re-processing)")
    print(f"{'='*60}")
    print(f"Input:  {args.input_json}")
    print(f"Output: {output_path}_refined.*")
    print(f"{'='*60}\n")

    # Load the raw transcript
    print("Loading transcript...")
    try:
        result = load_transcript(args.input_json)
    except Exception as e:
        print(f"Error loading transcript: {e}")
        sys.exit(1)

    # Validate the transcript structure
    required_keys = ["segments", "word_segments", "language", "speakers"]
    missing_keys = [k for k in required_keys if k not in result]
    if missing_keys:
        print(f"Warning: Transcript is missing expected keys: {missing_keys}")
        print("This may not be a valid whisper_diarization.py output file.")

        # Try to proceed anyway if we at least have segments
        if "segments" not in result:
            print("Error: No 'segments' key found. Cannot proceed.")
            sys.exit(1)

    print(f"✓ Loaded transcript with {len(result.get('segments', []))} segments")
    print(f"  Language: {result.get('language', 'unknown')}")
    print(f"  Speakers: {len(result.get('speakers', []))} detected")

    # Initialize LLM refiner
    print("\nInitializing Claude AI (Vertex AI)...")
    try:
        refiner = TranscriptRefiner(
            project_id=args.vertex_project_id,
            region=args.vertex_region,
            model=args.claude_model
        )
    except Exception as e:
        print(f"\nError initializing LLM refiner: {e}")
        print("\nMake sure:")
        print("  1. You have google-auth installed: pip install 'anthropic[vertex]'")
        print("  2. ANTHROPIC_VERTEX_PROJECT_ID env var is set")
        print("  3. You're authenticated with Google Cloud: gcloud auth application-default login")
        sys.exit(1)

    # Apply LLM refinement
    print(f"\nRefining transcript with {args.claude_model}...")
    try:
        refined_result = refiner.refine_transcript(
            result,
            chunk_size=args.chunk_size,
            custom_prompt=args.refinement_prompt
        )
    except Exception as e:
        print(f"\nError during LLM refinement: {e}")
        print("Refinement failed. See error above for details.")
        sys.exit(1)

    # Save refined results
    print(f"\nSaving refined transcript to: {output_path}_refined.*")
    save_results(refined_result, str(output_path), args.format, suffix="_refined")

    # Print summary
    print("\n" + "="*60)
    print("REFINEMENT COMPLETE")
    print("="*60)
    print(f"Speakers identified: {', '.join(refined_result['speakers'])}")
    print(f"Total segments: {len(refined_result['segments'])}")

    if refined_result.get('speaker_mapping'):
        print("\nSpeaker mappings applied:")
        for old, new in refined_result['speaker_mapping'].items():
            print(f"  {old} → {new}")

    print(f"\nOutput files:")
    output_base = Path(output_path)
    for ext in ['json', 'txt', 'srt']:
        file_path = output_base.parent / f"{output_base.stem}_refined.{ext}"
        if file_path.exists():
            print(f"  ✓ {file_path}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()

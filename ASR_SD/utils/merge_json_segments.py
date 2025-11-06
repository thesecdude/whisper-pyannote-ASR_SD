#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


def merge_consecutive_speakers(segments):
    if not segments:
        return []

    merged = []
    current = None

    for segment in segments:
        speaker = segment.get('speaker', 'UNKNOWN')

        # Start new merged segment or continue current one
        if current is None:
            # First segment
            current = {
                'speaker': speaker,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            }
        elif current['speaker'] == speaker:
            # Same speaker - merge into current
            current['end'] = segment['end']
            current['text'] = current['text'] + ' ' + segment['text']
        else:
            # Different speaker - save current and start new
            merged.append(current)
            current = {
                'speaker': speaker,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            }

    # Don't forget the last segment
    if current is not None:
        merged.append(current)

    return merged


def merge_json_file(input_path, output_path=None):
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_merged.json"
    else:
        output_path = Path(output_path)

    # Read input JSON
    print(f"Reading: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get segments
    if 'segments' not in data:
        print(f"Error: No 'segments' field found in {input_path}")
        return None

    original_segments = data['segments']
    original_count = len(original_segments)

    print(f"Original segments: {original_count}")

    # Merge consecutive speakers
    merged_segments = merge_consecutive_speakers(original_segments)
    merged_count = len(merged_segments)

    # Update data with merged segments
    data['segments'] = merged_segments

    # Write output JSON
    print(f"Merged segments: {merged_count}")
    print(f"Reduction: {original_count - merged_count} segments ({(original_count - merged_count) / original_count * 100:.1f}%)")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")

    return {
        'original_count': original_count,
        'merged_count': merged_count,
        'reduction': original_count - merged_count,
        'reduction_percent': (original_count - merged_count) / original_count * 100
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge consecutive segments from the same speaker in JSON transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,       
    )

    parser.add_argument(
        'input_files',
        nargs='+',
        type=str,
        help='Input JSON file(s) to process'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output JSON file (only works with single input file)'
    )

    args = parser.parse_args()

    # Check output argument
    if args.output and len(args.input_files) > 1:
        print("Error: --output can only be used with a single input file")
        sys.exit(1)

    # Process files
    total_stats = {
        'files_processed': 0,
        'total_original': 0,
        'total_merged': 0
    }

    for input_file in args.input_files:
        if not Path(input_file).exists():
            print(f"Warning: File not found: {input_file}")
            continue

        print(f"\n{'='*80}")
        print(f"Processing: {input_file}")
        print(f"{'='*80}")

        stats = merge_json_file(input_file, args.output)

        if stats:
            total_stats['files_processed'] += 1
            total_stats['total_original'] += stats['original_count']
            total_stats['total_merged'] += stats['merged_count']

    # Print summary
    if total_stats['files_processed'] > 1:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Files processed: {total_stats['files_processed']}")
        print(f"Total original segments: {total_stats['total_original']}")
        print(f"Total merged segments: {total_stats['total_merged']}")
        reduction = total_stats['total_original'] - total_stats['total_merged']
        print(f"Total reduction: {reduction} segments ({reduction / total_stats['total_original'] * 100:.1f}%)")


if __name__ == "__main__":
    main()

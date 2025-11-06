#!/usr/bin/env python3
"""
Merge consecutive SRT segments from the same speaker into single segments.
This creates monologues and improves readability by combining speaker turns.
"""

import re
import sys
from pathlib import Path


def parse_srt_file(file_path):
    """Parse an SRT file and return a list of segments."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to get segments
    segments = []
    current_segment = []

    for line in content.split('\n'):
        if line.strip() == '':
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        else:
            current_segment.append(line)

    # Add last segment if exists
    if current_segment:
        segments.append(current_segment)

    return segments


def parse_segment(segment_lines):
    """Parse a single segment into its components."""
    if len(segment_lines) < 3:
        return None

    segment_num = segment_lines[0].strip()
    timestamp = segment_lines[1].strip()
    text_lines = segment_lines[2:]
    text = ' '.join(text_lines).strip()

    # Extract speaker from text
    speaker_match = re.match(r'\[(.*?)\](.*)', text)
    if speaker_match:
        speaker = speaker_match.group(1).strip()
        content = speaker_match.group(2).strip()
    else:
        speaker = None
        content = text

    # Parse timestamps
    time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', timestamp)
    if time_match:
        start_time = time_match.group(1)
        end_time = time_match.group(2)
    else:
        start_time = None
        end_time = None

    return {
        'number': segment_num,
        'start_time': start_time,
        'end_time': end_time,
        'speaker': speaker,
        'content': content,
        'text': text
    }


def merge_consecutive_speakers(segments, skip_unknown=True):
    """Merge consecutive segments from the same speaker.

    Args:
        segments: List of segment lines
        skip_unknown: If True, skip [UNKNOWN] segments and merge around them
    """
    if not segments:
        return []

    parsed_segments = []
    for seg in segments:
        parsed = parse_segment(seg)
        if parsed:
            parsed_segments.append(parsed)

    if not parsed_segments:
        return []

    merged = []
    current = None

    for i in range(len(parsed_segments)):
        next_seg = parsed_segments[i]

        # Skip UNKNOWN segments if requested
        if skip_unknown and next_seg['speaker'] == 'UNKNOWN':
            # If we have accumulated content from a real speaker, merge the UNKNOWN text
            if current and current['speaker'] != 'UNKNOWN':
                current['end_time'] = next_seg['end_time']
                current['content'] = (current['content'] + ' ' + next_seg['content']).strip()
            continue

        # Initialize current if this is the first real segment
        if current is None:
            current = next_seg.copy()
            continue

        # Check if same speaker
        if current['speaker'] == next_seg['speaker']:
            # Merge: keep start time, update end time, combine content
            current['end_time'] = next_seg['end_time']
            current['content'] = (current['content'] + ' ' + next_seg['content']).strip()
        else:
            # Different speaker, save current and start new
            merged.append(current)
            current = next_seg.copy()

    # Add the last segment
    if current:
        merged.append(current)

    return merged


def format_srt_output(merged_segments):
    """Format merged segments back into SRT format."""
    output_lines = []

    for i, seg in enumerate(merged_segments, 1):
        # Segment number
        output_lines.append(str(i))

        # Timestamp
        output_lines.append(f"{seg['start_time']} --> {seg['end_time']}")

        # Text with speaker
        if seg['speaker']:
            output_lines.append(f"[{seg['speaker']}] {seg['content']}")
        else:
            output_lines.append(seg['content'])

        # Blank line between segments
        output_lines.append('')

    return '\n'.join(output_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_speaker_segments.py <input_srt_file> [output_srt_file]")
        print("\nMerges consecutive segments from the same speaker into single segments.")
        print("If output file is not specified, creates a '_merged.srt' version.")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # Determine output file
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = input_file.with_name(input_file.stem + '_merged.srt')

    print(f"Reading: {input_file}")
    segments = parse_srt_file(input_file)
    print(f"Found {len(segments)} segments")

    print("Merging consecutive speaker segments...")
    merged = merge_consecutive_speakers(segments)
    print(f"Merged into {len(merged)} segments")

    print(f"Writing: {output_file}")
    output_content = format_srt_output(merged)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)

    print(f"\nSuccess! Reduced from {len(segments)} to {len(merged)} segments")
    print(f"Reduction: {len(segments) - len(merged)} segments ({100 * (len(segments) - len(merged)) / len(segments):.1f}%)")


if __name__ == '__main__':
    main()

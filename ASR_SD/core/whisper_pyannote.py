import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bisect import bisect_right
from datetime import timedelta

import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

try:
    from anthropic import AnthropicVertex
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class WhisperDiarization:
    """Combines Whisper transcription with pyannote speaker diarization."""

    def __init__(
        self,
        whisper_model: str = "turbo",
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load Whisper model (leave 'turbo' alone if your repo supports it)
        print(f"Loading Whisper model: {whisper_model}...")
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)

        # Load pyannote diarization pipeline
        print(f"Loading diarization model: {diarization_model}...")
        if hf_token:
            self.diarization_pipeline = Pipeline.from_pretrained(
                diarization_model,
                token=hf_token
            )
        else:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(diarization_model)
            except Exception as e:
                print("\nError: Pyannote models require HuggingFace authentication.")
                print("Please provide a HuggingFace token using --hf-token")
                raise e

        # Move diarization to the same device (fix #8: make this robust)
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                self.diarization_pipeline.to(torch.device("cuda"))
            except Exception:
                # if the pipeline doesn't support .to or CUDA isn't properly configured, silently continue on CPU
                pass

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        multilingual: bool = True,
        **whisper_kwargs
    ) -> Dict:
        print(f"\nProcessing: {audio_path}")

        # Step 1: Transcribe with Whisper (with word timestamps)
        print("Step 1/3: Running Whisper transcription...")

        transcribe_options = {
            "word_timestamps": True,
            **whisper_kwargs
        }

        # For multilingual/code-switching: don't specify language, let Whisper auto-detect
        # Also use task='transcribe' for better multilingual support
        if multilingual:
            print("  → Multilingual mode enabled (code-switching detection)")
            transcribe_options["task"] = "transcribe"
            # Don't set language - let it auto-detect per segment
        else:
            transcribe_options["language"] = language

        whisper_result = self.whisper_model.transcribe(audio_path, **transcribe_options)

        # Step 2: Run speaker diarization
        print("Step 2/3: Running speaker diarization...")

        diarization_options = {}
        if num_speakers is not None:
            diarization_options["num_speakers"] = num_speakers
        elif min_speakers is not None or max_speakers is not None:
            diarization_options["min_speakers"] = min_speakers
            diarization_options["max_speakers"] = max_speakers

        diarization = self.diarization_pipeline(audio_path, **diarization_options)

        # Step 3: Align speakers with transcribed words
        print("Step 3/3: Aligning speakers with transcription...")

        result = self._align_speakers_with_transcription(whisper_result, diarization)

        print(f"✓ Transcription complete! Detected {len(result['speakers'])} speakers.")

        return result

    def _align_speakers_with_transcription(
        self,
        whisper_result: Dict,
        diarization: Annotation
    ) -> Dict:
        # Extract words with timestamps from Whisper result
        word_segments = []

        for segment in whisper_result.get("segments", []):
            segment_language = segment.get("language", None)

            if "words" not in segment:
                # Fallback: if no word timestamps, use segment timestamps
                speaker = self._get_speaker_at_timestamp(
                    diarization,
                    (segment["start"] + segment["end"]) / 2
                )
                word_segments.append({
                    "word": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": speaker,
                    "probability": segment.get("probability", 1.0),
                    "language": segment_language
                })
            else:
                for word_info in segment["words"]:
                    # Get speaker at the middle of the word
                    word_middle = (word_info["start"] + word_info["end"]) / 2
                    speaker = self._get_speaker_at_timestamp(diarization, word_middle)

                    word_segments.append({
                        "word": word_info["word"],
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "speaker": speaker,
                        "probability": word_info.get("probability", 1.0),
                        "language": segment_language
                    })

        # Group consecutive words by the same speaker into segments (fix #2 spacing)
        speaker_segments = self._group_by_speaker(word_segments)

        # Get unique speakers
        speakers = sorted(set(ws["speaker"] for ws in word_segments if ws["speaker"]))

        return {
            "text": whisper_result["text"],
            "segments": speaker_segments,
            "word_segments": word_segments,
            "language": whisper_result.get("language", "unknown"),
            "speakers": speakers
        }

    def _get_speaker_at_timestamp(
        self,
        diarization: Annotation,
        timestamp: float
    ) -> Optional[str]:
        best_speaker = None
        best_dist = None
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.start <= timestamp <= segment.end:
                center = (segment.start + segment.end) / 2.0
                dist = abs(center - timestamp)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_speaker = speaker
        return best_speaker

    def _smart_join(self, prev_text: str, token: str) -> str:
        if not prev_text:
            return token

        closing_punct = {",", ".", "!", "?", ":", ";", ")", "]", "}", "’", "”", "'"}
        opening_punct = {"(", "[", "{", "‘", "“"}

        # No space before closing punctuation
        if token in closing_punct:
            return prev_text + token

        # No space after opening punctuation, or if prev ends with opening
        if prev_text[-1] in opening_punct:
            return prev_text + token

        # Apostrophes within words: don't add spaces around "'"
        if token == "'" and prev_text and prev_text[-1].isalnum():
            return prev_text + token

        # Default: add a space
        return prev_text + " " + token

    def _group_by_speaker(self, word_segments: List[Dict]) -> List[Dict]:
        if not word_segments:
            return []

        segments = []
        current_segment = {
            "speaker": word_segments[0]["speaker"],
            "start": word_segments[0]["start"],
            "end": word_segments[0]["end"],
            "text": word_segments[0]["word"],
            "words": [word_segments[0]]
        }

        for word_info in word_segments[1:]:
            if word_info["speaker"] == current_segment["speaker"]:
                current_segment["text"] = self._smart_join(current_segment["text"], word_info["word"])
                current_segment["end"] = word_info["end"]
                current_segment["words"].append(word_info)
            else:
                segments.append(current_segment)
                current_segment = {
                    "speaker": word_info["speaker"],
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "text": word_info["word"],
                    "words": [word_info]
                }

        segments.append(current_segment)
        return segments


class TranscriptRefiner:
    """Uses Claude API to refine transcripts."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        region: str = "us-east5",
        model: str = "claude-3-5-sonnet-v2@20241022"
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")

        self.project_id = project_id or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
        if not self.project_id:
            raise ValueError(
                "Vertex AI project ID required. Set ANTHROPIC_VERTEX_PROJECT_ID env var "
                "or pass project_id parameter"
            )

        self.region = region
        self.model = model

        # Initialize Vertex AI client (unchanged per your request)
        self.client = AnthropicVertex(
            project_id=self.project_id,
            region=self.region
        )

    def refine_transcript(
        self,
        result: Dict,
        max_tokens: int = 200000,
        custom_prompt: Optional[str] = None
    ) -> Dict:
        print("\nStep 4/4: Refining transcript with Claude...")

        segments = result["segments"]
        word_segments = result["word_segments"]

        # STEP 1: Merge consecutive segments from same speaker before LLM refinement
        print(f"    Merging consecutive speaker segments...")
        print(f"    Original segments: {len(segments)}")
        merged_input_segments = self._merge_consecutive_segments(segments)
        print(f"    Merged segments: {len(merged_input_segments)}")

        # STEP 2: Create chunks based on token limit, never splitting speaker segments
        print(f"  → Creating chunks (max {max_tokens} tokens per chunk)...")
        chunks = self._create_smart_chunks(merged_input_segments, max_tokens)
        print(f"    Created {len(chunks)} chunks")

        # STEP 3: Process each chunk for LLM refinement
        refined_segments = []

        for chunk_num, chunk in enumerate(chunks, 1):
            print(f"  → Processing chunk {chunk_num}/{len(chunks)} ({len(chunk)} segments)...")

            refined_chunk = self._refine_chunk(chunk, custom_prompt)
            refined_segments.extend(refined_chunk)

        # Update speaker labels in word segments based on refined segments
        refined_word_segments = self._update_word_segments(word_segments, refined_segments)

        # Use refined segments directly (no heuristic name mapping)
        final_segments = refined_segments
        final_word_segments = refined_word_segments

        # Get unique speakers after refinement
        speakers = sorted(set(seg["speaker"] for seg in final_segments if seg["speaker"]))

        print(f"  Refinement complete! Identified speakers: {', '.join(speakers)}")
        print(f"  Final segments: {len(final_segments)}")

        return {
            "text": result["text"], 
            "segments": final_segments,
            "word_segments": final_word_segments,
            "language": result["language"],
            "speakers": speakers,
            "refinement_applied": True
        }

    def _create_smart_chunks(self, segments: List[Dict], max_tokens: int) -> List[List[Dict]]:
        chunks = []
        current_chunk = []
        current_tokens = 0

        usable_tokens = max_tokens - 2000

        for segment in segments:
            # Estimate tokens for this segment (rough: 4 chars per token)
            # Include speaker label, text, timestamps, and JSON formatting
            segment_text = f"{segment.get('speaker', 'UNKNOWN')}: {segment.get('text', '')}"
            segment_tokens = len(segment_text) // 4 + 50  # +50 for JSON overhead

            # Check if adding this segment would exceed limit
            if current_tokens + segment_tokens > usable_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [segment]
                current_tokens = segment_tokens
            else:
                # Add segment to current chunk
                current_chunk.append(segment)
                current_tokens += segment_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _refine_chunk(self, segments: List[Dict], custom_prompt: Optional[str] = None) -> List[Dict]:
        """Refine a chunk of segments using Claude API."""

        # Prepare transcript chunk
        transcript_text = self._format_segments_for_llm(segments)

        # Build the refinement prompt
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            system_prompt = """You are a transcript refinement expert used in an automated speech pipeline.
                Your job is to CLEAN the text but NEVER break alignment.

                NON-NEGOTIABLE RULES (follow in this exact priority order):

                1. DO NOT change the number of segments. If input has N segments, output MUST have N segments.
                2. DO NOT change timestamps. Keep each segment's `start` and `end` exactly as in the input.
                3. DO NOT merge, split, reorder, or drop segments.
                4. Only change:
                - `speaker`
                - `text`
                Keep everything else as-is.

                REFINEMENT RULES:

                1. Speaker assignment and merging:
                - Use stable, descriptive labels: “Person A”, “Person B”, “Person C”, etc. Assign the same label for the same speaker across all segments in this chunk.
                - If a short segment (1-3 words or ≤1 second duration) labeled as UNKNOWN appears between two consecutive segments from the same speaker, assume it is a continuation of the same person's speech.
                - In such cases, reassign the UNKNOWN segment's speaker to match the surrounding speaker.
                - Then MERGE all three (or more) consecutive segments into a single segment.
                - Set the merged segment's `start` to the first segment's start and `end` to the last segment's end.
                - Concatenate their texts naturally (with a space or comma where appropriate).
                - Remove redundant merged segments, reducing the total count accordingly.

                2. Spelling & grammar:
                - Fix obvious ASR mistakes and casing.
                - Keep technical terms, product names, code, and IDs exactly if they look intentional.

                3. Punctuation:
                - Add commas, periods, and question marks to make it readable.
                - Do not add long stylistic rewrites.

                4. Multilingual / Hindi-English:
                - When text is in Hindi or mixed Hindi-English, translate to clear conversational English.
                - Preserve cultural/intent nuance (“yaar”, “acha”, “haan”) by using lightweight equivalents (“hey”, “okay”, “yeah”) when needed.
                - If translation is ambiguous, keep the original phrase.

                5. Filler words:
                - Remove only obvious fillers that don't change meaning (“um”, “uh”, “like” at the start).
                - Keep hesitations that show intent (“I… I don't know”, “let me think”).

                OUTPUT FORMAT:

                - Return ONLY a JSON array.
                - Each item MUST have exactly these keys: `speaker`, `text`, `start`, `end`.
                - `start` and `end` MUST be the original numeric values from input unless segments were merged, in which case use the earliest start and latest end of the merged range.
                - Do NOT wrap the JSON in markdown fences.
                - Do NOT add explanations, comments, or metadata."""
        user_prompt = f"""Refine this transcript chunk:

{transcript_text}

Return ONLY a valid JSON array of segments. Each segment must have: speaker, text, start, end
Do not include any explanation or markdown formatting, just the JSON array."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Extract JSON from response
            response_text = response.content[0].text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # strip ``` or ```json fences
                if len(lines) >= 2 and lines[-1].strip().startswith("```"):
                    response_text = "\n".join(lines[1:-1])
                if response_text.strip().lower().startswith("json"):
                    response_text = response_text[4:].strip()

            # Validate JSON
            refined_segments = json.loads(response_text)

            # Fix (#5): robust validation & per-segment fallback
            return self._validate_and_merge_segments(segments, refined_segments)

        except Exception as e:
            print(f"  Warning: LLM refinement failed for chunk: {e}")
            print(f"  Falling back to original segments")
            return segments

    def _format_segments_for_llm(self, segments: List[Dict]) -> str:
        """Format segments for LLM input."""
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "").strip()
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            lines.append(f'[{speaker}] ({start:.2f}-{end:.2f}s): {text}')
        return "\n".join(lines)

    def _validate_and_merge_segments(self, original: List[Dict], refined: List[Dict]) -> List[Dict]:
        if not isinstance(refined, list) or len(refined) != len(original):
            print(f"  Warning: Segment count mismatch or non-list; using original")
            return original

        merged = []
        for i, (orig, ref) in enumerate(zip(original, refined)):
            if not isinstance(ref, dict):
                merged.append(orig)
                continue

            spk = ref.get("speaker", orig.get("speaker"))
            txt = ref.get("text", orig.get("text"))

            # guard types
            if not isinstance(spk, (str, type(None))):
                spk = orig.get("speaker")
            if not isinstance(txt, str):
                txt = orig.get("text", "")

            merged.append({
                "speaker": spk,
                "text": txt,
                "start": orig["start"],  # Preserve original times
                "end": orig["end"],
                "words": orig.get("words", [])
            })

        return merged

    def _update_word_segments(self, word_segments: List[Dict], refined_segments: List[Dict]) -> List[Dict]:
        if not refined_segments:
            return word_segments

        starts = [seg["start"] for seg in refined_segments]
        rs = refined_segments
        updated_words = []

        for word in word_segments:
            mid = (word["start"] + word["end"]) / 2
            i = max(0, bisect_right(starts, mid) - 1)

            match = None
            # check neighborhood ±1
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(rs) and rs[j]["start"] <= mid <= rs[j]["end"]:
                    match = rs[j]
                    break

            updated_words.append({
                **word,
                "speaker": match["speaker"] if match else word.get("speaker")
            })

        return updated_words

    def _merge_consecutive_segments(self, segments: List[Dict]) -> List[Dict]:
        if not segments:
            return []

        merged = []
        current = {
            "speaker": segments[0].get("speaker"),
            "text": segments[0].get("text", "").strip(),
            "start": segments[0].get("start"),
            "end": segments[0].get("end"),
            "words": segments[0].get("words", [])
        }

        for segment in segments[1:]:
            speaker = segment.get("speaker")

            # If same speaker, merge the segments
            if speaker == current["speaker"]:
                # Append text with proper spacing
                new_text = segment.get("text", "").strip()
                if current["text"] and new_text:
                    # Add space between segments unless the previous ends with certain punctuation
                    if current["text"][-1] not in {'.', '!', '?', ':', ';', ','}:
                        current["text"] += " " + new_text
                    else:
                        current["text"] += " " + new_text
                elif new_text:
                    current["text"] = new_text

                # Extend the end time
                current["end"] = segment.get("end")

                # Merge words if available
                current["words"].extend(segment.get("words", []))
            else:
                # Different speaker - save current and start new
                merged.append(current)
                current = {
                    "speaker": speaker,
                    "text": segment.get("text", "").strip(),
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "words": segment.get("words", [])
                }

        # Don't forget the last segment
        merged.append(current)

        return merged


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def save_results(result: Dict, output_path: str, format: str = "all", suffix: str = ""):
    output_base = Path(output_path)
    if suffix:
        output_base = output_base.parent / f"{output_base.stem}{suffix}"

    # JSON format (complete data)
    if format in ["json", "all"]:
        json_path = output_base.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {json_path}")

    # TXT format (speaker labels + text)
    if format in ["txt", "all"]:
        txt_path = output_base.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                speaker = segment["speaker"] or "UNKNOWN"
                f.write(f"[{speaker}] {segment['text'].strip()}\n")
        print(f"Saved TXT: {txt_path}")

    # SRT format (subtitles with speakers)
    if format in ["srt", "all"]:
        srt_path = output_base.with_suffix(".srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], 1):
                speaker = segment["speaker"] or "UNKNOWN"
                start = format_timestamp(segment["start"]).replace(".", ",")
                end = format_timestamp(segment["end"]).replace(".", ",")
                text = f"[{speaker}] {segment['text'].strip()}"
                # text = _wrap_srt(text)  # DISABLED: Keep natural flow without line breaks

                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        print(f"Saved SRT: {srt_path}")


def main():
    """Command-line interface for Whisper + Diarization"""
    parser = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization using Whisper + pyannote",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "audio",
        type=str,
        help="Path to audio file"
    )

    # Model arguments
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="turbo",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
        help="Whisper model size"
    )

    parser.add_argument(
        "--diarization-model",
        type=str,
        default="pyannote/speaker-diarization-3.1",
        help="HuggingFace diarization model"
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for pyannote models"
    )

    # Transcription arguments
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'es', 'fr') or None for auto-detect"
    )

    # Speaker arguments
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers (if known)"
    )

    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers"
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers"
    )

    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Enable multilingual/code-switching mode (e.g., English-Hindi mixing)"
    )

    # LLM Refinement arguments
    parser.add_argument(
        "--refine-with-llm",
        action="store_true",
        default=True,
        help="Enable LLM-based transcript refinement using Claude via Vertex AI (default: enabled)"
    )

    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable LLM refinement (skip the refinement step)"
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
        help="Vertex AI region (default: us-east5)"
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
        "--max-tokens",
        type=int,
        default=200000,
        help="Maximum tokens per LLM chunk (default: 200k for ~256k context window)"
    )

    # Output arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (without extension)"
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="all",
        choices=["txt", "json", "srt", "all"],
        help="Output format"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use for inference"
    )

    args = parser.parse_args()

    # Handle --no-refine flag
    if args.no_refine:
        args.refine_with_llm = False

    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return

    # Initialize pipeline
    try:
        pipeline = WhisperDiarization(
            whisper_model=args.whisper_model,
            diarization_model=args.diarization_model,
            device=args.device,
            hf_token=args.hf_token
        )
    except Exception as e:
        print(f"\nError initializing pipeline: {e}")
        return

    # Transcribe
    try:
        result = pipeline.transcribe(
            args.audio,
            language=args.language,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            multilingual=args.multilingual
        )

        # Determine output path
        if args.output is None:
            audio_path = Path(args.audio)
            output_path = audio_path.parent / f"{audio_path.stem}_transcription"
        else:
            output_path = args.output

        # Save raw results
        save_results(result, output_path, args.format, suffix="_raw" if args.refine_with_llm else "")

        # Apply LLM refinement if requested (Vertex creds behavior intentionally unchanged)
        if args.refine_with_llm:
            try:
                refiner = TranscriptRefiner(
                    project_id=args.vertex_project_id,
                    region=args.vertex_region,
                    model=args.claude_model
                )

                refined_result = refiner.refine_transcript(
                    result,
                    max_tokens=args.max_tokens,
                    custom_prompt=args.refinement_prompt
                )

                # Save refined results
                save_results(refined_result, output_path, args.format, suffix="_refined")

                # Use refined result for summary display
                result = refined_result

            except Exception as e:
                print(f"\nError during LLM refinement: {e}")
                print("Continuing with raw transcript only...")
        else:
            # Save results without suffix if no refinement
            pass
    except KeyboardInterrupt:
        print("\n\n⚠ Transcription cancelled by user")
        return

    # Print summary
    print("TRANSCRIPTION SUMMARY")
    print(f"Language: {result['language']}")
    print(f"Speakers detected: {len(result['speakers'])}")
    print(f"Speakers: {', '.join(result['speakers'])}")
    print(f"\nSegments: {len(result['segments'])}")


if __name__ == "__main__":
    main()

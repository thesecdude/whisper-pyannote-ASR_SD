#!/bin/bash
# Test script for LLM refinement feature via Vertex AI

echo "=========================================="
echo "LLM Refinement Test Script (Vertex AI)"
echo "=========================================="
echo ""

# Check if ANTHROPIC_VERTEX_PROJECT_ID is set
if [ -z "$ANTHROPIC_VERTEX_PROJECT_ID" ]; then
    echo "Error: ANTHROPIC_VERTEX_PROJECT_ID environment variable not set"
    echo ""
    echo "Please set it with:"
    echo "  export ANTHROPIC_VERTEX_PROJECT_ID='your-project-id'"
    echo ""
    echo "Or add to ~/.zshrc for persistence"
    exit 1
fi

echo "✓ Vertex AI project ID found: $ANTHROPIC_VERTEX_PROJECT_ID"
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable not set"
    echo "You may need to provide it with --hf-token"
    echo ""
fi

# Select test file
TEST_FILE="results/sahil and I/Friday at 15-14.mp3"

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found: $TEST_FILE"
    echo ""
    echo "Please specify a different file as the first argument:"
    echo "  bash test_llm_refinement.sh path/to/audio.mp3"
    exit 1
fi

if [ ! -z "$1" ]; then
    TEST_FILE="$1"
fi

echo "Test file: $TEST_FILE"
echo ""

# Run transcription with LLM refinement
echo "Running transcription with LLM refinement..."
echo "This will:"
echo "  1. Transcribe with Whisper (turbo model)"
echo "  2. Perform speaker diarization with pyannote"
echo "  3. Save raw transcript"
echo "  4. Refine with Claude AI"
echo "  5. Save refined transcript"
echo ""

OUTPUT_PATH="${TEST_FILE%.mp3}_llm_test"

python whisper_diarization.py \
    "$TEST_FILE" \
    --whisper-model turbo \
    --hf-token "$HF_TOKEN" \
    --multilingual \
    --refine-with-llm \
    --output "$OUTPUT_PATH" \
    --chunk-size 20

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Test completed successfully!"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo ""
    echo "Raw transcripts (before LLM):"
    ls -lh "${OUTPUT_PATH}_raw."* 2>/dev/null
    echo ""
    echo "Refined transcripts (after LLM):"
    ls -lh "${OUTPUT_PATH}_refined."* 2>/dev/null
    echo ""
    echo "Compare the outputs:"
    echo "  Raw text:     cat '${OUTPUT_PATH}_raw.txt'"
    echo "  Refined text: cat '${OUTPUT_PATH}_refined.txt'"
    echo ""
    echo "  Raw JSON:     cat '${OUTPUT_PATH}_raw.json'"
    echo "  Refined JSON: cat '${OUTPUT_PATH}_refined.json'"
else
    echo ""
    echo "=========================================="
    echo "Test failed!"
    echo "=========================================="
    echo ""
    echo "Check the error messages above."
fi

#!/bin/bash
# Progress checker for transcription job

echo "=========================================="
echo "Transcription Progress Monitor"
echo "=========================================="
echo ""

# Check if process is running
if ps aux | grep -q "[w]hisper_diarization.py"; then
    echo "✓ Process is RUNNING"
    echo ""

    # Show CPU/Memory usage
    echo "Resource Usage:"
    ps aux | grep "[w]hisper_diarization.py" | awk '{print "  CPU: " $3 "% | Memory: " $4 "% | Time: " $10}'
    echo ""

    # Check if output files exist
    echo "Output Files:"
    if [ -f "results/PP/PPWalkthrough_transcript.json" ]; then
        SIZE=$(ls -lh "results/PP/PPWalkthrough_transcript.json" | awk '{print $5}')
        echo "  ✓ JSON file exists ($SIZE)"
    else
        echo "  ⏳ JSON file - not yet created"
    fi

    if [ -f "results/PP/PPWalkthrough_transcript.txt" ]; then
        SIZE=$(ls -lh "results/PP/PPWalkthrough_transcript.txt" | awk '{print $5}')
        echo "  ✓ TXT file exists ($SIZE)"
    else
        echo "  ⏳ TXT file - not yet created"
    fi

    if [ -f "results/PP/PPWalkthrough_transcript.srt" ]; then
        SIZE=$(ls -lh "results/PP/PPWalkthrough_transcript.srt" | awk '{print $5}')
        echo "  ✓ SRT file exists ($SIZE)"
    else
        echo "  ⏳ SRT file - not yet created"
    fi

    echo ""
    echo "Status: Processing... (check again in a few minutes)"

else
    echo "✗ Process is NOT running"
    echo ""

    # Check if output files were created
    if [ -f "results/PP/PPWalkthrough_transcript.json" ]; then
        echo "✓ Process COMPLETED! Output files:"
        ls -lh results/PP/PPWalkthrough_transcript.* 2>/dev/null
    else
        echo "✗ Process may have failed. Check logs."
    fi
fi

echo ""
echo "=========================================="
echo "To check again, run: bash check_progress.sh"
echo "=========================================="

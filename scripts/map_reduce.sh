#!/bin/bash

# A Map-Reduce script that splits a large text file into chunks
# and spawns Gemini CLI sub-agents to analyze each chunk in parallel or sequentially.

set -euo pipefail

BOOK_FILE="$1"
FINDINGS_FILE="$2"
MAX_RETRIES="${3:-2}"

if [ -z "$BOOK_FILE" ] || [ -z "$FINDINGS_FILE" ]; then
    echo "Usage: $0 <path_to_book.txt> <output_findings.txt> [max_retries]"
    exit 1
fi

if [ ! -f "$BOOK_FILE" ]; then
    echo "Error: Book file not found: $BOOK_FILE"
    exit 1
fi

CHUNK_DIR="/tmp/nartopo_chunks_$(basename "$BOOK_FILE" .txt)"
mkdir -p "$CHUNK_DIR"
rm -f "$CHUNK_DIR"/*
rm -f "$FINDINGS_FILE"

echo "Splitting book into manageable chunks for sub-agents..."
# Split by size, approx 150k chars per chunk
split -C 150k "$BOOK_FILE" "$CHUNK_DIR/chunk_"

# Count chunks
CHUNK_COUNT=$(ls -1 "$CHUNK_DIR"/chunk_* 2>/dev/null | wc -l)
if [ "$CHUNK_COUNT" -eq 0 ]; then
    echo "Error: No chunks created. Is the book file empty?"
    exit 1
fi
echo "Created $CHUNK_COUNT chunks."

FAILED_CHUNKS=()

# Process a single chunk with retry logic
process_chunk() {
    local chunk="$1"
    local outfile="${chunk}_out.txt"
    local attempt=0

    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        echo "## Findings from $(basename "$chunk")" > "$outfile"
        if gemini -y -p "You are a structural analysis sub-agent. Read the following chunk of a novel. Extract the most important plot events, character shifts, pacing metrics, and structural themes (especially regarding systemic threats and character agency). Keep it to dense, high-signal bullet points. Do not summarize the whole story, just extract the structural data from this specific text." < "$chunk" >> "$outfile" 2>/dev/null; then
            # Verify non-empty output (more than just the header line)
            local line_count
            line_count=$(wc -l < "$outfile")
            if [ "$line_count" -gt 2 ]; then
                echo "Sub-agent finished $(basename "$chunk") (attempt $((attempt + 1)))"
                return 0
            fi
            echo "Warning: $(basename "$chunk") produced empty output (attempt $((attempt + 1)))"
        else
            echo "Warning: $(basename "$chunk") failed with exit code $? (attempt $((attempt + 1)))"
        fi

        attempt=$((attempt + 1))
        if [ "$attempt" -le "$MAX_RETRIES" ]; then
            local backoff=$((attempt * 5))
            echo "Retrying $(basename "$chunk") in ${backoff}s..."
            sleep "$backoff"
        fi
    done

    echo "ERROR: $(basename "$chunk") failed after $((MAX_RETRIES + 1)) attempts"
    return 1
}

# Process in parallel using background jobs
for chunk in "$CHUNK_DIR"/chunk_*; do
    echo "Spawning sub-agent for $(basename "$chunk")..."
    (
        if ! process_chunk "$chunk"; then
            echo "$(basename "$chunk")" >> "$CHUNK_DIR/_failed.txt"
        fi
    ) &
done

echo "Waiting for all $CHUNK_COUNT sub-agents to complete..."
wait

# Check for failures
if [ -f "$CHUNK_DIR/_failed.txt" ]; then
    FAIL_COUNT=$(wc -l < "$CHUNK_DIR/_failed.txt")
    echo ""
    echo "ERROR: $FAIL_COUNT/$CHUNK_COUNT chunks failed:"
    cat "$CHUNK_DIR/_failed.txt"
    echo ""
    echo "Partial results will still be synthesized, but the analysis is incomplete."
    echo "Re-run with a higher retry count: $0 \"$BOOK_FILE\" \"$FINDINGS_FILE\" 3"
fi

# Validate that at least some output was produced
OUT_COUNT=$(ls -1 "$CHUNK_DIR"/chunk_*_out.txt 2>/dev/null | wc -l)
if [ "$OUT_COUNT" -eq 0 ]; then
    echo "FATAL: No chunks produced output. Aborting."
    exit 1
fi

echo "Synthesizing findings ($OUT_COUNT/$CHUNK_COUNT chunks)..."
cat "$CHUNK_DIR"/chunk_*_out.txt > "$FINDINGS_FILE"

if [ -f "$CHUNK_DIR/_failed.txt" ]; then
    echo "Map-reduce complete with errors. Partial results saved to $FINDINGS_FILE"
    exit 1
else
    echo "Map-reduce complete. Synthesized structural data saved to $FINDINGS_FILE"
fi

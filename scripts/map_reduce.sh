#!/bin/bash

# A Map-Reduce script that splits a large text file into chunks 
# and spawns Gemini CLI sub-agents to analyze each chunk in parallel or sequentially.

BOOK_FILE="$1"
FINDINGS_FILE="$2"

if [ -z "$BOOK_FILE" ] || [ -z "$FINDINGS_FILE" ]; then
    echo "Usage: $0 <path_to_book.txt> <output_findings.txt>"
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
CHUNK_COUNT=$(ls -1 "$CHUNK_DIR"/chunk_* | wc -l)
echo "Created $CHUNK_COUNT chunks."

# Process in parallel using background jobs and wait
for chunk in "$CHUNK_DIR"/chunk_*; do
    echo "Spawning sub-agent for $(basename "$chunk")..."
    (
        echo "## Findings from $(basename "$chunk")" > "${chunk}_out.txt"
        gemini -y -p "You are a structural analysis sub-agent. Read the following chunk of a novel. Extract the most important plot events, character shifts, pacing metrics, and structural themes (especially regarding systemic threats and character agency). Keep it to dense, high-signal bullet points. Do not summarize the whole story, just extract the structural data from this specific text." < "$chunk" >> "${chunk}_out.txt"
        echo "Sub-agent finished $(basename "$chunk")"
    ) &
done

echo "Waiting for all $CHUNK_COUNT sub-agents to complete..."
wait

echo "Synthesizing findings..."
cat "$CHUNK_DIR"/chunk_*_out.txt > "$FINDINGS_FILE"

echo "Map-reduce complete. Synthesized structural data saved to $FINDINGS_FILE"

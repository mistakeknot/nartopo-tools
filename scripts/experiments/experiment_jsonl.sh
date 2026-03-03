#!/bin/bash

BOOK_FILE="$1"
OUTPUT_FILE="$2"

if [ -z "$BOOK_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <path_to_book.txt> <output_findings.jsonl>"
    exit 1
fi

CHUNK_DIR="/tmp/nartopo_jsonl_chunks_$(basename "$BOOK_FILE" .txt)"
mkdir -p "$CHUNK_DIR"
rm -f "$CHUNK_DIR"/*
rm -f "$OUTPUT_FILE"

echo "Splitting book into manageable chunks..."
# Split by size, approx 150k chars per chunk
split -C 150k "$BOOK_FILE" "$CHUNK_DIR/chunk_"

for chunk in "$CHUNK_DIR"/chunk_*; do
    echo "Spawning sub-agent for $(basename "$chunk")..."
    (
        gemini -y -p "You are a structural analysis sub-agent. Read the following chunk of a novel. DO NOT output any prose or markdown. You must extract every major structural beat, scene, or event and output it as a strict JSON Lines (JSONL) document. Each line MUST be a valid JSON object matching this schema: {\"type\": \"action\"|\"dialogue\"|\"exposition\"|\"bureaucracy\", \"summary\": \"Brief 1-sentence summary of the event\"}. Output only the raw JSONL." < "$chunk" >> "${chunk}_out.jsonl"
        echo "Sub-agent finished $(basename "$chunk")"
    ) &
done

wait

echo "Synthesizing JSONL findings..."
cat "$CHUNK_DIR"/chunk_*_out.jsonl | grep -v '```' | grep '^{' > "$OUTPUT_FILE"

echo "JSONL extraction complete. Structural timeline saved to $OUTPUT_FILE"

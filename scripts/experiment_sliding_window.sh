#!/bin/bash

BOOK_FILE="$1"
OUTPUT_FILE="$2"

if [ -z "$BOOK_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <path_to_book.txt> <output_findings.txt>"
    exit 1
fi

CHUNK_DIR="/tmp/nartopo_sliding_chunks_$(basename "$BOOK_FILE" .txt)"
mkdir -p "$CHUNK_DIR"
rm -f "$CHUNK_DIR"/*
rm -f "$OUTPUT_FILE"

echo "Splitting book into manageable chunks..."
# Split by size, approx 150k chars per chunk
split -C 150k "$BOOK_FILE" "$CHUNK_DIR/chunk_"

PREVIOUS_SUMMARY=""

for chunk in "$CHUNK_DIR"/chunk_*; do
    echo "Processing $(basename "$chunk")..."
    
    # We must run this sequentially to pass the summary forward
    cat <<PROMPT > /tmp/sliding_prompt.txt
You are a structural analysis sub-agent reading through a novel sequentially. 

Here is the running summary of the story and structure up to this point:
---
${PREVIOUS_SUMMARY:-"(Start of the book)"}
---

Now, read the next chunk of the novel provided below. Extract the most important plot events, character shifts, pacing metrics, and structural themes. 
Finally, provide an updated, unified summary (under 500 words) that incorporates the previous summary and the new chunk, so the next agent has full context.

Return your response in two sections:
## Structural Findings
...
## Updated Running Summary
...
PROMPT

    cat /tmp/sliding_prompt.txt "$chunk" | gemini -y -p "Read the prompt and input text from stdin and execute the analysis." > "${chunk}_out.txt"
    
    # Extract just the running summary for the next agent
    PREVIOUS_SUMMARY=$(awk '/## Updated Running Summary/{flag=1; next} flag' "${chunk}_out.txt")
    
    # Extract just the findings for our final log
    echo -e "

### Findings from $(basename "$chunk")
" >> "$OUTPUT_FILE"
    awk '/## Structural Findings/{flag=1; next} /## Updated Running Summary/{flag=0} flag' "${chunk}_out.txt" >> "$OUTPUT_FILE"
    
    echo "Summary passed forward."
done

echo "Sliding Window complete. Synthesized structural data saved to $OUTPUT_FILE"
#!/bin/bash

BOOK_FILE="$1"
OUTPUT_FILE="$2"

if [ -z "$BOOK_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <path_to_book.txt> <output_findings.txt>"
    exit 1
fi

echo "Running tldrs chunk-summary on the book..."
# Since tldrs expects code and doesn't explicitly support giant pure-text prose blocks 
# natively without some structural syntax, let's see how its internal chunk-summary 
# compressor deals with the text file.

tldrs extract "$BOOK_FILE" --compress chunk-summary > "$OUTPUT_FILE"
echo "Done. Extracted summary."

import sys
import os
import json
import urllib.request
import faiss
import numpy as np
import subprocess

if len(sys.argv) < 3:
    print("Usage: python3 experiment_grand_unification.py <book.txt> <output.jsonl>")
    sys.exit(1)

BOOK_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]

with open(BOOK_FILE, "r", encoding="utf-8") as f:
    text = f.read()

macro_chunks = [text[i:i+150000] for i in range(0, len(text), 150000)]

def get_nomic_embedding(text):
    payload = {"model": "nomic-embed-text", "prompt": text}
    req = urllib.request.Request(
        "http://localhost:11434/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        response = urllib.request.urlopen(req)
        return json.loads(response.read())["embedding"]
    except Exception as e:
        return []

keywords = [
    "major plot progression action",
    "character emotional shift dialogue",
    "bureaucracy exposition history",
    "systemic threat or conflict"
]

previous_summary = "(Start of the book)"
final_jsonl = []

for chunk_idx, macro_chunk in enumerate(macro_chunks):
    print(f"Processing chunk {chunk_idx + 1}/{len(macro_chunks)}...")
    micro_chunks = [macro_chunk[i:i+4000] for i in range(0, len(macro_chunk), 4000)]
    
    embeddings = []
    for p in micro_chunks:
        emb = get_nomic_embedding(p)
        if emb: embeddings.append(emb)
    
    if not embeddings: continue
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    
    snippets = []
    for kw in keywords:
        q_vec = np.array([get_nomic_embedding(kw)], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        _, indices = index.search(q_vec, 3)
        for i in indices[0]:
            if i < len(micro_chunks):
                snippets.append(micro_chunks[i])
    
    snippets_text = "\n\n---\n\n".join(list(set(snippets)))
    
    prompt = f"""You are a structural analysis sub-agent.
Here is the running summary of the story so far:
{previous_summary}

Here are the extracted semantic snippets from the CURRENT chunk of the novel:
{snippets_text}

Task:
1. Extract the major structural beats from the snippets and output them as strict JSON Lines (JSONL). Schema: {{"type": "action"|"dialogue"|"exposition"|"bureaucracy", "summary": "Brief 1-sentence summary"}}
2. Output an updated running summary (under 300 words) that incorporates the previous summary and the new events.

Format your response EXACTLY like this (do not use markdown blocks for the JSON):
## JSONL
{{"type": "...", "summary": "..."}}
## SUMMARY
[Your updated summary here]
"""
    
    print("Calling Gemini CLI...")
    process = subprocess.Popen(
        ["gemini", "-y", "-p", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    
    jsonl_part = ""
    summary_part = ""
    current_section = None
    for line in stdout.split('\n'):
        if line.startswith('## JSONL'):
            current_section = 'jsonl'
            continue
        elif line.startswith('## SUMMARY'):
            current_section = 'summary'
            continue
        
        if current_section == 'jsonl':
            if line.strip().startswith('{'):
                final_jsonl.append(line.strip())
        elif current_section == 'summary':
            summary_part += line + "\n"
            
    if summary_part.strip():
        previous_summary = summary_part.strip()
        print("Updated running summary successfully.")
        
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(final_jsonl))

print("Done. Saved to", OUTPUT_FILE)
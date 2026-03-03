import sys
import os
import json
import urllib.request
import faiss
import numpy as np

BOOK_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]
print(f"Loading {BOOK_FILE}...")

with open(BOOK_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Chunk by 150k blocks first (Map-Reduce layer)
print("Partitioning text into 150k chunks...")
macro_chunks = [text[i:i+150000] for i in range(0, len(text), 150000)]
print(f"Created {len(macro_chunks)} macro-chunks.")

def get_nomic_embedding(text):
    payload = {
        "model": "nomic-embed-text",
        "prompt": text
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        return result["embedding"]
    except Exception as e:
        print("Error getting embedding:", e)
        return []

keywords = [
    "inciting incident disruption Outside Context Problem",
    "climax of the conflict fleet battle",
    "theme of systemic failure or AI bureaucracy",
    "betrayal or emotional shift human",
    "pacing speed combat microseconds",
    "protagonist agency or control",
    "ideological price or sacrifice"
]

all_results = {}
total_chars = 0

for chunk_idx, macro_chunk in enumerate(macro_chunks):
    print(f"\nProcessing macro-chunk {chunk_idx + 1}/{len(macro_chunks)}...")
    
    # Sub-agent semantic layer: chunk into 4000-char blocks
    micro_chunks = [macro_chunk[i:i+4000] for i in range(0, len(macro_chunk), 4000)]
    
    embeddings = []
    for i, p in enumerate(micro_chunks):
        emb = get_nomic_embedding(p)
        if emb:
            embeddings.append(emb)
            
    if not embeddings:
        continue
        
    embeddings_array = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    
    def search(query, k=3):
        query_vector = np.array([get_nomic_embedding(query)], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector, k)
        return [micro_chunks[i] for i in indices[0]]
        
    for kw in keywords:
        snippets = search(kw, k=3)
        if kw not in all_results:
            all_results[kw] = []
        all_results[kw].extend(snippets)
        for s in snippets:
            total_chars += len(s)

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved Hybrid Index snippets to {OUTPUT_FILE}. Total chars extracted: {total_chars} (down from {len(text)})")
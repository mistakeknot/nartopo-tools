import sys
import os
import json
import urllib.request
import faiss
import numpy as np

BOOK_FILE = sys.argv[1]
print(f"Loading {BOOK_FILE}...")

with open(BOOK_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Chunk by larger sizes since Nomic has an 8192 context window (unlike MiniLM's 512)
# We can use ~4000 characters per chunk safely to gather much broader context
print("Chunking text into 4000-character blocks...")
paragraphs = [text[i:i+4000] for i in range(0, len(text), 4000)]
print(f"Extracted {len(paragraphs)} paragraphs.")

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

print("Encoding paragraphs via Ollama (nomic-embed-text)...")
embeddings = []
for i, p in enumerate(paragraphs):
    if i % 10 == 0:
        print(f"Encoding {i}/{len(paragraphs)}")
    emb = get_nomic_embedding(p)
    if emb:
        embeddings.append(emb)

embeddings_array = np.array(embeddings, dtype=np.float32)

print("Building FAISS index...")
dimension = embeddings_array.shape[1]
index = faiss.IndexFlatIP(dimension)

faiss.normalize_L2(embeddings_array)
index.add(embeddings_array)

def search(query, k=10):
    query_vector = np.array([get_nomic_embedding(query)], dtype=np.float32)
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)
    return [paragraphs[i] for i in indices[0]]

keywords = [
    "inciting incident disruption Outside Context Problem",
    "climax of the conflict fleet battle",
    "theme of systemic failure or AI bureaucracy",
    "betrayal or emotional shift human Dajeil",
    "pacing speed combat microseconds",
    "protagonist agency or control",
    "ideological price or sacrifice"
]

results = {}
total_chars = 0
for kw in keywords:
    print(f"Querying: {kw}")
    snippets = search(kw, k=10)
    results[kw] = snippets
    for s in snippets:
        total_chars += len(s)

with open("/tmp/rag_snippets_nomic.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved Nomic RAG snippets to /tmp/rag_snippets_nomic.json. Total chars extracted: {total_chars} (down from {len(text)})")

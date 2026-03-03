import sys
import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

BOOK_FILE = sys.argv[1]
print(f"Loading {BOOK_FILE}...")

with open(BOOK_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Chunk by paragraphs for fine-grained retrieval
print("Chunking text by double-newlines...")
paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
print(f"Extracted {len(paragraphs)} paragraphs.")

print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding paragraphs...")
embeddings = model.encode(paragraphs, show_progress_bar=True)

print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)
index.add(embeddings)

def search(query, k=15):
    query_vector = model.encode([query])
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
    snippets = search(kw, k=15)
    results[kw] = snippets
    for s in snippets:
        total_chars += len(s)

with open("/tmp/rag_snippets.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved RAG snippets to /tmp/rag_snippets.json. Total chars extracted: {total_chars} (down from {len(text)})")

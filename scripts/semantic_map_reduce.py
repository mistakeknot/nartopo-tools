import sys
import os
import json
import urllib.request
import faiss
import numpy as np
import asyncio
import argparse
import time
import re

def get_nomic_embedding_sync(text):
    payload = {"model": "nomic-embed-text", "prompt": text}

    # Try the remote RTX 4090 via Tailscale first, fallback to localhost
    ollama_host = os.environ.get("OLLAMA_HOST", "http://100.107.177.128:11434")

    req = urllib.request.Request(
        f"{ollama_host}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        response = urllib.request.urlopen(req, timeout=10)
        return json.loads(response.read())["embedding"]
    except Exception as e:
        # If remote fails, fallback to local
        if "100.107.177.128" in ollama_host:
            req = urllib.request.Request(
                "http://localhost:11434/api/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            try:
                response = urllib.request.urlopen(req, timeout=10)
                return json.loads(response.read())["embedding"]
            except Exception as e2:
                pass
        return []

async def get_nomic_embedding(text):
    return await asyncio.to_thread(get_nomic_embedding_sync, text)

async def run_gemini_cli(prompt):
    """Runs the Gemini CLI sub-agent asynchronously."""
    process = await asyncio.create_subprocess_exec(
        "gemini", "-y", "-p", prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return stdout.decode('utf-8', errors='replace')

async def process_chunk(chunk_idx, micro_chunks, keywords, global_outline, book_file):
    chunk_start = time.time()
    print(f"[{chunk_idx}] Generating or loading embeddings...")

    # Cache paths for this specific chunk
    cache_base = f"{book_file}_chunk{chunk_idx}"
    faiss_path = f"{cache_base}.faiss"
    chunks_path = f"{cache_base}_chunks.json"
    
    valid_micro_chunks = []
    
    emb_start = time.time()
    if os.path.exists(faiss_path) and os.path.exists(chunks_path):
        print(f"[{chunk_idx}] Loading cached embeddings from {faiss_path}")
        index = faiss.read_index(faiss_path)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            valid_micro_chunks = json.load(f)
    else:
        # Get embeddings for all micro_chunks concurrently
        tasks = [get_nomic_embedding(p) for p in micro_chunks]
        embeddings = await asyncio.gather(*tasks)
        
        valid_data = [(m, e) for m, e in zip(micro_chunks, embeddings) if e]
        if not valid_data:
            return ""
            
        valid_micro_chunks, valid_embeddings = zip(*valid_data)
        valid_micro_chunks = list(valid_micro_chunks) # ensure list for JSON serialization
        
        embeddings_array = np.array(valid_embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(embeddings_array.shape[1])
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
        # Save to cache
        faiss.write_index(index, faiss_path)
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(valid_micro_chunks, f)
            
    emb_time = time.time() - emb_start
    print(f"[{chunk_idx}] Embeddings ready in {emb_time:.2f}s")
    
    snippets = []
    # Get embeddings for keywords concurrently
    kw_tasks = [get_nomic_embedding(kw) for kw in keywords]
    kw_embeddings = await asyncio.gather(*kw_tasks)
    
    for kw_emb in kw_embeddings:
        if not kw_emb: continue
        q_vec = np.array([kw_emb], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        _, indices = index.search(q_vec, 3)
        for i in indices[0]:
            if i < len(valid_micro_chunks):
                snippets.append(valid_micro_chunks[i])
                
    snippets_text = "\n\n---\n\n".join(list(set(snippets)))
    
    print(f"[{chunk_idx}] Invoking Gemini CLI for extraction...")
    ext_start = time.time()
    
    schema_instruction = '{"type": "action"|"dialogue"|"exposition"|"bureaucracy", "summary": "Brief 1-sentence summary"}'

    prompt = f"""You are a structural analysis sub-agent analyzing Chunk {chunk_idx}.

Here is the GLOBAL OUTLINE of the entire novel to provide you with broad narrative context:
{global_outline}

Here are the extracted semantic snippets from YOUR SPECIFIC CHUNK of the novel:
{snippets_text}

Task:
Extract the major structural beats from the snippets and output them as strict JSON Lines (JSONL).
Do not hallucinate events outside these snippets. Use the global outline only for understanding context and character identities.

Schema: {schema_instruction}

Format your response EXACTLY like this (do not use markdown blocks for the JSON):
## JSONL
{{"type": "...", "summary": "..."}}
"""

    output = await run_gemini_cli(prompt)
    ext_time = time.time() - ext_start
    
    jsonl_lines = []
    in_jsonl = False
    for line in output.split('\n'):
        if line.startswith('## JSONL'):
            in_jsonl = True
            continue
        if in_jsonl and line.strip().startswith('{'):
            jsonl_lines.append(line.strip())
            
    chunk_time = time.time() - chunk_start
    print(f"[{chunk_idx}] Completed in {chunk_time:.2f}s (Ext: {ext_time:.2f}s). Extracted {len(jsonl_lines)} events.")
    return "\n".join(jsonl_lines)

SNIPPET_SIZE = 1500
N_SNIPPETS = 30


def select_snippets_stratified(text, n_snippets=N_SNIPPETS, snippet_size=SNIPPET_SIZE):
    """Select evenly-spaced snippets with guaranteed opening and closing coverage."""
    text_len = len(text)
    if text_len <= snippet_size * 2:
        return [(0.0, text)]

    snippets = []
    snippets.append((0.0, text[:snippet_size]))
    snippets.append((100.0, text[-snippet_size:]))

    n_middle = n_snippets - 2
    step = text_len // (n_middle + 1)
    for i in range(1, n_middle + 1):
        start = i * step
        start = min(start, text_len - snippet_size)
        pct = (start / text_len) * 100
        snippets.append((pct, text[start : start + snippet_size]))

    snippets.sort(key=lambda x: x[0])
    return snippets


OUTLINE_PROMPT = """You are a master literary structuralist performing a fast-pass structural survey.
I am giving you {n} evenly-spaced text samples labeled with position (0%=opening, 100%=final pages).

{snippets_text}

Based on these samples, produce the following structured outline:

## DRAMATIS PERSONAE
Every named character: Name (aliases), Role, One-sentence description

## NARRATIVE STRUCTURE
- Temporal mode (linear, flashbacks, parallel timelines, frame narrative)
- POV type and POV characters
- Narrative frame (1st person, 3rd limited, omniscient, epistolary)

## PLOT OUTLINE
Chronological summary, 1 paragraph per major narrative movement.

## THEMATIC TENSIONS
2-4 central binary oppositions driving the narrative.

Be concrete. Use actual names. Do not speculate about content between snippets."""


async def generate_global_outline(text):
    """Generate a structured global outline using stratified snippet sampling."""
    t0 = time.time()
    print("\n--- Phase 1: Generating Global Outline (stratified, 30 snippets) ---")

    snippets = select_snippets_stratified(text)
    coverage = sum(len(s) for _, s in snippets)
    print(f"  Snippets: {len(snippets)}, coverage: {coverage:,} chars ({coverage / len(text) * 100:.1f}%)")

    parts = []
    for pct, snippet_text in snippets:
        parts.append(f"[~{pct:.0f}%] {snippet_text}")
    snippets_text = "\n\n---\n\n".join(parts)

    prompt = OUTLINE_PROMPT.format(n=len(snippets), snippets_text=snippets_text)

    print("  Invoking Gemini CLI for structured outline...")
    outline = await run_gemini_cli(prompt)
    t1 = time.time()
    print(f"  Outline generated in {t1 - t0:.2f}s ({len(outline):,} chars)")
    return outline


# ---------------------------------------------------------------------------
# Dynamic keyword extraction from outline
# ---------------------------------------------------------------------------

STATIC_KEYWORDS = [
    "major plot progression action",
    "character emotional shift dialogue",
    "bureaucracy exposition history",
    "systemic threat or conflict",
]

KEYWORD_EXTRACTION_PROMPT = """Extract retrieval keywords from this novel outline for semantic search.

{outline}

Output a JSON array of 8-12 search queries, each 3-8 words. Include:
- 2-3 queries with major character names + their key actions/relationships
- 2-3 queries about central conflicts and plot turning points
- 2-3 queries about thematic elements and narrative tension
- 1-2 queries about the climax/resolution

Return ONLY a JSON array of strings, no other text:
["query 1", "query 2", ...]"""


async def extract_dynamic_keywords(outline):
    """Extract book-specific retrieval keywords from the structured outline."""
    t0 = time.time()
    print("\n--- Extracting dynamic keywords from outline ---")

    prompt = KEYWORD_EXTRACTION_PROMPT.format(outline=outline)
    result = await run_gemini_cli(prompt)

    keywords = None
    for match in re.finditer(r'\[.*?\]', result, re.DOTALL):
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                keywords = parsed
                break
        except json.JSONDecodeError:
            continue

    if not keywords:
        print("  Warning: Could not parse dynamic keywords, falling back to static")
        return STATIC_KEYWORDS

    t1 = time.time()
    print(f"  Extracted {len(keywords)} dynamic keywords in {t1 - t0:.2f}s:")
    for kw in keywords:
        print(f"    - {kw}")
    return keywords

async def synthesize_frameworks(full_jsonl):
    t0 = time.time()
    print("\n--- Phase 3: Synthesizing Frameworks ---")

    frameworks = [
        "Todorov's Equilibrium",
        "Actantial Model",
        "Quadrant Scores",
        "Lévi-Strauss's Binary Oppositions",
        "Cognitive Estrangement",
        "Bakhtin's Chronotope",
        "Aristotelian Poetics",
        "Jungian Archetypal Analysis",
        "Genette's Transtextuality"
    ]

    async def synthesize_single(fw):
        fw_start = time.time()
        print(f"Synthesizing: {fw}...")

        prompt = f"""You are a Synthesis Sub-Agent specializing in {fw}.

Here is the structural event timeline of the novel:
{full_jsonl}

Task:
Analyze the timeline and output ONLY the {fw} mapping in valid JSON format.
"""
        if fw == "Quadrant Scores":
            prompt += """
Specifically, output exactly 6 floats between 0.0 and 1.0 for these metrics based on the timeline's pacing, conflict types, and plot structure:
- time_linearity: 0.0=Linear, 1.0=Fractured
- pacing_velocity: 0.0=Action-Driven, 1.0=Observational
- threat_scale: 0.0=Individual, 1.0=Systemic
- protagonist_fate: 0.0=Victory, 1.0=Assimilation
- conflict_style: 0.0=Western Combat, 1.0=Kishōtenketsu
- price_type: 0.0=Physical, 1.0=Ideological

Format the output strictly as JSON:
{
  "time_linearity": 0.0,
  "pacing_velocity": 0.0,
  "threat_scale": 0.0,
  "protagonist_fate": 0.0,
  "conflict_style": 0.0,
  "price_type": 0.0
}
"""
        res = await run_gemini_cli(prompt)
        fw_time = time.time() - fw_start
        print(f"Completed: {fw} in {fw_time:.2f}s")
        return f"### {fw}\n{res}\n"

    tasks = [synthesize_single(fw) for fw in frameworks]
    results = await asyncio.gather(*tasks)
    t1 = time.time()
    print(f"Phase 3 completed in {t1-t0:.2f}s")
    return "\n".join(results)


async def main():
    parser = argparse.ArgumentParser(description="NTSMR Pipeline — Narrative Topology Semantic Map Reduce")
    parser.add_argument("book_file", help="Path to the raw text file")
    parser.add_argument("output_file", help="Path to save the output")

    args = parser.parse_args()

    start_time = time.time()

    with open(args.book_file, "r", encoding="utf-8") as f:
        text = f.read()

    macro_chunks = [text[i:i+150000] for i in range(0, len(text), 150000)]
    print(f"Loaded {args.book_file}: {len(text):,} characters, {len(macro_chunks)} macro-chunks.")

    # 1. Stratified Global Outline
    global_outline = await generate_global_outline(text)

    # 2. Combined keywords (static + dynamic from outline)
    dynamic_kw = await extract_dynamic_keywords(global_outline)
    keywords = STATIC_KEYWORDS + dynamic_kw
    print(f"  Combined: {len(STATIC_KEYWORDS)} static + {len(dynamic_kw)} dynamic = {len(keywords)} keywords")

    # 3. Parallel Fan-Out Chunk Extraction
    print(f"\n--- Phase 2: Parallel Fan-Out Chunk Extraction ({len(macro_chunks)} chunks) ---")
    tasks = []
    for chunk_idx, macro_chunk in enumerate(macro_chunks):
        micro_chunks = [macro_chunk[i:i+4000] for i in range(0, len(macro_chunk), 4000)]
        tasks.append(process_chunk(chunk_idx + 1, micro_chunks, keywords, global_outline, args.book_file))

    chunk_results = await asyncio.gather(*tasks)
    full_jsonl = "\n".join(chunk_results)
    all_lines = [l for l in full_jsonl.split("\n") if l.strip()]
    print(f"\nPhase 2 complete: {len(all_lines)} total events.")

    # 4. Framework Synthesis
    synthesis_result = await synthesize_frameworks(full_jsonl)
    final_output = full_jsonl + "\n\n=== SYNTHESIS ===\n" + synthesis_result

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(final_output)

    end_time = time.time()
    print(f"\nDone! Pipeline completed in {end_time - start_time:.2f} seconds.")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    asyncio.run(main())
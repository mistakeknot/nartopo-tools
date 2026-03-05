#!/usr/bin/env python3
"""Improved Phase 1 Outline Experiment — better global outline, identical Phase 2 & 3.

Changes ONLY Phase 1 (global outline generation) while keeping Phase 2 extraction and
Phase 3 synthesis identical to production semantic_map_reduce.py. Uses fixed 150K
macro-chunks to isolate the outline variable.

Strategies:
  - stratified: 30 evenly-spaced snippets (opening + closing + 28 distributed)
  - distributed: 30 snippets without forced opening/closing anchors
  - edge: baseline reproduction (first+last 1500 chars per macro-chunk)

Two-pass option (--two-pass): Uses FAISS to find gap-filling snippets based on
the initial outline, then regenerates with additional context.

Usage:
    uv run scripts/experiments/experiment_improved_outline.py <book.txt> <output.jsonl> \
        --strategy stratified \
        --two-pass \
        --output-dir DIR
"""

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

from experiment_adaptive_chunking import adaptive_chunk


# ---------------------------------------------------------------------------
# Embedding helpers (shared with other experiments)
# ---------------------------------------------------------------------------

def get_nomic_embedding_sync(text):
    payload = {"model": "nomic-embed-text", "prompt": text}
    ollama_host = os.environ.get("OLLAMA_HOST", "http://100.107.177.128:11434")

    req = urllib.request.Request(
        f"{ollama_host}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(req, timeout=10)
        return json.loads(response.read())["embedding"]
    except Exception:
        if "100.107.177.128" in ollama_host:
            req = urllib.request.Request(
                "http://localhost:11434/api/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                response = urllib.request.urlopen(req, timeout=10)
                return json.loads(response.read())["embedding"]
            except Exception:
                pass
        return []


async def get_nomic_embedding(text):
    return await asyncio.to_thread(get_nomic_embedding_sync, text)


# ---------------------------------------------------------------------------
# Gemini CLI wrapper
# ---------------------------------------------------------------------------

async def run_gemini_cli(prompt, model=None):
    cmd = ["gemini", "-y"]
    if model:
        cmd.extend(["-m", model])
    cmd.extend(["-p", prompt])

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return stdout.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Phase 1: Improved Global Outline
# ---------------------------------------------------------------------------

SNIPPET_SIZE = 1500
N_SNIPPETS = 30


def select_snippets_stratified(text, n_snippets=N_SNIPPETS, snippet_size=SNIPPET_SIZE):
    """Select evenly-spaced snippets with guaranteed opening and closing coverage.

    Returns list of (position_pct, snippet_text) tuples sorted by position.
    """
    text_len = len(text)
    if text_len <= snippet_size * 2:
        return [(0.0, text)]

    snippets = []

    # Guarantee opening
    snippets.append((0.0, text[:snippet_size]))
    # Guarantee closing
    snippets.append((100.0, text[-snippet_size:]))

    # Fill remaining slots evenly across the text
    n_middle = n_snippets - 2
    step = text_len // (n_middle + 1)
    for i in range(1, n_middle + 1):
        start = i * step
        # Clamp to avoid overlap with closing snippet
        start = min(start, text_len - snippet_size)
        pct = (start / text_len) * 100
        snippets.append((pct, text[start : start + snippet_size]))

    snippets.sort(key=lambda x: x[0])
    return snippets


def select_snippets_distributed(text, n_snippets=N_SNIPPETS, snippet_size=SNIPPET_SIZE):
    """Select evenly-spaced snippets without forced opening/closing anchors."""
    text_len = len(text)
    if text_len <= snippet_size * 2:
        return [(0.0, text)]

    snippets = []
    step = text_len // n_snippets
    for i in range(n_snippets):
        start = i * step
        start = min(start, text_len - snippet_size)
        pct = (start / text_len) * 100
        snippets.append((pct, text[start : start + snippet_size]))

    return snippets


def select_snippets_edge(text, macro_chunks):
    """Baseline reproduction: first+last 1500 chars per macro-chunk (production approach)."""
    snippets = []
    for chunk_idx, chunk in enumerate(macro_chunks):
        micro_chunks = [chunk[i : i + 4000] for i in range(0, len(chunk), 4000)]
        if len(micro_chunks) > 0:
            pct = (text.index(chunk[:100]) / len(text)) * 100 if len(text) > 0 else 0
            pct = min(pct, 100.0)
            snippets.append((pct, micro_chunks[0][:1500]))
        if len(micro_chunks) > 1:
            end_start = text.index(chunk[:100]) + len(chunk) if len(text) > 0 else 0
            pct = min((end_start / len(text)) * 100, 100.0)
            snippets.append((pct, micro_chunks[-1][-1500:]))
    snippets.sort(key=lambda x: x[0])
    return snippets[:30]


def format_snippets(snippets):
    """Format snippets with position labels for the prompt."""
    parts = []
    for pct, text in snippets:
        parts.append(f"[~{pct:.0f}%] {text}")
    return "\n\n---\n\n".join(parts)


IMPROVED_OUTLINE_PROMPT = """You are a master literary structuralist performing a fast-pass structural survey.
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


async def generate_improved_outline(text, strategy, macro_chunks,
                                     n_snippets=N_SNIPPETS, snippet_size=SNIPPET_SIZE):
    """Generate a structured global outline using the specified sampling strategy."""
    t0 = time.time()
    print(f"\n--- Phase 1: Generating Improved Global Outline (strategy={strategy}) ---")

    if strategy == "stratified":
        snippets = select_snippets_stratified(text, n_snippets, snippet_size)
    elif strategy == "distributed":
        snippets = select_snippets_distributed(text, n_snippets, snippet_size)
    elif strategy == "edge":
        snippets = select_snippets_edge(text, macro_chunks)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    coverage = sum(len(s) for _, s in snippets)
    print(f"  Snippets: {len(snippets)}, coverage: {coverage:,} chars ({coverage / len(text) * 100:.1f}%)")

    snippets_text = format_snippets(snippets)
    prompt = IMPROVED_OUTLINE_PROMPT.format(n=len(snippets), snippets_text=snippets_text)

    print("  Invoking Gemini CLI for structured outline...")
    outline = await run_gemini_cli(prompt)
    t1 = time.time()
    print(f"  Outline generated in {t1 - t0:.2f}s ({len(outline):,} chars)")
    return outline


# ---------------------------------------------------------------------------
# Phase 1b: Two-pass FAISS gap-filling
# ---------------------------------------------------------------------------

GAP_QUERIES = [
    "climax turning point crisis peak tension",
    "resolution ending denouement final outcome conclusion",
    "character relationships alliances betrayals bonds",
    "temporal shifts flashbacks time jumps memories past",
    "revelations secrets discoveries twists surprises",
]


async def generate_two_pass_outline(text, strategy, macro_chunks, book_file,
                                     n_snippets=N_SNIPPETS, snippet_size=SNIPPET_SIZE):
    """Two-pass outline: initial outline -> FAISS gap-filling -> refined outline."""
    # Pass 1: Generate initial outline
    initial_outline = await generate_improved_outline(
        text, strategy, macro_chunks, n_snippets, snippet_size
    )

    print("\n--- Phase 1b: Two-pass gap-filling with FAISS ---")
    t0 = time.time()

    # Build FAISS index over all 4K micro-chunks of full text
    micro_size = 4000
    all_micro_chunks = [text[i : i + micro_size] for i in range(0, len(text), micro_size)]
    print(f"  Building FAISS index over {len(all_micro_chunks)} micro-chunks...")

    # Check for cached full-text index
    cache_base = f"{book_file}_improved_fulltext"
    faiss_path = f"{cache_base}.faiss"
    chunks_path = f"{cache_base}_chunks.json"

    if os.path.exists(faiss_path) and os.path.exists(chunks_path):
        print(f"  Loading cached full-text index from {faiss_path}")
        index = faiss.read_index(faiss_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            valid_micro_chunks = json.load(f)
    else:
        tasks = [get_nomic_embedding(chunk) for chunk in all_micro_chunks]
        embeddings = await asyncio.gather(*tasks)

        valid_data = [(m, e) for m, e in zip(all_micro_chunks, embeddings) if e]
        if not valid_data:
            print("  Warning: No embeddings generated, falling back to single-pass")
            return initial_outline

        valid_micro_chunks, valid_embeddings = zip(*valid_data)
        valid_micro_chunks = list(valid_micro_chunks)

        embeddings_array = np.array(valid_embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(embeddings_array.shape[1])
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)

        # Cache
        faiss.write_index(index, faiss_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(valid_micro_chunks, f)

    print(f"  Full-text index: {len(valid_micro_chunks)} chunks indexed")

    # Run gap queries to find targeted snippets
    gap_snippets = []
    for query in GAP_QUERIES:
        q_emb = await get_nomic_embedding(query)
        if not q_emb:
            continue
        q_vec = np.array([q_emb], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        distances, indices = index.search(q_vec, 5)
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(valid_micro_chunks):
                # Find approximate position in original text
                chunk_text = valid_micro_chunks[idx]
                pos = text.find(chunk_text[:200])
                pct = (pos / len(text) * 100) if pos >= 0 else 50.0
                gap_snippets.append((pct, chunk_text[:snippet_size], dist))

    # Deduplicate by keeping unique snippets (avoid overlap)
    seen_positions = set()
    unique_gap_snippets = []
    for pct, snippet, dist in sorted(gap_snippets, key=lambda x: -x[2]):
        # Bucket to nearest 2% to avoid near-duplicates
        bucket = round(pct / 2) * 2
        if bucket not in seen_positions:
            seen_positions.add(bucket)
            unique_gap_snippets.append((pct, snippet))
        if len(unique_gap_snippets) >= 15:
            break

    print(f"  Gap-filling: {len(unique_gap_snippets)} targeted snippets from {len(GAP_QUERIES)} queries")

    # Pass 2: Regenerate outline with initial outline + gap snippets
    gap_text = format_snippets(sorted(unique_gap_snippets, key=lambda x: x[0]))

    pass2_prompt = f"""You are a master literary structuralist refining a structural survey.

Here is a ROUGH OUTLINE generated from an initial pass:
{initial_outline}

Here are {len(unique_gap_snippets)} TARGETED SNIPPETS retrieved to fill gaps in the initial outline
(labeled with position percentages):
{gap_text}

Using both the rough outline and these targeted snippets, produce a REFINED structured outline:

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

Be concrete. Use actual names. Correct any errors from the rough outline using the new snippets."""

    print("  Invoking Gemini CLI for refined outline (pass 2)...")
    refined = await run_gemini_cli(pass2_prompt)
    t1 = time.time()
    print(f"  Two-pass outline completed in {t1 - t0:.2f}s ({len(refined):,} chars)")
    return refined


# ---------------------------------------------------------------------------
# Phase 2: Parallel chunk extraction (IDENTICAL to production)
# ---------------------------------------------------------------------------

async def process_chunk(chunk_idx, micro_chunks, keywords, global_outline, cache_base):
    """Process a single macro-chunk. Identical to production semantic_map_reduce.py."""
    chunk_start = time.time()
    print(f"[{chunk_idx}] Generating or loading embeddings...")

    faiss_path = f"{cache_base}.faiss"
    chunks_path = f"{cache_base}_chunks.json"

    valid_micro_chunks = []

    emb_start = time.time()
    if os.path.exists(faiss_path) and os.path.exists(chunks_path):
        print(f"[{chunk_idx}] Loading cached embeddings from {faiss_path}")
        index = faiss.read_index(faiss_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            valid_micro_chunks = json.load(f)
    else:
        tasks = [get_nomic_embedding(p) for p in micro_chunks]
        embeddings = await asyncio.gather(*tasks)

        valid_data = [(m, e) for m, e in zip(micro_chunks, embeddings) if e]
        if not valid_data:
            return ""

        valid_micro_chunks, valid_embeddings = zip(*valid_data)
        valid_micro_chunks = list(valid_micro_chunks)

        embeddings_array = np.array(valid_embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(embeddings_array.shape[1])
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)

        faiss.write_index(index, faiss_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(valid_micro_chunks, f)

    emb_time = time.time() - emb_start
    print(f"[{chunk_idx}] Embeddings ready in {emb_time:.2f}s ({len(valid_micro_chunks)} micro-chunks)")

    # Semantic retrieval via FAISS
    snippets = []
    kw_tasks = [get_nomic_embedding(kw) for kw in keywords]
    kw_embeddings = await asyncio.gather(*kw_tasks)

    for kw_emb in kw_embeddings:
        if not kw_emb:
            continue
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
    for line in output.split("\n"):
        if line.startswith("## JSONL"):
            in_jsonl = True
            continue
        if in_jsonl and line.strip().startswith("{"):
            jsonl_lines.append(line.strip())

    chunk_time = time.time() - chunk_start
    print(
        f"[{chunk_idx}] Completed in {chunk_time:.2f}s (Ext: {ext_time:.2f}s). "
        f"Extracted {len(jsonl_lines)} events."
    )
    return "\n".join(jsonl_lines)


# ---------------------------------------------------------------------------
# Phase 3: Quadrant score synthesis (IDENTICAL to production)
# ---------------------------------------------------------------------------

async def synthesize_quadrant_scores(full_jsonl):
    t0 = time.time()
    print("\n--- Phase 3: Synthesizing Quadrant Scores ---")

    prompt = f"""You are a Synthesis Sub-Agent specializing in Quadrant Scores.

Here is the structural event timeline of the novel:
{full_jsonl}

Task:
Analyze the timeline and output ONLY the Quadrant Scores mapping in valid JSON format.

Specifically, output exactly 6 floats between 0.0 and 1.0 for these metrics based on the timeline's pacing, conflict types, and plot structure:
- time_linearity: 0.0=Linear, 1.0=Fractured
- pacing_velocity: 0.0=Action-Driven, 1.0=Observational
- threat_scale: 0.0=Individual, 1.0=Systemic
- protagonist_fate: 0.0=Victory, 1.0=Assimilation
- conflict_style: 0.0=Western Combat, 1.0=Kishotenketsu
- price_type: 0.0=Physical, 1.0=Ideological

Format the output strictly as JSON:
{{
  "time_linearity": 0.0,
  "pacing_velocity": 0.0,
  "threat_scale": 0.0,
  "protagonist_fate": 0.0,
  "conflict_style": 0.0,
  "price_type": 0.0
}}
"""
    result = await run_gemini_cli(prompt)
    t1 = time.time()
    print(f"Phase 3 completed in {t1 - t0:.2f}s")
    return result


def extract_scores_json(text):
    for match in re.finditer(r"\{[^{}]+\}", text):
        try:
            obj = json.loads(match.group())
            if "time_linearity" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def format_yaml_scores(scores, chunking="fixed"):
    key = "combined_outline_adaptive_scores" if chunking == "adaptive" else "improved_outline_scores"
    lines = [f"{key}:"]
    for axis in [
        "time_linearity",
        "pacing_velocity",
        "threat_scale",
        "protagonist_fate",
        "conflict_style",
        "price_type",
    ]:
        val = scores.get(axis)
        if val is not None:
            lines.append(f"  {axis}: {float(val):.1f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Improved Phase 1 Outline Experiment"
    )
    parser.add_argument("book_file", help="Path to the raw text file")
    parser.add_argument("output_file", help="Path to save the output JSONL")
    parser.add_argument(
        "--strategy",
        choices=["stratified", "distributed", "edge"],
        default="stratified",
        help="Snippet selection strategy (default: stratified)",
    )
    parser.add_argument(
        "--two-pass",
        action="store_true",
        help="Enable two-pass FAISS gap-filling",
    )
    parser.add_argument(
        "--chunking",
        choices=["fixed", "adaptive"],
        default="fixed",
        help="Macro-chunking strategy: fixed 150K or adaptive structural boundaries (default: fixed)",
    )
    parser.add_argument(
        "--n-snippets",
        type=int,
        default=N_SNIPPETS,
        help=f"Number of snippets for outline (default: {N_SNIPPETS})",
    )
    parser.add_argument(
        "--snippet-size",
        type=int,
        default=SNIPPET_SIZE,
        help=f"Chars per snippet (default: {SNIPPET_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for intermediate files",
    )

    args = parser.parse_args()

    with open(args.book_file, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Improved Outline Experiment")
    print(f"  Book: {args.book_file}")
    print(f"  Total chars: {len(text):,}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Two-pass: {args.two_pass}")
    print(f"  Chunking: {args.chunking}")
    print(f"  Snippets: {args.n_snippets} x {args.snippet_size} chars")

    start_time = time.time()

    if args.chunking == "adaptive":
        chunks_with_labels = adaptive_chunk(text)
        macro_chunks = [chunk for chunk, _ in chunks_with_labels]
        chunk_labels = [label for _, label in chunks_with_labels]
        print(f"  Macro-chunks: {len(macro_chunks)} (adaptive, boundaries: {len(chunks_with_labels)})")
        for i, (chunk, label) in enumerate(chunks_with_labels):
            preview = chunk[:60].replace("\n", " ").strip()
            print(f"    [{label}] {len(chunk):>7,} chars | {preview}...")
    else:
        chunk_size = 150_000
        macro_chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        chunk_labels = [f"fixed_{i+1}" for i in range(len(macro_chunks))]
        print(f"  Macro-chunks: {len(macro_chunks)} (fixed 150K)")

    keywords = [
        "major plot progression action",
        "character emotional shift dialogue",
        "bureaucracy exposition history",
        "systemic threat or conflict",
    ]

    # --- Phase 1: Improved Global Outline ---
    if args.two_pass:
        global_outline = await generate_two_pass_outline(
            text, args.strategy, macro_chunks, args.book_file,
            args.n_snippets, args.snippet_size,
        )
    else:
        global_outline = await generate_improved_outline(
            text, args.strategy, macro_chunks,
            args.n_snippets, args.snippet_size,
        )

    output_dir = args.output_dir or os.path.dirname(args.output_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Save the outline for inspection
    outline_path = os.path.join(output_dir, "global_outline.md")
    with open(outline_path, "w", encoding="utf-8") as f:
        f.write(global_outline)
    print(f"  Outline saved to {outline_path}")

    # --- Phase 2: Parallel Fan-Out Chunk Extraction ---
    print(f"\n--- Phase 2: Parallel Fan-Out Chunk Extraction ({len(macro_chunks)} chunks, {args.chunking}) ---")

    # Use different cache prefix for adaptive to avoid collisions with fixed-chunk caches
    cache_prefix = "adaptive_chunk" if args.chunking == "adaptive" else "chunk"

    tasks = []
    for chunk_idx, chunk_text in enumerate(macro_chunks):
        micro_chunks = [chunk_text[i : i + 4000] for i in range(0, len(chunk_text), 4000)]
        cache_base = f"{args.book_file}_{cache_prefix}{chunk_idx + 1}"
        tasks.append(
            process_chunk(
                chunk_idx + 1, micro_chunks, keywords, global_outline, cache_base,
            )
        )

    chunk_results = await asyncio.gather(*tasks)

    # Save intermediates
    for chunk_idx, result in enumerate(chunk_results):
        if result.strip():
            intermediate_path = os.path.join(output_dir, f"chunk_{chunk_labels[chunk_idx]}.jsonl")
            with open(intermediate_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"[{chunk_labels[chunk_idx]}] Saved intermediate: {intermediate_path}")

    full_jsonl = "\n".join(chunk_results)
    all_lines = [l for l in full_jsonl.split("\n") if l.strip()]
    print(f"\nPhase 2 complete: {len(all_lines)} total events.")

    # --- Phase 3: Quadrant score synthesis (IDENTICAL to production) ---
    synthesis_result = await synthesize_quadrant_scores(full_jsonl)

    scores = extract_scores_json(synthesis_result)

    # Write output
    final_output = full_jsonl + "\n\n=== SYNTHESIS ===\n" + synthesis_result
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(final_output)

    end_time = time.time()
    print(f"\nDone! Completed in {end_time - start_time:.2f} seconds.")
    print(f"Results saved to {args.output_file}")

    if scores:
        yaml_block = format_yaml_scores(scores, args.chunking)
        print(f"\n--- YAML for data file injection ---")
        print(yaml_block)
        print(f"---")
    else:
        print("\nWarning: Could not extract quadrant scores from synthesis output.")


if __name__ == "__main__":
    asyncio.run(main())

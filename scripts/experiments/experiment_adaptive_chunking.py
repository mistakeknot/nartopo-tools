#!/usr/bin/env python3
"""Adaptive Chunking Experiment — splits at structural boundaries instead of fixed 150K chars.

Hypothesis: Splitting mid-scene loses structural context. Chapter-aware chunking keeps
narrative units intact, improving Phase 2 extraction quality and Phase 3 synthesis accuracy.

Usage:
    uv run scripts/experiments/experiment_adaptive_chunking.py <book.txt> <output.jsonl> \
        --output-dir /tmp/adaptive_output
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


# ---------------------------------------------------------------------------
# Structural boundary detection
# ---------------------------------------------------------------------------

# Patterns that indicate chapter/section boundaries (ordered by specificity)
# These handle both clean line-start markers AND epub-extracted texts with indentation
CHAPTER_PATTERNS = [
    # Explicit chapter markers — allow leading whitespace (epub indentation)
    re.compile(r"^\s*(?:CHAPTER|Chapter)\s+\w+", re.MULTILINE),
    # Part markers
    re.compile(r"^\s*(?:PART|Part)\s+\w+", re.MULTILINE),
    # Book/Section markers
    re.compile(r"^\s*(?:BOOK|Book|SECTION|Section)\s+\w+", re.MULTILINE),
    # Prologue/Epilogue (with optional leading whitespace)
    re.compile(r"^\s*(?:PROLOGUE|Prologue|EPILOGUE|Epilogue)\b", re.MULTILINE),
    # Numbered sections (e.g., "1.", "XIII.")
    re.compile(r"^\s*(?:[IVX]+|[0-9]+)\.\s*$", re.MULTILINE),
    # Named sections (all caps, standalone line, 3-40 chars, with optional indentation)
    re.compile(r"^\s*[A-Z][A-Z\s]{2,39}\s*$", re.MULTILINE),
]

# Inline patterns for epub texts where entire book is on few long lines
# These match chapter markers embedded within long lines
INLINE_CHAPTER_PATTERNS = [
    # "Chapter 1" or "CHAPTER ONE" preceded by sentence-ending punctuation + space
    re.compile(r'(?<=[.!?"\x27])\s+(?:CHAPTER|Chapter)\s+\w+'),
    # "Prologue" / "Epilogue" preceded by sentence end
    re.compile(r'(?<=[.!?"\x27])\s+(?:Prologue|Epilogue|PROLOGUE|EPILOGUE)\b'),
    # TOC-style markers: "One [filename.html#chap-N]"
    re.compile(r'\b(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen)\s*\[[\w.#-]+\]'),
    # Roman numeral TOC markers: "XIII [filename.html#chap-N]"
    re.compile(r'\b[IVX]+\s*\[[\w.#-]+\]'),
]

# Scene break patterns (secondary boundaries, used if chapters are too large)
SCENE_BREAK_PATTERNS = [
    re.compile(r"^\s*\*\s*\*\s*\*\s*$", re.MULTILINE),  # ***
    re.compile(r"^---+\s*$", re.MULTILINE),               # ---
    re.compile(r"^\s*\*{3,}\s*$", re.MULTILINE),           # ***
    re.compile(r"^\s*~{3,}\s*$", re.MULTILINE),            # ~~~
]

TARGET_MIN = 80_000    # Minimum chunk size (chars)
TARGET_MAX = 200_000   # Maximum chunk size (chars)
ABSOLUTE_MAX = 250_000 # Never exceed this


def find_boundaries(text):
    """Find all structural boundaries in the text. Returns sorted list of char offsets."""
    boundaries = set()

    # Try line-start chapter patterns first
    for pattern in CHAPTER_PATTERNS:
        for match in pattern.finditer(text):
            boundaries.add(match.start())

    # If few boundaries found, also try inline patterns (for epub long-line texts)
    if len(boundaries) < 3:
        for pattern in INLINE_CHAPTER_PATTERNS:
            for match in pattern.finditer(text):
                boundaries.add(match.start())

    # Add scene breaks for sub-splitting
    for pattern in SCENE_BREAK_PATTERNS:
        for match in pattern.finditer(text):
            boundaries.add(match.start())

    return sorted(boundaries)


def adaptive_chunk(text, target_min=TARGET_MIN, target_max=TARGET_MAX):
    """Split text at structural boundaries, grouping small segments to hit target range.

    Strategy:
    1. Find all chapter/scene boundaries
    2. Create segments between boundaries
    3. Greedily merge segments until adding the next would exceed target_max
    4. If a single segment exceeds ABSOLUTE_MAX, fall back to fixed splitting within it

    Returns list of (chunk_text, chunk_label) tuples.
    """
    boundaries = find_boundaries(text)

    if len(boundaries) < 2:
        # No structure detected — fall back to fixed 150K with paragraph-aligned splits
        return _fallback_paragraph_split(text)

    # Create segments between boundaries
    # Include start and end of text
    all_points = [0] + boundaries + [len(text)]
    # Deduplicate and sort
    all_points = sorted(set(all_points))

    segments = []
    for i in range(len(all_points) - 1):
        start, end = all_points[i], all_points[i + 1]
        seg_text = text[start:end]
        if seg_text.strip():  # Skip empty segments
            segments.append(seg_text)

    if not segments:
        return _fallback_paragraph_split(text)

    # Greedily merge segments into chunks
    chunks = []
    current = ""
    for seg in segments:
        if len(current) + len(seg) > target_max and len(current) >= target_min:
            # Current chunk is big enough, start a new one
            chunks.append(current)
            current = seg
        elif len(current) + len(seg) > ABSOLUTE_MAX:
            # Would exceed absolute max — flush current, then handle oversized segment
            if current.strip():
                chunks.append(current)
            # If this single segment is huge, sub-split it
            if len(seg) > ABSOLUTE_MAX:
                sub_chunks = _split_oversized(seg, target_max)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = seg
        else:
            current += seg

    if current.strip():
        chunks.append(current)

    # If last chunk is too small, merge with previous
    if len(chunks) > 1 and len(chunks[-1]) < target_min // 2:
        chunks[-2] += chunks[-1]
        chunks.pop()

    return [(chunk, f"adaptive_{i+1}") for i, chunk in enumerate(chunks)]


def _fallback_paragraph_split(text, size=150_000):
    """Split at paragraph boundaries near the target size."""
    chunks = []
    start = 0
    while start < len(text):
        if start + size >= len(text):
            chunks.append(text[start:])
            break
        # Find a paragraph break near the target
        search_start = max(start + size - 5000, start)
        search_end = min(start + size + 5000, len(text))
        region = text[search_start:search_end]
        # Look for double newline (paragraph break)
        para_break = region.rfind("\n\n")
        if para_break != -1:
            split_at = search_start + para_break + 2
        else:
            # Fall back to any newline
            nl = region.rfind("\n")
            split_at = search_start + nl + 1 if nl != -1 else start + size
        chunks.append(text[start:split_at])
        start = split_at

    return [(chunk, f"para_{i+1}") for i, chunk in enumerate(chunks)]


def _split_oversized(text, target_max):
    """Split an oversized segment at scene breaks or paragraphs."""
    # Try scene breaks first
    scene_points = [0]
    for pattern in SCENE_BREAK_PATTERNS:
        for match in pattern.finditer(text):
            scene_points.append(match.start())
    scene_points.append(len(text))
    scene_points = sorted(set(scene_points))

    if len(scene_points) > 2:
        # Merge scene segments to target size
        chunks = []
        current = ""
        for i in range(len(scene_points) - 1):
            seg = text[scene_points[i]:scene_points[i + 1]]
            if len(current) + len(seg) > target_max and current.strip():
                chunks.append(current)
                current = seg
            else:
                current += seg
        if current.strip():
            chunks.append(current)
        return chunks

    # No scene breaks — fall back to paragraph split
    return [c for c, _ in _fallback_paragraph_split(text, target_max)]


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
# Phase 1: Global Outline
# ---------------------------------------------------------------------------

async def generate_global_outline(macro_chunks, keywords):
    t0 = time.time()
    print("\n--- Phase 1: Generating Global Fast-Pass Outline ---")

    global_snippets = []
    for chunk in macro_chunks:
        micro_chunks = [chunk[i : i + 4000] for i in range(0, len(chunk), 4000)]
        if len(micro_chunks) > 0:
            global_snippets.append(micro_chunks[0][:1500])
        if len(micro_chunks) > 1:
            global_snippets.append(micro_chunks[-1][-1500:])

    snippets_text = "\n\n---\n\n".join(global_snippets[:30])

    prompt = f"""You are a master literary structuralist.
I am giving you scattered snippets spanning the beginning, middle, and end of a novel.

Snippets:
{snippets_text}

Task:
Write a comprehensive 1-page Global Outline summarizing the overarching plot, main characters, and central conflicts.
This outline will be broadcast to parallel sub-agents, so it must provide strong orienting context.
"""
    print("Invoking Gemini CLI for Global Outline...")
    outline = await run_gemini_cli(prompt)
    t1 = time.time()
    print(f"Global Outline Generated successfully in {t1 - t0:.2f}s.")
    return outline


# ---------------------------------------------------------------------------
# Phase 2: Parallel chunk extraction
# ---------------------------------------------------------------------------

async def process_chunk(chunk_idx, chunk_label, micro_chunks, keywords, global_outline, cache_base):
    """Process a single structure-aware chunk."""
    chunk_start = time.time()
    print(f"[{chunk_label}] Generating or loading embeddings...")

    # Cache paths — use adaptive-specific prefix to avoid conflicts with fixed chunking
    faiss_path = f"{cache_base}.faiss"
    chunks_path = f"{cache_base}_chunks.json"

    valid_micro_chunks = []

    emb_start = time.time()
    if os.path.exists(faiss_path) and os.path.exists(chunks_path):
        print(f"[{chunk_label}] Loading cached embeddings from {faiss_path}")
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
    print(f"[{chunk_label}] Embeddings ready in {emb_time:.2f}s ({len(valid_micro_chunks)} micro-chunks)")

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

    print(f"[{chunk_label}] Invoking Gemini CLI for extraction...")
    ext_start = time.time()

    schema_instruction = '{"type": "action"|"dialogue"|"exposition"|"bureaucracy", "summary": "Brief 1-sentence summary"}'

    prompt = f"""You are a structural analysis sub-agent analyzing Chunk {chunk_idx} of the novel.

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
        f"[{chunk_label}] Completed in {chunk_time:.2f}s (Ext: {ext_time:.2f}s). "
        f"Extracted {len(jsonl_lines)} events."
    )
    return "\n".join(jsonl_lines)


# ---------------------------------------------------------------------------
# Phase 3: Quadrant score synthesis
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
- conflict_style: 0.0=Western Combat, 1.0=Kishōtenketsu
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


def format_yaml_scores(scores):
    lines = ["adaptive_chunking_scores:"]
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
        description="Adaptive Chunking Experiment — structure-aware splitting"
    )
    parser.add_argument("book_file", help="Path to the raw text file")
    parser.add_argument("output_file", help="Path to save the output JSONL")
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        default=True,
        help="Save per-chunk JSONL files (default: true)",
    )
    parser.add_argument(
        "--no-save-intermediates",
        action="store_true",
        help="Disable saving per-chunk intermediates",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for intermediate files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show chunk boundaries without running the pipeline",
    )

    args = parser.parse_args()
    save_intermediates = args.save_intermediates and not args.no_save_intermediates

    with open(args.book_file, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Adaptive Chunking Experiment")
    print(f"  Book: {args.book_file}")
    print(f"  Total chars: {len(text):,}")

    # Detect boundaries
    boundaries = find_boundaries(text)
    print(f"  Structural boundaries found: {len(boundaries)}")

    # Show detected boundary types
    chapter_count = 0
    scene_count = 0
    for b in boundaries:
        context = text[b:b+80].split("\n")[0].strip()
        is_chapter = any(p.match(context) for p in CHAPTER_PATTERNS)
        if not is_chapter:
            is_chapter = any(p.search(context) for p in INLINE_CHAPTER_PATTERNS)
        if is_chapter:
            chapter_count += 1
        else:
            scene_count += 1
    print(f"  Chapter-level: {chapter_count}, Scene-level: {scene_count}")

    # Perform adaptive chunking
    chunks_with_labels = adaptive_chunk(text)
    print(f"  Adaptive chunks: {len(chunks_with_labels)}")

    # Compare with fixed chunking
    fixed_count = (len(text) + 149999) // 150000
    print(f"  Fixed 150K chunks would be: {fixed_count}")
    print()

    for i, (chunk, label) in enumerate(chunks_with_labels):
        preview = chunk[:80].replace("\n", " ").strip()
        print(f"  [{label}] {len(chunk):>7,} chars | starts: {preview}...")

    if args.dry_run:
        print("\nDry run complete. Use without --dry-run to execute the pipeline.")
        return

    start_time = time.time()
    print()

    # Extract just the chunk texts
    macro_chunks = [chunk for chunk, _ in chunks_with_labels]
    chunk_labels = [label for _, label in chunks_with_labels]

    keywords = [
        "major plot progression action",
        "character emotional shift dialogue",
        "bureaucracy exposition history",
        "systemic threat or conflict",
    ]

    # --- Phase 1: Global Outline ---
    global_outline = await generate_global_outline(macro_chunks, keywords)

    # --- Phase 2: Parallel Fan-Out Chunk Extraction ---
    print(f"\n--- Phase 2: Parallel Fan-Out Chunk Extraction (adaptive chunks) ---")

    output_dir = args.output_dir or os.path.dirname(args.output_file) or "."
    if save_intermediates:
        os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for chunk_idx, (chunk_text, chunk_label) in enumerate(chunks_with_labels):
        micro_chunks = [chunk_text[i : i + 4000] for i in range(0, len(chunk_text), 4000)]
        # Use adaptive-specific cache path to avoid polluting fixed-chunk caches
        cache_base = f"{args.book_file}_adaptive_chunk{chunk_idx + 1}"
        tasks.append(
            process_chunk(
                chunk_idx + 1, chunk_label, micro_chunks, keywords, global_outline,
                cache_base,
            )
        )

    chunk_results = await asyncio.gather(*tasks)

    # Save intermediates
    if save_intermediates:
        for chunk_idx, result in enumerate(chunk_results):
            if result.strip():
                intermediate_path = os.path.join(
                    output_dir,
                    f"chunk_{chunk_labels[chunk_idx]}.jsonl",
                )
                with open(intermediate_path, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"[{chunk_labels[chunk_idx]}] Saved intermediate: {intermediate_path}")

    full_jsonl = "\n".join(chunk_results)

    all_lines = [l for l in full_jsonl.split("\n") if l.strip()]
    print(f"\nPhase 2 complete: {len(all_lines)} total events.")

    # --- Phase 3: Quadrant score synthesis ---
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
        yaml_block = format_yaml_scores(scores)
        print(f"\n--- YAML for data file injection ---")
        print(yaml_block)
        print(f"---")
    else:
        print("\nWarning: Could not extract quadrant scores from synthesis output.")


if __name__ == "__main__":
    asyncio.run(main())

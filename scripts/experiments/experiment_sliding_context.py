#!/usr/bin/env python3
"""Sliding Context Experiment — passes previous chunk's JSONL events as context to the next chunk.

Hypothesis: sequential processing with inter-chunk context reduces quadrant score drift
for non-linear narratives (Banks, Gibson) compared to parallel fan-out.

Usage:
    uv run scripts/experiments/experiment_sliding_context.py <book.txt> <output.jsonl> \
        --context-window 10 --save-intermediates --output-dir /tmp/sliding_ctx
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
# Embedding helpers (copied from semantic_map_reduce.py)
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
# Gemini CLI wrapper (copied from semantic_map_reduce.py)
# ---------------------------------------------------------------------------

async def run_gemini_cli(prompt):
    """Runs the Gemini CLI sub-agent asynchronously."""
    process = await asyncio.create_subprocess_exec(
        "gemini", "-y", "-p", prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return stdout.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Phase 1: Global Outline (copied from semantic_map_reduce.py)
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
# Phase 2: Sequential chunk extraction with sliding context
# ---------------------------------------------------------------------------

async def process_chunk_with_context(
    chunk_idx, micro_chunks, keywords, global_outline, previous_events, book_file
):
    """Process a single macro-chunk, injecting previous chunk's JSONL events for continuity."""
    chunk_start = time.time()
    print(f"[{chunk_idx}] Generating or loading embeddings...")

    # Cache paths — same scheme as semantic_map_reduce.py so caches are shared
    cache_base = f"{book_file}_chunk{chunk_idx}"
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
    print(f"[{chunk_idx}] Embeddings ready in {emb_time:.2f}s")

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

    # Build context injection block
    context_block = ""
    if previous_events:
        events_text = "\n".join(previous_events)
        context_block = f"""
Here are the last {len(previous_events)} structural events from the PREVIOUS chunk (for continuity):
{events_text}

Use these for character/timeline continuity. Do NOT repeat these events — only reference them for context.
"""

    print(f"[{chunk_idx}] Invoking Gemini CLI for extraction (context: {len(previous_events)} events)...")
    ext_start = time.time()

    schema_instruction = '{"type": "action"|"dialogue"|"exposition"|"bureaucracy", "summary": "Brief 1-sentence summary"}'

    prompt = f"""You are a structural analysis sub-agent analyzing Chunk {chunk_idx}.

Here is the GLOBAL OUTLINE of the entire novel to provide you with broad narrative context:
{global_outline}
{context_block}
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
# Phase 3: Quadrant score synthesis (simplified — scores only)
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
    """Extract the first JSON object containing quadrant score keys from LLM output."""
    for match in re.finditer(r"\{[^{}]+\}", text):
        try:
            obj = json.loads(match.group())
            if "time_linearity" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def format_yaml_scores(scores, context_window):
    """Format scores as a YAML block for injection into data files."""
    lines = [f"sliding_context_scores:  # context_window={context_window}"]
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
        description="Sliding Context Experiment — sequential Phase 2 with inter-chunk context"
    )
    parser.add_argument("book_file", help="Path to the raw text file")
    parser.add_argument("output_file", help="Path to save the output JSONL")
    parser.add_argument(
        "--context-window",
        type=int,
        default=10,
        help="Number of JSONL events from previous chunk to pass as context "
        "(default: 10, 0=control/no context, -1=all events)",
    )
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
        help="Directory for intermediate files (default: next to output_file)",
    )

    args = parser.parse_args()
    save_intermediates = args.save_intermediates and not args.no_save_intermediates

    start_time = time.time()
    context_window = args.context_window

    print(f"Sliding Context Experiment")
    print(f"  context_window = {context_window} ({'control' if context_window == 0 else 'all' if context_window == -1 else f'last {context_window} events'})")

    with open(args.book_file, "r", encoding="utf-8") as f:
        text = f.read()

    macro_chunks = [text[i : i + 150000] for i in range(0, len(text), 150000)]
    print(f"Loaded {args.book_file}: {len(text)} characters, {len(macro_chunks)} macro-chunks.")

    keywords = [
        "major plot progression action",
        "character emotional shift dialogue",
        "bureaucracy exposition history",
        "systemic threat or conflict",
    ]

    # --- Phase 1: Global Outline (reused from production pipeline) ---
    global_outline = await generate_global_outline(macro_chunks, keywords)

    # --- Phase 2: Sequential chunk extraction with sliding context ---
    print(f"\n--- Phase 2: Sequential Chunk Extraction (context_window={context_window}) ---")

    # Set up intermediates directory
    output_dir = args.output_dir or os.path.dirname(args.output_file) or "."
    if save_intermediates:
        os.makedirs(output_dir, exist_ok=True)

    all_jsonl_lines = []  # Accumulate all events across chunks
    chunk_results = []

    for chunk_idx, macro_chunk in enumerate(macro_chunks):
        micro_chunks = [macro_chunk[i : i + 4000] for i in range(0, len(macro_chunk), 4000)]

        # Determine context to pass from previous chunks
        if context_window == 0 or not all_jsonl_lines:
            previous_events = []
        elif context_window == -1:
            previous_events = list(all_jsonl_lines)  # All accumulated events
        else:
            previous_events = all_jsonl_lines[-context_window:]

        result = await process_chunk_with_context(
            chunk_idx + 1,
            micro_chunks,
            keywords,
            global_outline,
            previous_events,
            args.book_file,
        )

        chunk_results.append(result)

        # Parse new JSONL lines from this chunk
        new_lines = [l for l in result.split("\n") if l.strip()]
        all_jsonl_lines.extend(new_lines)

        # Save intermediate per-chunk file
        if save_intermediates and result.strip():
            intermediate_path = os.path.join(
                output_dir,
                f"chunk_{chunk_idx + 1}_ctx{context_window}.jsonl",
            )
            with open(intermediate_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"[{chunk_idx + 1}] Saved intermediate: {intermediate_path}")

    # Deduplicate while preserving order
    deduped_lines = list(dict.fromkeys(all_jsonl_lines))
    full_jsonl = "\n".join(deduped_lines)

    print(f"\nPhase 2 complete: {len(all_jsonl_lines)} total events, {len(deduped_lines)} after dedup.")

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
        yaml_block = format_yaml_scores(scores, context_window)
        print(f"\n--- YAML for data file injection ---")
        print(yaml_block)
        print(f"---")
    else:
        print("\nWarning: Could not extract quadrant scores from synthesis output.")
        print("Check the synthesis section in the output file for raw results.")


if __name__ == "__main__":
    asyncio.run(main())

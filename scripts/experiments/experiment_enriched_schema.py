#!/usr/bin/env python3
"""Enriched Schema Experiment — adds structured metadata fields to Phase 2 JSONL events
and pre-synthesis aggregation for better quadrant score accuracy.

Hypothesis: Constrained-enum metadata fields (temporal, agency_delta, conflict_mode, stakes)
give Phase 3 concrete signals instead of forcing it to guess from free-text summaries.
Pre-computed aggregation stats (hard counts/ratios) eliminate LLM counting errors.

Usage:
    uv run scripts/experiments/experiment_enriched_schema.py <book.txt> <output.jsonl> \
        --output-dir /tmp/enriched_output
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
# Gemini CLI wrapper
# ---------------------------------------------------------------------------

async def run_gemini_cli(prompt, model=None):
    """Runs the Gemini CLI sub-agent asynchronously, optionally with a specific model."""
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
# Phase 1: Global Outline (full model)
# ---------------------------------------------------------------------------

async def generate_global_outline(macro_chunks, keywords):
    t0 = time.time()
    print("\n--- Phase 1: Generating Global Fast-Pass Outline (full model) ---")

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
# Phase 2: Parallel chunk extraction with enriched schema
# ---------------------------------------------------------------------------

ENRICHED_SCHEMA = """{
  "type": "action"|"dialogue"|"exposition"|"introspection"|"flashback",
  "temporal": "present"|"flashback"|"foreshadow",
  "agency_delta": "gains"|"loses"|"neutral",
  "conflict_mode": "confrontation"|"negotiation"|"revelation"|"spiral",
  "stakes": "personal"|"systemic",
  "summary": "Brief 1-sentence summary"
}"""


async def process_chunk(chunk_idx, micro_chunks, keywords, global_outline, book_file, model=None):
    """Process a single macro-chunk with enriched schema extraction."""
    chunk_start = time.time()
    model_label = model or "default"
    print(f"[{chunk_idx}] Generating or loading embeddings (model: {model_label})...")

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

    print(f"[{chunk_idx}] Invoking Gemini CLI for enriched extraction (model: {model_label})...")
    ext_start = time.time()

    prompt = f"""You are a structural analysis sub-agent analyzing Chunk {chunk_idx}.

Here is the GLOBAL OUTLINE of the entire novel to provide you with broad narrative context:
{global_outline}

Here are the extracted semantic snippets from YOUR SPECIFIC CHUNK of the novel:
{snippets_text}

Task:
Extract the major structural beats from the snippets and output them as strict JSON Lines (JSONL).
Do not hallucinate events outside these snippets. Use the global outline only for understanding context and character identities.

Each event MUST include ALL of these fields:
- type: One of "action", "dialogue", "exposition", "introspection", "flashback"
- temporal: One of "present", "flashback", "foreshadow" — is this event happening in the narrative present, recalled from the past, or anticipating the future?
- agency_delta: One of "gains", "loses", "neutral" — does the protagonist gain power/agency/knowledge, lose it, or neither? IMPORTANT: A character who is effective and competent but being used as a tool by others scores "gains" (they are still winning/succeeding), NOT "loses". "loses" means actual loss of identity, autonomy, or absorption into something larger.
- conflict_mode: One of "confrontation", "negotiation", "revelation", "spiral" — is the conflict resolved through direct combat/opposition, through bargaining/diplomacy, through information/discovery, or through escalating complexity without clear resolution?
- stakes: One of "personal", "systemic" — are the stakes affecting individuals or entire systems/societies/civilizations?
- summary: Brief 1-sentence summary of the event

Schema: {ENRICHED_SCHEMA}

Format your response EXACTLY like this (do not use markdown blocks for the JSON):
## JSONL
{{"type": "action", "temporal": "present", "agency_delta": "gains", "conflict_mode": "confrontation", "stakes": "systemic", "summary": "..."}}
"""

    output = await run_gemini_cli(prompt, model=model)
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
# Pre-synthesis aggregation
# ---------------------------------------------------------------------------

def aggregate_signals(jsonl_text):
    """Parse JSONL events and compute field distribution stats for Phase 3.

    Returns a human-readable stats block with counts and ratios.
    Gracefully skips events with missing or invalid fields.
    """
    counts = {
        "type": {},
        "temporal": {},
        "agency_delta": {},
        "conflict_mode": {},
        "stakes": {},
    }

    total = 0
    for line in jsonl_text.split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        total += 1
        for field in counts:
            val = event.get(field)
            if val and isinstance(val, str):
                counts[field][val] = counts[field].get(val, 0) + 1

    if total == 0:
        return "No events to aggregate."

    lines = [f"Total events: {total}", ""]

    # Type distribution + observational ratio
    type_counts = counts["type"]
    type_parts = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
    observational = type_counts.get("introspection", 0) + type_counts.get("exposition", 0)
    active = type_counts.get("action", 0) + type_counts.get("dialogue", 0)
    denom = observational + active
    obs_ratio = observational / denom if denom > 0 else 0.5
    lines.append(f"type: {type_parts} (observational_ratio={obs_ratio:.2f})")

    # Temporal distribution + fractured ratio
    temp_counts = counts["temporal"]
    temp_parts = ", ".join(f"{k}={v}" for k, v in sorted(temp_counts.items()))
    fractured = temp_counts.get("flashback", 0) + temp_counts.get("foreshadow", 0)
    temp_total = sum(temp_counts.values())
    frac_ratio = fractured / temp_total if temp_total > 0 else 0.0
    lines.append(f"temporal: {temp_parts} (fractured_ratio={frac_ratio:.2f})")

    # Agency delta + victory ratio
    agency_counts = counts["agency_delta"]
    agency_parts = ", ".join(f"{k}={v}" for k, v in sorted(agency_counts.items()))
    gains = agency_counts.get("gains", 0)
    loses = agency_counts.get("loses", 0)
    gl_denom = gains + loses
    victory_ratio = gains / gl_denom if gl_denom > 0 else 0.5
    lines.append(f"agency_delta: {agency_parts} (victory_ratio={victory_ratio:.2f})")

    # Conflict mode + western ratio
    conflict_counts = counts["conflict_mode"]
    conflict_parts = ", ".join(f"{k}={v}" for k, v in sorted(conflict_counts.items()))
    confrontation = conflict_counts.get("confrontation", 0)
    spiral = conflict_counts.get("spiral", 0)
    conf_total = sum(conflict_counts.values())
    western_ratio = confrontation / conf_total if conf_total > 0 else 0.5
    spiral_ratio = spiral / conf_total if conf_total > 0 else 0.0
    lines.append(f"conflict_mode: {conflict_parts} (western_ratio={western_ratio:.2f}, spiral_ratio={spiral_ratio:.2f})")

    # Stakes + systemic ratio
    stakes_counts = counts["stakes"]
    stakes_parts = ", ".join(f"{k}={v}" for k, v in sorted(stakes_counts.items()))
    systemic = stakes_counts.get("systemic", 0)
    stakes_total = sum(stakes_counts.values())
    systemic_ratio = systemic / stakes_total if stakes_total > 0 else 0.5
    lines.append(f"stakes: {stakes_parts} (systemic_ratio={systemic_ratio:.2f})")

    # Cross-reference: price_type signal
    ideological_signal = 0
    physical_signal = 0
    for line_raw in jsonl_text.split("\n"):
        line_raw = line_raw.strip()
        if not line_raw.startswith("{"):
            continue
        try:
            event = json.loads(line_raw)
        except json.JSONDecodeError:
            continue
        etype = event.get("type", "")
        estakes = event.get("stakes", "")
        if etype in ("exposition", "introspection") and estakes == "systemic":
            ideological_signal += 1
        if etype == "action" and estakes == "personal":
            physical_signal += 1

    pi_denom = ideological_signal + physical_signal
    ideological_ratio = ideological_signal / pi_denom if pi_denom > 0 else 0.5
    lines.append(f"price_type cross-ref: ideological_events={ideological_signal}, physical_events={physical_signal} (ideological_ratio={ideological_ratio:.2f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 3: Enriched quadrant score synthesis (full model)
# ---------------------------------------------------------------------------

async def synthesize_quadrant_scores(full_jsonl, aggregation_stats):
    t0 = time.time()
    print("\n--- Phase 3: Synthesizing Quadrant Scores with Aggregation Stats (full model) ---")

    prompt = f"""You are a Synthesis Sub-Agent specializing in Quadrant Scores.

Here are the pre-computed signal distributions from the structural event timeline:
{aggregation_stats}

Here is the full structural event timeline:
{full_jsonl}

Task: Map the distributions to exactly 6 scores (0.0-1.0).

Scoring methodology:
- time_linearity: fractured_ratio directly maps. 0.0 if all present, 1.0 if heavily flashback/foreshadow. Use the fractured_ratio as a strong anchor but adjust based on narrative structure in the events.
- pacing_velocity: observational_ratio maps. High introspection+exposition = observational (1.0). High action+dialogue = action-driven (0.0).
- threat_scale: systemic_ratio maps. Mostly systemic stakes = 1.0 (Systemic). Mostly personal stakes = 0.0 (Individual).
- protagonist_fate: victory_ratio INVERSELY maps. High gains = Victory (0.0). High loses = Assimilation (1.0). IMPORTANT: A character who is effective/competent but used as a tool still scores LOW (Victory) — assimilation (1.0) requires actual loss of identity, autonomy, or absorption into something larger. Being traumatized or in a dark narrative does NOT mean assimilation if the character remains effective and victorious.
- conflict_style: western_ratio maps to 0.0 (Western Combat). High spiral = Kishōtenketsu (1.0). Negotiation and revelation fall in the middle.
- price_type: Use the ideological_ratio from the cross-reference. Physical action + personal stakes = Physical (0.0). Ideological exposition + systemic stakes = Ideological (1.0).

Output strictly as JSON (no markdown, no explanation):
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


def format_yaml_scores(scores):
    """Format scores as a YAML block for injection into data files."""
    lines = ["enriched_schema_scores:"]
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
        description="Enriched Schema Experiment — structured metadata + pre-synthesis aggregation"
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
        help="Directory for intermediate files (default: next to output_file)",
    )

    args = parser.parse_args()
    save_intermediates = args.save_intermediates and not args.no_save_intermediates

    start_time = time.time()

    print(f"Enriched Schema Experiment")
    print(f"  Phase 1 (outline):      default model")
    print(f"  Phase 2 (extraction):   default model (enriched schema)")
    print(f"  Phase 2.5 (aggregate):  deterministic Python")
    print(f"  Phase 3 (synthesis):    default model (with aggregation stats)")

    with open(args.book_file, "r", encoding="utf-8") as f:
        text = f.read()

    macro_chunks = [text[i : i + 150000] for i in range(0, len(text), 150000)]
    print(f"Loaded {args.book_file}: {len(text)} characters, {len(macro_chunks)} macro-chunks.")

    keywords = [
        "major plot progression action",
        "character emotional shift dialogue",
        "bureaucracy exposition history",
        "systemic threat or conflict",
        "introspection reflection identity",
        "flashback memory past",
    ]

    # --- Phase 1: Global Outline (full model) ---
    global_outline = await generate_global_outline(macro_chunks, keywords)

    # --- Phase 2: Parallel Fan-Out Chunk Extraction (enriched schema) ---
    print(f"\n--- Phase 2: Parallel Fan-Out Chunk Extraction (enriched schema) ---")

    output_dir = args.output_dir or os.path.dirname(args.output_file) or "."
    if save_intermediates:
        os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for chunk_idx, macro_chunk in enumerate(macro_chunks):
        micro_chunks = [macro_chunk[i : i + 4000] for i in range(0, len(macro_chunk), 4000)]
        tasks.append(
            process_chunk(
                chunk_idx + 1, micro_chunks, keywords, global_outline,
                args.book_file,
            )
        )

    chunk_results = await asyncio.gather(*tasks)

    # Save intermediates
    if save_intermediates:
        for chunk_idx, result in enumerate(chunk_results):
            if result.strip():
                intermediate_path = os.path.join(
                    output_dir,
                    f"chunk_{chunk_idx + 1}_enriched.jsonl",
                )
                with open(intermediate_path, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"[{chunk_idx + 1}] Saved intermediate: {intermediate_path}")

    full_jsonl = "\n".join(chunk_results)

    # Count events
    all_lines = [l for l in full_jsonl.split("\n") if l.strip()]
    print(f"\nPhase 2 complete: {len(all_lines)} total events.")

    # --- Phase 2.5: Pre-synthesis aggregation (deterministic) ---
    print("\n--- Phase 2.5: Pre-Synthesis Aggregation ---")
    aggregation_stats = aggregate_signals(full_jsonl)
    print(aggregation_stats)

    if save_intermediates:
        agg_path = os.path.join(output_dir, "aggregation_stats.txt")
        with open(agg_path, "w", encoding="utf-8") as f:
            f.write(aggregation_stats)
        print(f"Saved aggregation stats: {agg_path}")

    # --- Phase 3: Quadrant score synthesis (full model, with aggregation) ---
    synthesis_result = await synthesize_quadrant_scores(full_jsonl, aggregation_stats)

    scores = extract_scores_json(synthesis_result)

    # Write output
    final_output = (
        full_jsonl
        + "\n\n=== AGGREGATION STATS ===\n"
        + aggregation_stats
        + "\n\n=== SYNTHESIS ===\n"
        + synthesis_result
    )
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
        print("Check the synthesis section in the output file for raw results.")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Embedding Model Comparison Experiment (nt-jgii).

Evaluates different Ollama embedding models for FAISS retrieval quality
by measuring downstream quadrant score MAE. Uses the best-performing pipeline
configuration (stratified outline + combined keywords) and only varies the
embedding model.

Models tested:
  - nomic-embed-text (768-dim, 8192 ctx, 274MB) — current production baseline
  - nomic-embed-text-v2-moe (768-dim, 8192 ctx, 957MB) — MoE upgrade
  - mxbai-embed-large (1024-dim, 512 ctx, 669MB) — higher MTEB but short ctx
  - snowflake-arctic-embed2 (1024-dim, 8192 ctx, 1.2GB) — strong retrieval

Usage:
    uv run scripts/experiments/experiment_embedding_models.py <book.txt> <output.jsonl> \
        --model nomic-embed-text-v2-moe \
        --output-dir DIR

    # Run all models:
    uv run scripts/experiments/experiment_embedding_models.py <book.txt> <output.jsonl> \
        --model all \
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

# Import shared utilities from the improved outline experiment
sys.path.insert(0, os.path.dirname(__file__))
from experiment_improved_outline import (
    select_snippets_stratified,
    generate_improved_outline,
    extract_dynamic_keywords,
    run_gemini_cli,
    extract_scores_json,
    STATIC_KEYWORDS,
)

# ---------------------------------------------------------------------------
# Models to evaluate
# ---------------------------------------------------------------------------

MODELS = {
    "nomic-embed-text": {"dim": 768, "ctx": 8192, "label": "nomic_v1"},
    "nomic-embed-text-v2-moe": {"dim": 768, "ctx": 8192, "label": "nomic_v2_moe"},
    "mxbai-embed-large": {"dim": 1024, "ctx": 512, "label": "mxbai_large"},
    "snowflake-arctic-embed2": {"dim": 1024, "ctx": 8192, "label": "snowflake_arctic2"},
}

# ---------------------------------------------------------------------------
# Embedding helpers — parameterized by model name
# ---------------------------------------------------------------------------

# Semaphore to limit concurrent Ollama requests (GPU processes sequentially)
_OLLAMA_SEM = None


def _get_sem():
    global _OLLAMA_SEM
    if _OLLAMA_SEM is None:
        _OLLAMA_SEM = asyncio.Semaphore(8)
    return _OLLAMA_SEM


def get_embedding_sync(text, model_name):
    """Get embedding from Ollama using the /api/embed endpoint (handles long text)."""
    payload = {"model": model_name, "input": text}
    ollama_host = os.environ.get("OLLAMA_HOST", "http://100.107.177.128:11434")

    req = urllib.request.Request(
        f"{ollama_host}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(req, timeout=120)
        data = json.loads(response.read())
        return data.get("embeddings", [[]])[0]
    except Exception as e:
        if "100.107.177.128" in ollama_host:
            req = urllib.request.Request(
                "http://localhost:11434/api/embed",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                response = urllib.request.urlopen(req, timeout=120)
                data = json.loads(response.read())
                return data.get("embeddings", [[]])[0]
            except Exception:
                pass
        print(f"    [embed-error] {model_name}: {e}", file=sys.stderr)
        return []


async def get_embedding(text, model_name):
    async with _get_sem():
        return await asyncio.to_thread(get_embedding_sync, text, model_name)


# ---------------------------------------------------------------------------
# Phase 2: Chunk extraction with parameterized embedding model
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an expert structural analyst. Given these relevant text snippets from a novel section, extract a structural event timeline.

Global Outline (for context and character identification):
{outline}

Relevant text snippets:
{snippets}

For each significant narrative event in these snippets, output one JSON line:
{{"type": "action"|"dialogue"|"exposition"|"bureaucracy", "summary": "Brief 1-sentence summary"}}

Output ONLY valid JSONL lines, one per event. No other text."""


async def process_chunk(chunk_idx, micro_chunks, keywords, global_outline,
                        cache_base, model_name):
    """Process a single macro-chunk with a specific embedding model."""
    chunk_start = time.time()
    print(f"  [{chunk_idx}] Embedding {len(micro_chunks)} micro-chunks with {model_name}...")

    faiss_path = f"{cache_base}.faiss"
    chunks_path = f"{cache_base}_chunks.json"

    valid_micro_chunks = []

    emb_start = time.time()
    if os.path.exists(faiss_path) and os.path.exists(chunks_path):
        print(f"  [{chunk_idx}] Loading cached embeddings from {faiss_path}")
        index = faiss.read_index(faiss_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            valid_micro_chunks = json.load(f)
    else:
        tasks = [get_embedding(p, model_name) for p in micro_chunks]
        embeddings = await asyncio.gather(*tasks)

        valid_data = [(m, e) for m, e in zip(micro_chunks, embeddings) if e]
        if not valid_data:
            print(f"  [{chunk_idx}] Warning: no embeddings returned")
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
    print(f"  [{chunk_idx}] Embeddings ready in {emb_time:.2f}s")

    # Retrieve snippets via keyword queries
    snippets = []
    kw_tasks = [get_embedding(kw, model_name) for kw in keywords]
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

    prompt = EXTRACTION_PROMPT.format(outline=global_outline, snippets=snippets_text)
    result = await run_gemini_cli(prompt)

    elapsed = time.time() - chunk_start
    n_events = len([l for l in result.split("\n") if l.strip()])
    print(f"  [{chunk_idx}] Done in {elapsed:.2f}s — {n_events} events")
    return result


# ---------------------------------------------------------------------------
# Phase 3: Quadrant score synthesis (identical to production)
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """You are a narrative structure analyst synthesizing a structural event timeline into quantitative scores.

Here is the complete structural event timeline of a novel:
{jsonl}

Analyze the timeline and produce scores on 6 axes (0.0 to 1.0):

1. time_linearity: 0.0 = heavily fractured/non-linear, 1.0 = strictly chronological
2. pacing_velocity: 0.0 = slow/contemplative, 1.0 = fast/action-driven
3. threat_scale: 0.0 = personal/domestic, 1.0 = existential/cosmic
4. protagonist_fate: 0.0 = total defeat/death, 1.0 = total victory/transcendence
5. conflict_style: 0.0 = internal/psychological, 1.0 = external/physical
6. price_type: 0.0 = no cost/free victory, 1.0 = pyrrhic/devastating cost

Output ONLY a JSON object with these 6 keys and float values. No other text."""


async def synthesize_quadrant_scores(jsonl_text):
    prompt = SYNTHESIS_PROMPT.format(jsonl=jsonl_text)
    return await run_gemini_cli(prompt)


# ---------------------------------------------------------------------------
# Run experiment for a single model
# ---------------------------------------------------------------------------

async def run_single_model(book_file, text, model_name, model_info,
                           global_outline, keywords, output_dir):
    """Run the full Phase 2 + Phase 3 pipeline with a specific embedding model."""
    label = model_info["label"]
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({model_info['dim']}-dim, {model_info['ctx']} ctx)")
    print(f"{'='*60}")

    model_dir = os.path.join(output_dir, label)
    os.makedirs(model_dir, exist_ok=True)

    # Macro-chunk the text
    chunk_size = 150_000
    macro_chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"  Macro-chunks: {len(macro_chunks)} (fixed 150K)")

    # Phase 2: Extract with this embedding model
    t0 = time.time()
    tasks = []
    for chunk_idx, chunk_text in enumerate(macro_chunks):
        micro_chunks = [chunk_text[i : i + 4000] for i in range(0, len(chunk_text), 4000)]
        # Model-specific cache path to avoid cross-contamination
        cache_base = os.path.join(model_dir, f"{os.path.basename(book_file)}_chunk{chunk_idx + 1}")
        tasks.append(
            process_chunk(
                chunk_idx + 1, micro_chunks, keywords, global_outline,
                cache_base, model_name,
            )
        )

    chunk_results = await asyncio.gather(*tasks)

    full_jsonl = "\n".join(chunk_results)
    all_lines = [l for l in full_jsonl.split("\n") if l.strip()]
    t1 = time.time()
    print(f"  Phase 2 complete: {len(all_lines)} events in {t1-t0:.2f}s")

    # Save JSONL
    jsonl_path = os.path.join(model_dir, "events.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(full_jsonl)

    # Phase 3: Synthesize quadrant scores
    synthesis_result = await synthesize_quadrant_scores(full_jsonl)
    scores = extract_scores_json(synthesis_result)

    # Save synthesis
    synth_path = os.path.join(model_dir, "synthesis.txt")
    with open(synth_path, "w", encoding="utf-8") as f:
        f.write(synthesis_result)

    if scores:
        print(f"  Scores: {json.dumps(scores)}")
    else:
        print(f"  Warning: Could not extract scores")

    return {
        "model": model_name,
        "label": label,
        "dim": model_info["dim"],
        "ctx": model_info["ctx"],
        "n_events": len(all_lines),
        "scores": scores,
        "time_s": t1 - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Embedding Model Comparison Experiment (nt-jgii)"
    )
    parser.add_argument("book_file", help="Path to the raw text file")
    parser.add_argument("output_file", help="Path to save the combined output")
    parser.add_argument(
        "--model",
        default="all",
        help="Model to test: nomic-embed-text, nomic-embed-text-v2-moe, "
             "mxbai-embed-large, snowflake-arctic-embed2, or 'all' (default: all)",
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

    output_dir = args.output_dir or os.path.dirname(args.output_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    print(f"Embedding Model Experiment (nt-jgii)")
    print(f"  Book: {args.book_file}")
    print(f"  Total chars: {len(text):,}")

    # Determine which models to test
    if args.model == "all":
        models_to_test = MODELS
    elif args.model in MODELS:
        models_to_test = {args.model: MODELS[args.model]}
    else:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available: {', '.join(MODELS.keys())}, all")
        sys.exit(1)

    print(f"  Models: {', '.join(models_to_test.keys())}")

    start_time = time.time()

    # Phase 1: Generate outline (shared across all models — outline doesn't use embeddings)
    macro_chunks_for_outline = [text[i : i + 150_000] for i in range(0, len(text), 150_000)]
    global_outline = await generate_improved_outline(
        text, "stratified", macro_chunks_for_outline
    )

    outline_path = os.path.join(output_dir, "global_outline.md")
    with open(outline_path, "w", encoding="utf-8") as f:
        f.write(global_outline)
    print(f"  Outline saved to {outline_path}")

    # Generate combined keywords (best method)
    dynamic_kw = await extract_dynamic_keywords(global_outline)
    keywords = STATIC_KEYWORDS + dynamic_kw
    print(f"  Keywords: {len(STATIC_KEYWORDS)} static + {len(dynamic_kw)} dynamic = {len(keywords)} combined")

    # Run each model sequentially (they share the GPU, can't parallelize)
    results = []
    for model_name, model_info in models_to_test.items():
        result = await run_single_model(
            args.book_file, text, model_name, model_info,
            global_outline, keywords, output_dir,
        )
        results.append(result)

    # Summary
    end_time = time.time()
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {end_time - start_time:.1f}s\n")

    print(f"{'Model':<30s} {'Dim':>4s} {'Ctx':>5s} {'Events':>6s} {'Time':>6s}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<30s} {r['dim']:>4d} {r['ctx']:>5d} {r['n_events']:>6d} {r['time_s']:>5.1f}s")

    print(f"\n{'Model':<30s} {'tl':>5s} {'pv':>5s} {'ts':>5s} {'pf':>5s} {'cs':>5s} {'pt':>5s}")
    print("-" * 60)
    for r in results:
        if r["scores"]:
            s = r["scores"]
            print(f"{r['model']:<30s} {s.get('time_linearity','-'):>5} {s.get('pacing_velocity','-'):>5} "
                  f"{s.get('threat_scale','-'):>5} {s.get('protagonist_fate','-'):>5} "
                  f"{s.get('conflict_style','-'):>5} {s.get('price_type','-'):>5}")
        else:
            print(f"{r['model']:<30s}   (no scores)")

    # Save results JSON
    results_path = os.path.join(output_dir, "embedding_comparison.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Output YAML blocks for injection
    print(f"\n--- YAML blocks for data file injection ---")
    for r in results:
        if r["scores"]:
            key = f"embedding_{r['label']}_scores"
            print(f"{key}:")
            for axis in ["time_linearity", "pacing_velocity", "threat_scale",
                          "protagonist_fate", "conflict_style", "price_type"]:
                val = r["scores"].get(axis)
                if val is not None:
                    print(f"  {axis}: {float(val):.1f}")
    print("---")


if __name__ == "__main__":
    asyncio.run(main())

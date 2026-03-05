# 5. Pipeline Efficiency & NTSMR 2.2

For accurate Nartopo analysis on large texts, you must balance context limits with structural fidelity.

Our research (see `../research/token-efficiency-experiments.md`) demonstrated that simple RAG extraction destroys the ability to measure pacing, while brute-forcing the entire text is computationally and financially expensive.

To achieve optimal token-efficiency without sacrificing structural accuracy, use the **NTSMR 2.2 (Narrative Topology Semantic Map Reduce)** methodology.

## The NTSMR 2.2 Pipeline

This is a **Hybrid Parallel Architecture** combining:
1. **Embedding Caching:** Generates and caches `nomic-embed-text` FAISS micro-indexes locally per chunk, making subsequent runs instantaneous.
2. **Stratified Global Outline:** Samples opening, ending, and evenly spaced interior snippets to build a global outline plus a condensed character/reference context.
3. **Precomputed Keyword Retrieval:** Deduplicates static plus dynamic retrieval keywords and embeds them once per run, rather than once per chunk.
4. **Parallel Fan-Out with Evidence IDs:** Processes each 150k-character chunk into validated JSONL events carrying `event_id`, `chunk_id`, `snippet_ids`, structural type, and canonical framework signals.
5. **Routed Framework Synthesis:** Framework agents receive the smallest relevant event subset possible and must cite `evidence_event_ids` in a strict synthesis object.
6. **Split Machine Artifacts:** The pipeline emits strict `events.jsonl`, `synthesis.json`, and `report.json` artifacts so downstream tooling can validate and ingest them without regex parsing.

**Usage:**
```bash
uv run scripts/semantic_map_reduce.py "../../books/{Author Name}/{Book Title}.txt" "/tmp/{Book Title}_timeline.jsonl"
```

This command writes:
- `/tmp/{Book Title}_timeline.jsonl` — strict event stream
- `/tmp/{Book Title}_timeline.synthesis.json` — per-framework synthesis payload
- `/tmp/{Book Title}_timeline.report.json` — routing, coverage, and snippet provenance report

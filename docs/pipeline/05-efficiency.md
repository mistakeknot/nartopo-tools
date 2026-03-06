# 5. Pipeline Efficiency & NTSMR 2.3

For accurate Nartopo analysis on large texts, you must balance context limits with structural fidelity.

Our research (see `../research/token-efficiency-experiments.md`) demonstrated that simple RAG extraction destroys the ability to measure pacing, while brute-forcing the entire text is computationally and financially expensive.

To achieve optimal token-efficiency without sacrificing structural accuracy, use the **NTSMR 2.3 (Narrative Topology Semantic Map Reduce)** methodology.

## The NTSMR 2.3 Pipeline

This is a **Hybrid Parallel Architecture** combining:
1. **Embedding Caching:** Generates and caches `nomic-embed-text` FAISS micro-indexes locally per chunk, making subsequent runs instantaneous.
2. **Stratified Global Outline:** Samples opening, ending, and evenly spaced interior snippets to build a global outline plus a condensed character/reference context.
3. **Precomputed Keyword Retrieval:** Deduplicates static plus dynamic retrieval keywords and embeds them once per run, rather than once per chunk.
4. **Parallel Fan-Out with Evidence IDs:** Processes each 150k-character chunk into validated JSONL events carrying `event_id`, `chunk_id`, `snippet_ids`, structural type, and canonical framework signals.
5. **Reusable Narrative Substrate:** Persists standalone `outline`, `snippets`, `characters`, `dialogue`, and `settings` artifacts, each tied back to stable snippet IDs for later framework work without rerunning extraction.
6. **Routed Framework Synthesis:** Framework agents receive the smallest relevant event subset possible and must cite `evidence_event_ids` in a strict synthesis object.
7. **Split Machine Artifacts:** The pipeline emits strict machine-readable artifacts so downstream tooling can validate and ingest them without regex parsing.

**Usage:**
```bash
uv run scripts/semantic_map_reduce.py "../../books/{Author Name}/{Book Title}.txt" "/tmp/{Book Title}_timeline.jsonl"
```

**Backend overrides:**
```bash
# Default production Gemini run label
uv run scripts/semantic_map_reduce.py "../../books/{Author Name}/{Book Title}.txt" "/tmp/{Book Title}_timeline.jsonl" \
  --run-label NTSMR-2.3-gemini-3.1-pr-preview

# Codex Exec backend
uv run scripts/semantic_map_reduce.py "../../books/{Author Name}/{Book Title}.txt" "/tmp/{Book Title}_timeline.jsonl" \
  --llm-backend codex-exec \
  --llm-model gpt-5.4 \
  --llm-reasoning-effort high \
  --run-label NTSMR-2.3-gpt-5.4-high

# Reuse an existing substrate and rerun only the score synthesis
uv run scripts/semantic_map_reduce.py /dev/null "/tmp/{Book Title}_timeline.ntsmr-2-3-gpt-5-4-high.jsonl" \
  --llm-backend codex-exec \
  --llm-model gpt-5.4 \
  --llm-reasoning-effort high \
  --run-label NTSMR-2.3-gpt-5.4-high \
  --reuse-from "/tmp/{Book Title}_timeline.jsonl" \
  --score-only
```

This command writes:
- `/tmp/{Book Title}_timeline.outline.json` — stratified outline samples, condensed outline context, and retrieval keywords
- `/tmp/{Book Title}_timeline.snippets.jsonl` — retrieved snippet evidence with scores and source offsets
- `/tmp/{Book Title}_timeline.characters.jsonl` — grounded character mentions and roles
- `/tmp/{Book Title}_timeline.dialogue.jsonl` — grounded dialogue-turn records
- `/tmp/{Book Title}_timeline.settings.jsonl` — grounded setting records
- `/tmp/{Book Title}_timeline.jsonl` — strict event stream
- `/tmp/{Book Title}_timeline.synthesis.json` — per-framework synthesis payload
- `/tmp/{Book Title}_timeline.report.json` — routing, coverage, and snippet provenance report

The machine-readable payloads include `ntsmr_run_label` and backend/model metadata so multiple `2.3` runs can coexist without overwriting each other. Reused runs also record `source_run_label`, and score-only comparison runs write only the `Quadrant Scores` synthesis block while copying the saved substrate artifacts forward.

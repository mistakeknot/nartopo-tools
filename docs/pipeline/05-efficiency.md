# 5. Pipeline Efficiency & Semantic Map-Reduce

For accurate Nartopo analysis on large texts, you must balance context limits with structural fidelity.

Our research (see `../research/token-efficiency-experiments.md`) demonstrated that simple RAG extraction destroys the ability to measure pacing, while brute-forcing the entire text is computationally and financially expensive.

To achieve optimal token-efficiency without sacrificing structural accuracy, use the **Semantic Map-Reduce** methodology.

## The Semantic Map-Reduce Pipeline

This is a Meta-Hybrid approach combining:
1. **Map-Reduce:** Splitting the massive text into 150k-character macro-chunks.
2. **Local FAISS Micro-Indexing:** Semantically retrieving only the structural connective tissue within each chunk using `nomic-embed-text` to drastically cut token volume.
3. **Sliding-Window Context:** Passing a running summary between sequential sub-agents to guarantee narrative continuity across the chunk boundaries.
4. **JSONL Representation:** Forcing sub-agents to output strict JSON Lines (`{"type": "action|dialogue", "summary": "..."}`) to eliminate LLM "vibe" bias and allow the primary agent to algorithmically calculate pacing based on tag ratios.

**Usage:**
```bash
python3 scripts/semantic_map_reduce.py "../../books/{Author Name}/{Book Title}.txt" "/tmp/{Book Title}_timeline.jsonl"
```

*This method guarantees >40% token savings on massive epics while perfectly matching brute-force accuracy.*
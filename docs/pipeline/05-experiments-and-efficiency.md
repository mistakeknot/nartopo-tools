# 5. Token Efficiency & Context-Window Experiments

When performing structural analysis on large books (e.g., 300k+ characters), brute-force context loading becomes incredibly expensive. We have run several experiments to optimize this:

## 1. The Baseline: Brute-Force Context Loading
* **Method:** Reading the entire raw text into a single agent's context window.
* **Result:** Produces the highest fidelity, "ground truth" structural analysis, capturing all micro-pacing anomalies and deep thematic resonances.
* **Cost:** Extremely high (e.g., ~880k tokens for *Excession*, 385k for *Solaris*).

## 2. The Solution: Map-Reduce Parallel Sub-Agents (`map_reduce.sh`)
* **Method:** Splitting the text into 150k-character chunks and spawning headless `gemini` sub-agents to extract localized structural data simultaneously.
* **Result:** **Perfect match with brute-force.** By synthesizing the localized findings, the primary agent reconstructs an analysis that perfectly mirrors the brute-force baseline, without exceeding individual context window limits or sacrificing reasoning quality.
* **Cost:** High (distributes the token cost horizontally).

## 3. Failed Experiment: `tldr-swinton`
* **Method:** Attempted to use the `tldr-swinton` AST/indentation compressor (`--compress chunk-summary`).
* **Result:** **Incompatible.** The tool explicitly expects code files. It cannot parse or compress hundreds of thousands of characters of unformatted narrative prose without destroying the text or crashing.

## 4. Failed Experiment: Target Semantic Extraction (Local RAG)
* **Method:** Chunked the book and used local embedding models (`all-MiniLM-L6-v2` and `nomic-embed-text`) to retrieve snippets matching structural keywords (e.g., "inciting incident", "climax").
* **Result:** **Structurally Inadequate.** While this successfully dropped token consumption by 68-92% (e.g., extracting only 70k-280k characters), it completely destroyed the agent's ability to analyze pacing, narrative duration, and atmospheric tone. It surfaced the main plot beats but missed the "boring" connective tissue that defines the actual structure of the novel. RAG is excellent for factual QA, but useless for holistic structural mapping.

## 5. The Ultimate Optimization: The "Grand Unification" Meta-Hybrid Pipeline
* **Method:** Combining the successes of Map-Reduce with sliding-window continuous context, local FAISS micro-indexing, and rigid JSONL data structures.
* **Result:** **A scalable, mathematically rigorous structural timeline.** The sub-agents output highly compressed, strict JSON Lines of events tagged by structural category (e.g. `action`, `dialogue`). This eliminates LLM "vibe" bias from pacing analysis entirely—the final agent simply algorithms the ratio of the tags. The results were nearly identical to the brute force method, achieving the same 0.85 Pacing Velocity ("Observational") score for *Solaris* based on tag ratios, but using a fraction of the context window per chunk.
* **Cost:** Low / Highly Scalable.

---
**Conclusion:** For optimal balance of token-efficiency and structural accuracy, use the **Grand Unification** methodology (see `experiment_grand_unification.py`).
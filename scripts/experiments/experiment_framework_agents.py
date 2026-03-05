#!/usr/bin/env python3
"""Framework-Specialized Phase 2/3 Sub-Agents Experiment (nt-oh6f).

Enriches Phase 2 extraction with framework_signals tags (zero extra Gemini calls)
and adds structured per-framework synthesis prompts in Phase 3.

Phase 1: Reuses stratified outline from experiment_improved_outline.py
Phase 2: Enriched extraction schema with framework_signals field
Phase 3: 17 parallel framework synthesis sub-agents with structured prompts

Usage:
    uv run scripts/experiments/experiment_framework_agents.py <book.txt> <output.jsonl> \\
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
# Embedding helpers (shared)
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
# Framework signal vocabulary
# ---------------------------------------------------------------------------

FRAMEWORK_SIGNALS = {
    "todorov": ["equilibrium", "disruption", "recognition", "repair", "new_equilibrium"],
    "freytag": ["exposition", "rising_action", "climax", "falling_action", "denouement"],
    "actantial": ["subject", "object", "sender", "receiver", "helper", "opponent"],
    "three_act": ["setup", "confrontation", "resolution"],
    "monomyth": ["ordinary_world", "call", "threshold", "ordeal", "return"],
    "harmon": ["you", "need", "go", "search", "find", "take", "return", "change"],
    "save_the_cat": ["opening_image", "catalyst", "debate", "break_into_two", "midpoint", "all_is_lost", "finale", "final_image"],
    "propp": ["absentation", "interdiction", "violation", "villainy", "departure", "donor", "struggle", "victory", "return", "recognition", "punishment", "wedding"],
    "kishotenketsu": ["ki", "sho", "ten", "ketsu"],
    "protocol": ["rule", "failure", "insight"],
    "genette_narrative": ["order", "duration", "frequency", "mood", "voice"],
    "levi_strauss": ["binary", "mediator"],
    "estrangement": ["familiar", "mechanism", "shift"],
    "bakhtin": ["spatial", "temporal", "intersection"],
    "aristotle": ["hamartia", "peripeteia", "anagnorisis", "catharsis"],
    "jung": ["persona", "shadow", "anima", "animus", "trickster", "self"],
    "transtextuality": ["inter", "para", "meta", "hyper", "archi"],
}

SIGNAL_VOCABULARY_TEXT = "\n".join(
    f"- {fw}:{{{','.join(tags)}}}"
    for fw, tags in FRAMEWORK_SIGNALS.items()
)


# ---------------------------------------------------------------------------
# Phase 3: Structured framework synthesis prompts
# ---------------------------------------------------------------------------

FRAMEWORK_SYNTHESIS_PROMPTS = {
    "Todorov's Equilibrium": {
        "filter_tag": "todorov",
        "prompt_suffix": """Identify the five stages of Todorov's narrative equilibrium model.

Output as JSON:
{{
  "equilibrium": "Description of the starting status quo",
  "disruption": "The inciting incident or protocol failure",
  "recognition": "When the protagonist realizes the disruption",
  "repair": "The attempt to fix or survive it",
  "new_equilibrium": "The new, altered status quo"
}}""",
    },
    "Actantial Model": {
        "filter_tag": "actantial",
        "prompt_suffix": """Identify the six actantial roles from Greimas's model.

Output as JSON:
{{
  "subject": "The protagonist pursuing the central goal",
  "object": "What the subject seeks or desires",
  "sender": "The force/entity that initiates or compels the quest",
  "receiver": "Who/what benefits from the quest's completion",
  "helper": "Allies and enabling forces",
  "opponent": "Antagonistic forces and obstacles"
}}""",
    },
    "Quadrant Scores": {
        "filter_tag": None,
        "prompt_suffix": """Specifically, output exactly 6 floats between 0.0 and 1.0 for these metrics based on the timeline's pacing, conflict types, and plot structure:
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
}}""",
    },
    "The Freytag Pyramid": {
        "filter_tag": "freytag",
        "prompt_suffix": """Map the narrative arc to Freytag's five dramatic stages.

Output as JSON:
{{
  "exposition": "Setup of world, characters, initial situation",
  "rising_action": "Key complications and escalation toward the turning point",
  "climax": "The decisive turning point or moment of highest tension",
  "falling_action": "Consequences and unwinding after the climax",
  "denouement": "Final resolution and the new state of affairs"
}}""",
    },
    "The Three-Act Structure": {
        "filter_tag": "three_act",
        "prompt_suffix": """Map the narrative to the three-act structure, identifying the two key plot points.

Output as JSON:
{{
  "act_1_setup": "The world, characters, and status quo before the inciting incident",
  "plot_point_1": "The event that launches the protagonist into the central conflict",
  "act_2_confrontation": "The escalating obstacles, complications, and midpoint reversal",
  "plot_point_2": "The crisis that forces the final confrontation",
  "act_3_resolution": "The climax and its aftermath"
}}""",
    },
    "The Monomyth": {
        "filter_tag": "monomyth",
        "prompt_suffix": """Analyze how this narrative relates to Campbell's Hero's Journey. Focus on how the work subverts or departs from the monomyth template.

Output as JSON:
{{
  "applicable_stages": ["List the monomyth stages that appear in this narrative"],
  "subversions": "How does this work depart from or subvert the traditional Hero's Journey?"
}}""",
    },
    "Dan Harmon's Story Circle": {
        "filter_tag": "harmon",
        "prompt_suffix": """Analyze the narrative through Dan Harmon's 8-step Story Circle. Focus on 'The Take' -- the price paid for the journey.

Output as JSON:
{{
  "circle_stages": {{
    "you": "Character in comfort zone",
    "need": "What they want/need",
    "go": "Entering unfamiliar territory",
    "search": "Adapting to the new situation",
    "find": "Getting what they wanted",
    "take": "The price paid",
    "return": "Going back to familiar territory",
    "change": "How they have changed"
  }},
  "the_take": "The specific price paid -- what was lost or sacrificed"
}}""",
    },
    "Save the Cat! Beat Sheet": {
        "filter_tag": "save_the_cat",
        "prompt_suffix": """Map the narrative to Blake Snyder's Save the Cat beat sheet. Focus on where the pacing deviates from the prescribed beat timing.

Output as JSON:
{{
  "beats_present": ["List the Save the Cat beats that appear"],
  "pacing_deviations": "Where and how does the pacing diverge from the expected beat sheet timing?"
}}""",
    },
    "Propp's Morphology": {
        "filter_tag": "propp",
        "prompt_suffix": """Identify which of Propp's narrative functions (narratemes) appear in this story.

Output as JSON:
{{
  "applicable_narratemes": ["List each Proppian function present, e.g. 'Absentation', 'Villainy', 'Departure', 'Struggle', 'Victory'"]
}}""",
    },
    "Kishotenketsu": {
        "filter_tag": "kishotenketsu",
        "prompt_suffix": """Analyze how well the four-act Kishotenketsu structure applies to this narrative. This structure emphasizes a twist/shift (ten) rather than conflict-driven drama.

Output as JSON:
{{
  "ki": "Introduction -- the initial situation and characters",
  "sho": "Development -- deepening without introducing conflict",
  "ten": "Twist -- the surprising shift or new perspective",
  "ketsu": "Conclusion -- reconciliation of the twist with the established narrative",
  "applicability": "How well does Kishotenketsu fit this narrative vs. Western conflict-driven models?"
}}""",
    },
    "Protocol Fiction Mapping": {
        "filter_tag": "protocol",
        "prompt_suffix": """Analyze this narrative through the Protocol Fiction lens (Summer of Protocols). Identify the rules/protocols the narrative renders, how they fail, and what human insight emerges.

Output as JSON:
{{
  "rule": "The protocol, rule, or system the narrative renders visible",
  "failure_mode": "How the protocol fails or is subverted",
  "human_insight": "The human truth revealed through the protocol's failure"
}}""",
    },
    "Genette's Narrative Discourse": {
        "filter_tag": "genette_narrative",
        "prompt_suffix": """Analyze the narrative discourse using Genette's categories.

Output as JSON:
{{
  "order": "How is story time arranged vs. discourse time? (analepsis, prolepsis, linear)",
  "duration": "Pacing techniques: scene, summary, ellipsis, pause, stretch",
  "focalization": "Who perceives? Zero focalization (omniscient), internal (character-bound), external (camera)"
}}""",
    },
    "Levi-Strauss's Binary Oppositions": {
        "filter_tag": "levi_strauss",
        "prompt_suffix": """Identify the central binary oppositions that structure this narrative's meaning.

Output as JSON:
{{
  "primary_binary": "The dominant opposition (e.g., nature/culture, individual/collective)",
  "secondary_binary": "A supporting opposition that reinforces or complicates the primary",
  "mediator": "The character, event, or concept that mediates or resolves the opposition"
}}""",
    },
    "Cognitive Estrangement": {
        "filter_tag": "estrangement",
        "prompt_suffix": """Analyze through Suvin/Shklovsky's cognitive estrangement: how does the narrative make the familiar strange to produce new understanding?

Output as JSON:
{{
  "familiar_concept": "The real-world concept or system being estranged",
  "estranging_mechanism": "The speculative element (novum) that defamiliarizes it",
  "cognitive_shift": "The new understanding or critical perspective produced"
}}""",
    },
    "Bakhtin's Chronotope": {
        "filter_tag": "bakhtin",
        "prompt_suffix": """Analyze the chronotope -- the fusion of time and space that defines this narrative's world.

Output as JSON:
{{
  "spatial_matrix": "The defining spatial logic (labyrinth, threshold, road, enclosed space, etc.)",
  "temporal_flow": "How time operates (cyclical, linear, geological, subjective, etc.)",
  "intersection": "Where and how space and time fuse to create the narrative's distinctive world-feeling"
}}""",
    },
    "Aristotelian Poetics": {
        "filter_tag": "aristotle",
        "prompt_suffix": """Analyze through Aristotle's Poetics: identify the tragic elements.

Output as JSON:
{{
  "hamartia": "The protagonist's tragic flaw or error of judgment",
  "peripeteia": "The reversal of fortune -- the moment when things turn",
  "anagnorisis": "The moment of critical recognition or discovery"
}}""",
    },
    "Jungian Archetypal Analysis": {
        "filter_tag": "jung",
        "prompt_suffix": """Identify the Jungian archetypes present in the narrative.

Output as JSON:
{{
  "persona": "The public mask or social role characters present",
  "shadow": "The repressed, dark, or denied aspects",
  "anima_animus": "The contrasexual inner figure -- the feminine in masculine or vice versa",
  "trickster": "The agent of chaos, boundary-crossing, or transformation"
}}""",
    },
    "Genette's Transtextuality": {
        "filter_tag": "transtextuality",
        "prompt_suffix": """Analyze the transtextual relationships -- how this text relates to other texts.

Output as JSON:
{{
  "intertextuality": "Direct quotations, allusions, or references to other works",
  "paratextuality": "How titles, epigraphs, prefaces, or cover art frame meaning",
  "metatextuality": "How the text comments on or critiques other texts or its own genre"
}}""",
    },
}


# ---------------------------------------------------------------------------
# Phase 1: Reuse from experiment_improved_outline
# ---------------------------------------------------------------------------

SNIPPET_SIZE = 1500
N_SNIPPETS = 30

def select_snippets_stratified(text, n_snippets=N_SNIPPETS, snippet_size=SNIPPET_SIZE):
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


def format_snippets(snippets):
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


async def generate_improved_outline(text, macro_chunks):
    t0 = time.time()
    print("\n--- Phase 1: Generating Improved Global Outline (stratified) ---")
    snippets = select_snippets_stratified(text)
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
# Dynamic keyword extraction
# ---------------------------------------------------------------------------

KEYWORD_EXTRACTION_PROMPT = """Extract retrieval keywords from this novel outline for semantic search.

{outline}

Output a JSON array of 8-12 search queries, each 3-8 words. Include:
- 2-3 queries with major character names + their key actions/relationships
- 2-3 queries about central conflicts and plot turning points
- 2-3 queries about thematic elements and narrative tension
- 1-2 queries about the climax/resolution

Return ONLY a JSON array of strings, no other text:
["query 1", "query 2", ...]"""

STATIC_KEYWORDS = [
    "major plot progression action",
    "character emotional shift dialogue",
    "bureaucracy exposition history",
    "systemic threat or conflict",
]


async def extract_dynamic_keywords(outline):
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


# ---------------------------------------------------------------------------
# Phase 2: Enriched extraction with framework signals
# ---------------------------------------------------------------------------

ENRICHED_EXTRACTION_PROMPT = """You are a structural analysis sub-agent analyzing Chunk {chunk_idx}.

Here is the GLOBAL OUTLINE of the entire novel to provide you with broad narrative context:
{global_outline}

Here are the extracted semantic snippets from YOUR SPECIFIC CHUNK of the novel:
{snippets_text}

Task:
Extract the major structural beats from the snippets and output them as strict JSON Lines (JSONL).
Do not hallucinate events outside these snippets. Use the global outline only for understanding context and character identities.

For each event, tag it with relevant narrative framework signals from this vocabulary:
{signal_vocabulary}

Schema: {{"type": "action"|"dialogue"|"exposition"|"bureaucracy", "framework_signals": ["todorov:disruption", "freytag:climax"], "summary": "Brief 1-sentence summary"}}

Include 1-5 relevant framework signals per event. Use the format "framework:stage". Omit the field if no signals apply.

Format your response EXACTLY like this (do not use markdown blocks for the JSON):
## JSONL
{{"type": "...", "framework_signals": ["..."], "summary": "..."}}
"""

PLAIN_EXTRACTION_PROMPT = """You are a structural analysis sub-agent analyzing Chunk {chunk_idx}.

Here is the GLOBAL OUTLINE of the entire novel to provide you with broad narrative context:
{global_outline}

Here are the extracted semantic snippets from YOUR SPECIFIC CHUNK of the novel:
{snippets_text}

Task:
Extract the major structural beats from the snippets and output them as strict JSON Lines (JSONL).
Do not hallucinate events outside these snippets. Use the global outline only for understanding context and character identities.

Schema: {{"type": "action"|"dialogue"|"exposition"|"bureaucracy", "summary": "Brief 1-sentence summary"}}

Format your response EXACTLY like this (do not use markdown blocks for the JSON):
## JSONL
{{"type": "...", "summary": "..."}}
"""


async def process_chunk(chunk_idx, micro_chunks, keywords, global_outline, cache_base, enriched=True):
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

    print(f"[{chunk_idx}] Invoking Gemini CLI for {'enriched' if enriched else 'plain'} extraction...")
    ext_start = time.time()

    if enriched:
        prompt = ENRICHED_EXTRACTION_PROMPT.format(
            chunk_idx=chunk_idx,
            global_outline=global_outline,
            snippets_text=snippets_text,
            signal_vocabulary=SIGNAL_VOCABULARY_TEXT,
        )
    else:
        prompt = PLAIN_EXTRACTION_PROMPT.format(
            chunk_idx=chunk_idx,
            global_outline=global_outline,
            snippets_text=snippets_text,
        )

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
# Phase 3: Framework-specialized synthesis
# ---------------------------------------------------------------------------

async def synthesize_all_frameworks(full_jsonl):
    t0 = time.time()
    print("\n--- Phase 3: Synthesizing All Frameworks ---")

    async def synthesize_single(fw_name, fw_config):
        fw_start = time.time()
        print(f"  Synthesizing: {fw_name}...")

        # Filter JSONL by framework signals if available
        filtered_jsonl = full_jsonl
        filter_tag = fw_config.get("filter_tag")
        if filter_tag:
            lines = [l for l in full_jsonl.split('\n')
                     if f'"{filter_tag}:' in l.lower() or f"'{filter_tag}:" in l.lower()]
            if len(lines) >= 3:
                filtered_jsonl = "\n".join(lines)
                print(f"    Filtered to {len(lines)} tagged events (tag: {filter_tag})")
            else:
                print(f"    Only {len(lines)} tagged events for {filter_tag}, using full JSONL")

        prompt = f"""You are a Synthesis Sub-Agent specializing in {fw_name}.

Here is the structural event timeline of the novel:
{filtered_jsonl}

Task:
Analyze the timeline and output ONLY the {fw_name} analysis in valid JSON format.

{fw_config['prompt_suffix']}"""

        result = await run_gemini_cli(prompt)
        fw_time = time.time() - fw_start
        print(f"  Completed: {fw_name} in {fw_time:.2f}s")
        return fw_name, result

    tasks = [synthesize_single(fw, cfg) for fw, cfg in FRAMEWORK_SYNTHESIS_PROMPTS.items()]
    results = await asyncio.gather(*tasks)
    t1 = time.time()
    print(f"Phase 3 completed in {t1 - t0:.2f}s ({len(results)} frameworks)")
    return {name: result for name, result in results}


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------

def analyze_framework_signals(full_jsonl):
    signal_counts = {}
    total_events = 0
    events_with_signals = 0

    for line in full_jsonl.split("\n"):
        if not line.strip().startswith("{"):
            continue
        total_events += 1
        try:
            event = json.loads(line)
            signals = event.get("framework_signals", [])
            if signals:
                events_with_signals += 1
                for sig in signals:
                    fw = sig.split(":")[0] if ":" in sig else sig
                    signal_counts[fw] = signal_counts.get(fw, 0) + 1
        except json.JSONDecodeError:
            continue

    return {
        "total_events": total_events,
        "events_with_signals": events_with_signals,
        "signal_coverage": f"{events_with_signals}/{total_events} ({events_with_signals/max(total_events,1)*100:.0f}%)",
        "per_framework": dict(sorted(signal_counts.items(), key=lambda x: -x[1])),
    }


def extract_scores_json(text):
    for match in re.finditer(r"\{[^{}]+\}", text):
        try:
            obj = json.loads(match.group())
            if "time_linearity" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def extract_json(text):
    """Extract any JSON object from text, including nested objects."""
    # Try simple flat objects first
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue
    # Try nested objects (one level deep)
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Framework-Specialized Sub-Agents Experiment")
    parser.add_argument("book_file", help="Path to the raw text file")
    parser.add_argument("output_file", help="Path to save the output")
    parser.add_argument("--signals", choices=["enriched", "plain"], default="enriched",
                        help="Phase 2 extraction: enriched (with framework_signals) or plain (default: enriched)")
    parser.add_argument("--keywords", choices=["static", "dynamic", "combined"], default="combined",
                        help="Keyword strategy (default: combined)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for intermediate files")

    args = parser.parse_args()

    with open(args.book_file, "r", encoding="utf-8") as f:
        text = f.read()

    print("Framework-Specialized Sub-Agents Experiment")
    print(f"  Book: {args.book_file}")
    print(f"  Total chars: {len(text):,}")
    print(f"  Signals: {args.signals}")
    print(f"  Keywords: {args.keywords}")

    start_time = time.time()

    # Fixed 150K macro-chunks
    chunk_size = 150_000
    macro_chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    chunk_labels = [f"fixed_{i+1}" for i in range(len(macro_chunks))]
    print(f"  Macro-chunks: {len(macro_chunks)} (fixed 150K)")

    # --- Phase 1: Improved Global Outline ---
    global_outline = await generate_improved_outline(text, macro_chunks)

    output_dir = args.output_dir or os.path.dirname(args.output_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    outline_path = os.path.join(output_dir, "global_outline.md")
    with open(outline_path, "w", encoding="utf-8") as f:
        f.write(global_outline)
    print(f"  Outline saved to {outline_path}")

    # --- Keyword selection ---
    if args.keywords == "dynamic":
        keywords = await extract_dynamic_keywords(global_outline)
    elif args.keywords == "combined":
        dynamic_kw = await extract_dynamic_keywords(global_outline)
        keywords = STATIC_KEYWORDS + dynamic_kw
        print(f"  Combined: {len(STATIC_KEYWORDS)} static + {len(dynamic_kw)} dynamic = {len(keywords)} keywords")
    else:
        keywords = STATIC_KEYWORDS

    # --- Phase 2: Enriched Extraction ---
    enriched = args.signals == "enriched"
    print(f"\n--- Phase 2: {'Enriched' if enriched else 'Plain'} Extraction ({len(macro_chunks)} chunks) ---")

    tasks = []
    for chunk_idx, chunk_text in enumerate(macro_chunks):
        micro_chunks = [chunk_text[i : i + 4000] for i in range(0, len(chunk_text), 4000)]
        cache_base = f"{args.book_file}_chunk{chunk_idx + 1}"
        tasks.append(
            process_chunk(chunk_idx + 1, micro_chunks, keywords, global_outline, cache_base, enriched=enriched)
        )

    chunk_results = await asyncio.gather(*tasks)

    for chunk_idx, result in enumerate(chunk_results):
        if result.strip():
            intermediate_path = os.path.join(output_dir, f"chunk_{chunk_labels[chunk_idx]}.jsonl")
            with open(intermediate_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"[{chunk_labels[chunk_idx]}] Saved intermediate: {intermediate_path}")

    full_jsonl = "\n".join(chunk_results)
    all_lines = [l for l in full_jsonl.split("\n") if l.strip()]
    print(f"\nPhase 2 complete: {len(all_lines)} total events.")

    # Analyze framework signals
    if enriched:
        signal_analysis = analyze_framework_signals(full_jsonl)
        print(f"\n--- Framework Signal Analysis ---")
        print(f"  Coverage: {signal_analysis['signal_coverage']}")
        print(f"  Per-framework signal counts:")
        for fw, count in signal_analysis["per_framework"].items():
            print(f"    {fw}: {count}")

        analysis_path = os.path.join(output_dir, "signal_analysis.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(signal_analysis, f, indent=2)

    # --- Phase 3: Framework-specialized synthesis ---
    framework_results = await synthesize_all_frameworks(full_jsonl)

    # Extract and display quadrant scores
    quadrant_result = framework_results.get("Quadrant Scores", "")
    scores = extract_scores_json(quadrant_result)

    # Save all framework results
    results_path = os.path.join(output_dir, "framework_results.json")
    framework_parsed = {}
    for fw_name, raw_result in framework_results.items():
        parsed = extract_json(raw_result)
        framework_parsed[fw_name] = parsed if parsed else raw_result
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(framework_parsed, f, indent=2, ensure_ascii=False)
    print(f"\nFramework results saved to {results_path}")

    # Write full output
    final_output = full_jsonl + "\n\n=== SYNTHESIS ===\n"
    for fw_name, result in framework_results.items():
        final_output += f"\n### {fw_name}\n{result}\n"
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(final_output)

    end_time = time.time()
    print(f"\nDone! Completed in {end_time - start_time:.2f} seconds.")
    print(f"Results saved to {args.output_file}")

    if scores:
        print(f"\n--- Quadrant Scores (YAML) ---")
        key = "framework_agents_scores"
        lines = [f"{key}:"]
        for axis in ["time_linearity", "pacing_velocity", "threat_scale", "protagonist_fate", "conflict_style", "price_type"]:
            val = scores.get(axis)
            if val is not None:
                lines.append(f"  {axis}: {float(val):.1f}")
        print("\n".join(lines))
        print("---")
    else:
        print("\nWarning: Could not extract quadrant scores from synthesis output.")

    # Print framework synthesis summary
    print(f"\n--- Framework Synthesis Summary ---")
    valid_count = 0
    for fw_name in FRAMEWORK_SYNTHESIS_PROMPTS:
        parsed = framework_parsed.get(fw_name)
        if isinstance(parsed, dict):
            valid_count += 1
            print(f"  ✓ {fw_name}")
        else:
            print(f"  ✗ {fw_name} (parse failed)")
    print(f"\n  Valid: {valid_count}/{len(FRAMEWORK_SYNTHESIS_PROMPTS)} frameworks")


if __name__ == "__main__":
    asyncio.run(main())

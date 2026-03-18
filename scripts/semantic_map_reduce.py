#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import threading

import faiss
import numpy as np

NTSMR_VERSION = "2.5"


class TokenAccumulator:
    """Thread-safe accumulator for LLM token usage across a pipeline run."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock if hasattr(self, "_lock") else _noop_ctx():
            self.input_tokens = 0
            self.output_tokens = 0
            self.cache_creation_tokens = 0
            self.cache_read_tokens = 0
            self.total_cost_usd = 0.0
            self.llm_calls = 0

    def add(self, input_tok: int = 0, output_tok: int = 0, cache_creation: int = 0,
            cache_read: int = 0, cost_usd: float = 0.0):
        with self._lock:
            self.input_tokens += input_tok
            self.output_tokens += output_tok
            self.cache_creation_tokens += cache_creation
            self.cache_read_tokens += cache_read
            self.total_cost_usd += cost_usd
            self.llm_calls += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_creation_tokens": self.cache_creation_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "llm_calls": self.llm_calls,
            }


class _noop_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


TOKEN_USAGE = TokenAccumulator()
CACHE_SCHEMA_VERSION = "substrate-v1"
MIN_SOURCE_TEXT_CHARS = int(os.environ.get("NTSMR_MIN_SOURCE_TEXT_CHARS", "1500"))
DEFAULT_GEMINI_MODEL_LABEL = os.environ.get("NTSMR_GEMINI_MODEL_LABEL", "gemini-3.1-pro-preview")
DEFAULT_LLM_BACKEND = os.environ.get("NTSMR_LLM_BACKEND", "gemini")
DEFAULT_CODEX_MODEL = os.environ.get("NTSMR_CODEX_MODEL", "gpt-5.4")
DEFAULT_CODEX_REASONING_EFFORT = os.environ.get("NTSMR_CODEX_REASONING_EFFORT")
DEFAULT_CLAUDE_MODEL = os.environ.get("NTSMR_CLAUDE_MODEL", "claude-sonnet-4-6")
DEFAULT_CLAUDE_REASONING_EFFORT = os.environ.get("NTSMR_CLAUDE_REASONING_EFFORT")
CLAUDE_MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
}
SUPPORTED_LLM_BACKENDS = {"gemini", "codex-exec", "claude"}
SUPPORTED_REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
CLAUDE_REASONING_EFFORTS = {"low", "medium", "high"}

MACRO_CHUNK_SIZE = 150_000
MICRO_CHUNK_SIZE = 4_000
SNIPPET_SIZE = 1_500
N_SNIPPETS = 30
MAX_SNIPPETS_PER_CHUNK = 12
SHORT_TEXT_THRESHOLD = 30_000
RETRIEVAL_TOP_K = 2
MIN_EVENTS_PER_CHUNK = 1
MIN_TAGGED_EVENTS = 2
MAX_OUTLINE_CONTEXT_CHARS = 5_000

EMBED_CONCURRENCY = int(os.environ.get("NTSMR_EMBED_CONCURRENCY", "8"))
CHUNK_CONCURRENCY = int(os.environ.get("NTSMR_CHUNK_CONCURRENCY", "4"))
SYNTHESIS_CONCURRENCY = int(os.environ.get("NTSMR_SYNTHESIS_CONCURRENCY", "8"))
GEMINI_TIMEOUT_SECONDS = int(os.environ.get("NTSMR_GEMINI_TIMEOUT_SECONDS", "180"))
GEMINI_RETRIES = int(os.environ.get("NTSMR_GEMINI_RETRIES", "2"))
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("NTSMR_OLLAMA_TIMEOUT_SECONDS", "10"))

EVENT_TYPES = {"action", "dialogue", "exposition", "bureaucracy"}
SUSPICIOUS_SOURCE_PATTERNS = (
    "@page {",
    "body {",
    "<html",
    "<body",
    "电子书",
    "公众号",
    "网站：",
    "cover",
)

STATIC_KEYWORDS = [
    "major plot progression action",
    "character emotional shift dialogue",
    "bureaucracy exposition history",
    "systemic threat or conflict",
]

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
ALLOWED_SIGNALS = {
    f"{family}:{tag}" for family, tags in FRAMEWORK_SIGNALS.items() for tag in tags
}
SIGNAL_VOCABULARY_TEXT = "\n".join(
    f"- {family}:{{{','.join(tags)}}}" for family, tags in FRAMEWORK_SIGNALS.items()
)

FRAMEWORK_SYNTHESIS_PROMPTS = {
    "Todorov's Equilibrium": {
        "filter_tag": "todorov",
        "prompt_suffix": """Identify the five stages of Todorov's narrative equilibrium model.\n\nThe analysis object must be:\n{\n  \"equilibrium\": \"Description of the starting status quo\",\n  \"disruption\": \"The inciting incident or protocol failure\",\n  \"recognition\": \"When the protagonist realizes the disruption\",\n  \"repair\": \"The attempt to fix or survive it\",\n  \"new_equilibrium\": \"The new, altered status quo\"\n}""",
    },
    "Actantial Model": {
        "filter_tag": "actantial",
        "prompt_suffix": """Identify the six actantial roles from Greimas's model.\n\nThe analysis object must be:\n{\n  \"subject\": \"The protagonist pursuing the central goal\",\n  \"object\": \"What the subject seeks or desires\",\n  \"sender\": \"The force/entity that initiates or compels the quest\",\n  \"receiver\": \"Who/what benefits from the quest's completion\",\n  \"helper\": \"Allies and enabling forces\",\n  \"opponent\": \"Antagonistic forces and obstacles\"\n}""",
    },
    "Quadrant Scores": {
        "filter_tag": None,
        "prompt_suffix": """Output exactly 6 floats between 0.0 and 1.0 for these metrics based on the timeline's pacing, conflict types, and plot structure:\n- time_linearity: 0.0=Non-linear/fragmented timeline, 1.0=Strictly chronological\n- pacing_velocity: 0.0=Slow/contemplative, 1.0=Fast/action-driven\n- threat_scale: 0.0=Personal/intimate stakes, 1.0=Cosmic/civilizational stakes\n- protagonist_fate: 0.0=Tragic/defeated, 1.0=Triumphant/vindicated\n- conflict_style: 0.0=Internal/psychological, 1.0=External/physical\n- price_type: 0.0=Abstract/spiritual cost, 1.0=Concrete/material cost\n\nCalibration anchors (verified by expert reviewers — use these as absolute reference points):\n- time_linearity: Story of Your Life ~0.10 (dual timelines blur causality), The Dispossessed ~0.20 (alternating past/present), Blindsight ~0.40 (frame + main), Neuromancer ~0.60 (mostly forward), Exhalation ~0.75 (linear with reflective frame), Dawn ~0.85 (straightforward chronology)\n- pacing_velocity: Never Let Me Go ~0.15 (quiet, reflective, no action), Solaris ~0.20 (slow philosophical), Story of Your Life ~0.25 (contemplative), The Dispossessed ~0.30 (deliberate), Annihilation ~0.30 (slow expedition), Dune ~0.50 (political + action balanced), Neuromancer ~0.70 (fast cyberpunk). CRITICAL: Most literary/philosophical SF scores 0.15-0.35. A contemplative novel with occasional action scenes is STILL slow-paced (0.20-0.35). Only score >0.50 for genuinely fast-paced, action-driven narratives.\n- threat_scale: Story of Your Life ~0.30 (personal loss), Never Let Me Go ~0.45 (institutional+personal), Neuromancer ~0.55 (mixed corporate/personal), Annihilation ~0.60 (ecological/institutional), Dune ~0.85 (civilizational), Dawn ~0.85 (species survival), Exhalation ~0.95 (cosmic entropy)\n- protagonist_fate: Never Let Me Go ~0.10 (total resignation), Dawn ~0.15 (forced compromise), Blindsight ~0.20 (dark ambiguity), Annihilation ~0.35 (transformation/loss), Story of Your Life ~0.40 (acceptance of loss), The Dispossessed ~0.50 (mixed success), Neuromancer ~0.65 (qualified victory), Dune ~0.68 (triumph with cost). CRITICAL: Ambiguous, bittersweet, or pyrrhic outcomes score 0.20-0.45. Only score >0.60 for clear victories.\n- conflict_style: Story of Your Life ~0.10 (pure internal revelation), Solaris ~0.15 (philosophical/internal), Never Let Me Go ~0.20 (quiet institutional), Annihilation ~0.30 (internal+environmental), The Dispossessed ~0.30 (social/ideological), Blindsight ~0.35 (philosophical with action), Dune ~0.40 (political with combat), Neuromancer ~0.60 (heist/combat). CRITICAL: Most SF is driven by ideas, not combat. Internal revelation, social tension, and philosophical confrontation all score 0.10-0.35. Only score >0.50 for primarily combat/action-driven conflict.\n- price_type: Exhalation ~0.15 (existential/knowledge), Solaris ~0.15 (existential/spiritual), Story of Your Life ~0.15 (emotional/temporal), Stars in My Pocket ~0.20 (identity/cultural), Annihilation ~0.25 (identity transformation), The Dispossessed ~0.25 (freedom/belonging), Blindsight ~0.30 (consciousness/autonomy), Never Let Me Go ~0.50 (bodily + identity), Neuromancer ~0.55 (mixed), Dune ~0.65 (political+physical). CRITICAL: Identity loss, epistemic cost, belief erosion, relational loss, and autonomy sacrifice are ALL abstract — score 0.10-0.35. Only score >0.50 when the dominant cost is bodily harm, material destruction, or concrete resource loss.\n\nScoring guidance:\n- Scores of exactly 0.0 or 1.0 are reserved for absolute extremes. Most works should score between 0.10 and 0.90.\n- Score the dominant narrative logic of the whole work, not just the loudest climax.\n- KNOWN BIAS: LLMs systematically over-score pacing_velocity, conflict_style, protagonist_fate, and price_type. Actively resist this tendency. If in doubt, score LOWER on these four axes.\n- For time_linearity, a mostly chronological narrative with minor flashbacks should score 0.7-0.85, not be pulled to the extremes.\n- For pacing_velocity, assess the overall tempo. A contemplative novel with a few action scenes is still slow-paced (0.20-0.35).\n- For threat_scale, institutions, class systems, biopolitical sorting, and civilizational stakes push upward. Personal stakes push downward.\n- For conflict_style, ask whether meaning is driven mainly by external combat/opposition (high) or by internal revelation, juxtaposition, and perspective shift (low). Most literary SF is idea-driven (0.15-0.40).\n- For price_type, bodily harm, material destruction, and concrete losses push upward. Identity, autonomy, belief, memory, and relational costs push downward (0.10-0.35).\n\nThe analysis object must be:\n{\n  \"time_linearity\": 0.0,\n  \"pacing_velocity\": 0.0,\n  \"threat_scale\": 0.0,\n  \"protagonist_fate\": 0.0,\n  \"conflict_style\": 0.0,\n  \"price_type\": 0.0\n}""",
    },
    "The Freytag Pyramid": {
        "filter_tag": "freytag",
        "prompt_suffix": """Map the narrative arc to Freytag's five dramatic stages. The five phases must be sequential: exposition → rising action → climax → falling action → denouement. Anchor each phase to specific events from the provided event timeline. Do not reference plot events that are not in the event list.\n\nThe analysis object must be:\n{\n  \"exposition\": \"Setup of world, characters, initial situation\",\n  \"rising_action\": \"Key complications and escalation toward the turning point\",\n  \"climax\": \"The decisive turning point or moment of highest tension\",\n  \"falling_action\": \"Consequences and unwinding after the climax\",\n  \"denouement\": \"Final resolution and the new state of affairs\"\n}""",
    },
    "The Three-Act Structure": {
        "filter_tag": "three_act",
        "prompt_suffix": """Map the narrative to the three-act structure, identifying the two key plot points. Each act boundary must correspond to a specific event from the provided event list. Do not import plot points from other works — only reference events that appear in the source material.\n\nThe analysis object must be:\n{\n  \"act_1_setup\": \"The world, characters, and status quo before the inciting incident\",\n  \"plot_point_1\": \"The event that launches the protagonist into the central conflict\",\n  \"act_2_confrontation\": \"The escalating obstacles, complications, and midpoint reversal\",\n  \"plot_point_2\": \"The crisis that forces the final confrontation\",\n  \"act_3_resolution\": \"The climax and its aftermath\"\n}""",
    },
    "The Monomyth": {
        "filter_tag": "monomyth",
        "prompt_suffix": """Analyze how this narrative relates to Campbell's Hero's Journey. Focus on how the work subverts or departs from the monomyth template.\n\nThe analysis object must be:\n{\n  \"applicable_stages\": [\"List the monomyth stages that appear in this narrative\"],\n  \"subversions\": \"How does this work depart from or subvert the traditional Hero's Journey?\"\n}""",
    },
    "Dan Harmon's Story Circle": {
        "filter_tag": "harmon",
        "prompt_suffix": """Analyze the narrative through Dan Harmon's 8-step Story Circle. Focus on 'The Take' -- the price paid for the journey.\n\nThe analysis object must be:\n{\n  \"circle_stages\": {\n    \"you\": \"Character in comfort zone\",\n    \"need\": \"What they want/need\",\n    \"go\": \"Entering unfamiliar territory\",\n    \"search\": \"Adapting to the new situation\",\n    \"find\": \"Getting what they wanted\",\n    \"take\": \"The price paid -- what was lost or sacrificed\",\n    \"return\": \"Going back to familiar territory\",\n    \"change\": \"How they have changed\"\n  }\n}""",
    },
    "Save the Cat! Beat Sheet": {
        "filter_tag": "save_the_cat",
        "prompt_suffix": """Map the narrative to Blake Snyder's Save the Cat beat sheet. Focus on where the pacing deviates from the prescribed beat timing.\n\nThe analysis object must be:\n{\n  \"beats_present\": [\"List the Save the Cat beats that appear\"],\n  \"pacing_deviations\": \"Where and how does the pacing diverge from the expected beat sheet timing?\"\n}""",
    },
    "Propp's Morphology": {
        "filter_tag": "propp",
        "prompt_suffix": """Identify which of Propp's narrative functions (narratemes) appear in this story.\n\nThe analysis object must be:\n{\n  \"applicable_narratemes\": [\"List each Proppian function present\"]\n}""",
    },
    "Kishotenketsu": {
        "filter_tag": "kishotenketsu",
        "prompt_suffix": """Analyze how well the four-act Kishotenketsu structure applies to this narrative.\n\nThe analysis object must be:\n{\n  \"ki\": \"Introduction -- the initial situation and characters\",\n  \"sho\": \"Development -- deepening without introducing conflict\",\n  \"ten\": \"Twist -- the surprising shift or new perspective\",\n  \"ketsu\": \"Conclusion -- reconciliation of the twist with the established narrative\",\n  \"applicability\": \"How well does Kishotenketsu fit this narrative vs. Western conflict-driven models?\"\n}""",
    },
    "Protocol Fiction Mapping": {
        "filter_tag": "protocol",
        "prompt_suffix": """Analyze this narrative through the Protocol Fiction lens (Summer of Protocols).\n\nThe analysis object must be:\n{\n  \"rule\": \"The protocol, rule, or system the narrative renders visible\",\n  \"failure_mode\": \"How the protocol fails or is subverted\",\n  \"human_insight\": \"The human truth revealed through the protocol's failure\"\n}""",
    },
    "Genette's Narrative Discourse": {
        "filter_tag": "genette_narrative",
        "prompt_suffix": """Analyze the narrative discourse using Genette's categories.\n\nThe analysis object must be:\n{\n  \"order\": \"How is story time arranged vs. discourse time?\",\n  \"duration\": \"Pacing techniques: scene, summary, ellipsis, pause, stretch\",\n  \"focalization\": \"Who perceives? Zero, internal, or external focalization\"\n}""",
    },
    "Levi-Strauss's Binary Oppositions": {
        "filter_tag": "levi_strauss",
        "prompt_suffix": """Identify the central binary oppositions that structure this narrative's meaning.\n\nThe analysis object must be:\n{\n  \"primary_binary\": \"The dominant opposition\",\n  \"secondary_binary\": \"A supporting opposition\",\n  \"mediator\": \"The character, event, or concept that mediates the opposition\"\n}""",
    },
    "Cognitive Estrangement": {
        "filter_tag": "estrangement",
        "prompt_suffix": """Analyze through cognitive estrangement.\n\nThe analysis object must be:\n{\n  \"familiar_concept\": \"The real-world concept or system being estranged\",\n  \"estranging_mechanism\": \"The speculative element that defamiliarizes it\",\n  \"cognitive_shift\": \"The new understanding produced\"\n}""",
    },
    "Bakhtin's Chronotope": {
        "filter_tag": "bakhtin",
        "prompt_suffix": """Analyze the chronotope -- the fusion of time and space that defines this narrative's world.\n\nThe analysis object must be:\n{\n  \"spatial_matrix\": \"The defining spatial logic\",\n  \"temporal_flow\": \"How time operates\",\n  \"intersection\": \"Where space and time fuse to create the world-feeling\"\n}""",
    },
    "Aristotelian Poetics": {
        "filter_tag": "aristotle",
        "prompt_suffix": """Analyze through Aristotle's Poetics.\n\nThe analysis object must be:\n{\n  \"hamartia\": \"The protagonist's tragic flaw or error of judgment\",\n  \"peripeteia\": \"The reversal of fortune\",\n  \"anagnorisis\": \"The critical recognition or discovery\"\n}""",
    },
    "Jungian Archetypal Analysis": {
        "filter_tag": "jung",
        "prompt_suffix": """Identify the Jungian archetypes present in the narrative.

Valid archetypes: Persona, Shadow, Anima/Animus, Self, Trickster, Wise Old Man/Woman, Child, Mother, Father. Do not assign archetypes outside this list. Each archetype assignment must cite a specific scene or event from the provided material.\n\nThe analysis object must be:\n{\n  \"persona\": \"The public mask or social role characters present\",\n  \"shadow\": \"The repressed, dark, or denied aspects\",\n  \"anima_animus\": \"The contrasexual inner figure\",\n  \"trickster\": \"The agent of chaos, boundary-crossing, or transformation\"\n}""",
    },
    "Genette's Transtextuality": {
        "filter_tag": "transtextuality",
        "prompt_suffix": """Analyze the transtextual relationships -- how this text relates to other texts. Only claim intertextual connections that are explicitly mentioned, quoted, or clearly alluded to in the provided text snippets. Do not infer connections based on genre conventions alone. If no clear intertextual references exist in the source material, state this explicitly rather than fabricating connections.\n\nThe analysis object must be:\n{\n  \"intertextuality\": \"Direct quotations, allusions, or references to other works\",\n  \"paratextuality\": \"How titles, epigraphs, prefaces, or cover art frame meaning\",\n  \"metatextuality\": \"How the text comments on or critiques other texts or its own genre\"\n}""",
    },
}

FRAMEWORK_ANALYSIS_SCHEMAS: dict[str, Any] = {
    "Todorov's Equilibrium": {
        "equilibrium": "str",
        "disruption": "str",
        "recognition": "str",
        "repair": "str",
        "new_equilibrium": "str",
    },
    "Actantial Model": {
        "subject": "str",
        "object": "str",
        "sender": "str",
        "receiver": "str",
        "helper": "str",
        "opponent": "str",
    },
    "Quadrant Scores": {
        "time_linearity": "float01",
        "pacing_velocity": "float01",
        "threat_scale": "float01",
        "protagonist_fate": "float01",
        "conflict_style": "float01",
        "price_type": "float01",
    },
    "The Freytag Pyramid": {
        "exposition": "str",
        "rising_action": "str",
        "climax": "str",
        "falling_action": "str",
        "denouement": "str",
    },
    "The Three-Act Structure": {
        "act_1_setup": "str",
        "plot_point_1": "str",
        "act_2_confrontation": "str",
        "plot_point_2": "str",
        "act_3_resolution": "str",
    },
    "The Monomyth": {"applicable_stages": ["str"], "subversions": "str"},
    "Dan Harmon's Story Circle": {
        "circle_stages": {
            "you": "str",
            "need": "str",
            "go": "str",
            "search": "str",
            "find": "str",
            "take": "str",
            "return": "str",
            "change": "str",
        },
    },
    "Save the Cat! Beat Sheet": {"beats_present": ["str"], "pacing_deviations": "str"},
    "Propp's Morphology": {"applicable_narratemes": ["str"]},
    "Kishotenketsu": {
        "ki": "str",
        "sho": "str",
        "ten": "str",
        "ketsu": "str",
        "applicability": "str",
    },
    "Protocol Fiction Mapping": {
        "rule": "str",
        "failure_mode": "str",
        "human_insight": "str",
    },
    "Genette's Narrative Discourse": {
        "order": "str",
        "duration": "str",
        "focalization": "str",
    },
    "Levi-Strauss's Binary Oppositions": {
        "primary_binary": "str",
        "secondary_binary": "str",
        "mediator": "str",
    },
    "Cognitive Estrangement": {
        "familiar_concept": "str",
        "estranging_mechanism": "str",
        "cognitive_shift": "str",
    },
    "Bakhtin's Chronotope": {
        "spatial_matrix": "str",
        "temporal_flow": "str",
        "intersection": "str",
    },
    "Aristotelian Poetics": {
        "hamartia": "str",
        "peripeteia": "str",
        "anagnorisis": "str",
    },
    "Jungian Archetypal Analysis": {
        "persona": "str",
        "shadow": "str",
        "anima_animus": "str",
        "trickster": "str",
    },
    "Genette's Transtextuality": {
        "intertextuality": "str",
        "paratextuality": "str",
        "metatextuality": "str",
    },
}

OUTLINE_PROMPT = """You are a master literary structuralist performing a fast-pass structural survey.
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

Be concrete. Use actual names. Do not speculate about content between snippets."""

KEYWORD_EXTRACTION_PROMPT = """Extract retrieval keywords from this novel outline for semantic search.

{outline}

Output a JSON array of 8-12 search queries, each 3-8 words. Include:
- 2-3 queries with major character names + their key actions/relationships
- 2-3 queries about central conflicts and plot turning points
- 2-3 queries about thematic elements and narrative tension
- 1-2 queries about the climax/resolution

Return ONLY a JSON array of strings, no other text:
[\"query 1\", \"query 2\", ...]"""

SUBSTRATE_EXTRACTION_PROMPT = """You are a narrative substrate extraction sub-agent analyzing {chunk_id}.

Here is the GLOBAL OUTLINE CONTEXT of the novel:
{outline_context}

Here are retrieved snippets from your specific chunk. Each snippet has a stable snippet ID:
{snippets_text}

Task:
Extract reusable narrative substrate records grounded strictly in these snippets and output a single JSON object with exactly these keys:
{{
  "characters": [
    {{
      "snippet_ids": ["{chunk_id}-s1"],
      "name": "Character name",
      "aliases": ["Alias 1"],
      "role": "Narrative or social role",
      "summary": "Brief grounded description"
    }}
  ],
  "dialogue_turns": [
    {{
      "snippet_ids": ["{chunk_id}-s2"],
      "speaker": "Speaker name or group",
      "addressee": "Addressee name, group, or unknown",
      "speech_act": "command|confession|debate|exposition|revelation|request|threat|warning|other",
      "summary": "Brief grounded summary of what is said or performed through speech"
    }}
  ],
  "settings": [
    {{
      "snippet_ids": ["{chunk_id}-s3"],
      "location": "Specific place or locale",
      "time_marker": "When this setting is situated in the narrative",
      "social_context": "Institutional, social, or environmental context",
      "summary": "Brief grounded description of the setting"
    }}
  ]
}}

Rules:
- Use only the provided snippet_ids.
- Use empty arrays when nothing is grounded strongly enough.
- Do not invent names, locations, or speech turns absent from the snippets.
- aliases may be an empty array.
- Output JSON only. No markdown. No prose.
"""

EMBED_SEMAPHORE = asyncio.Semaphore(EMBED_CONCURRENCY)
CHUNK_SEMAPHORE = asyncio.Semaphore(CHUNK_CONCURRENCY)
SYNTHESIS_SEMAPHORE = asyncio.Semaphore(SYNTHESIS_CONCURRENCY)


@dataclass(frozen=True)
class ArtifactPaths:
    events_path: str
    synthesis_path: str
    report_path: str
    outline_path: str
    snippets_path: str
    characters_path: str
    dialogue_path: str
    settings_path: str


@dataclass(frozen=True)
class LlmConfig:
    backend: str
    model: str
    reasoning_effort: str | None
    run_label: str


def ordered_dedupe(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def to_json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
        return [to_json_safe(inner) for inner in value]
    return value


def build_run_label(
    pipeline_version: str,
    model: str,
    reasoning_effort: str | None = None,
) -> str:
    parts = [f"NTSMR-{pipeline_version}", model.strip()]
    if reasoning_effort:
        parts.append(reasoning_effort.strip().lower())
    return "-".join(parts)


def build_llm_config(
    backend: str | None,
    model: str | None,
    reasoning_effort: str | None,
    run_label: str | None,
) -> LlmConfig:
    normalized_backend = (backend or DEFAULT_LLM_BACKEND).strip().lower()
    if normalized_backend not in SUPPORTED_LLM_BACKENDS:
        raise ValueError(f"Unsupported LLM backend: {normalized_backend}")

    normalized_reasoning = reasoning_effort.strip().lower() if reasoning_effort else None
    if normalized_backend == "gemini":
        normalized_model = (model or DEFAULT_GEMINI_MODEL_LABEL).strip()
        if normalized_reasoning:
            raise ValueError("Gemini backend does not support reasoning effort overrides")
    elif normalized_backend == "claude":
        raw_model = (model or DEFAULT_CLAUDE_MODEL).strip()
        normalized_model = CLAUDE_MODEL_ALIASES.get(raw_model, raw_model)
        normalized_reasoning = normalized_reasoning or DEFAULT_CLAUDE_REASONING_EFFORT
        if not normalized_reasoning:
            raise ValueError("Claude backend requires a reasoning effort")
        if normalized_reasoning not in CLAUDE_REASONING_EFFORTS:
            raise ValueError(f"Unsupported Claude reasoning effort: {normalized_reasoning} (must be low/medium/high)")
    else:
        normalized_model = (model or DEFAULT_CODEX_MODEL).strip()
        normalized_reasoning = normalized_reasoning or DEFAULT_CODEX_REASONING_EFFORT
        if not normalized_reasoning:
            raise ValueError("Codex Exec backend requires a reasoning effort")
        if normalized_reasoning not in SUPPORTED_REASONING_EFFORTS:
            raise ValueError(f"Unsupported reasoning effort: {normalized_reasoning}")

    normalized_run_label = (run_label or "").strip() or build_run_label(
        NTSMR_VERSION,
        normalized_model,
        normalized_reasoning,
    )
    return LlmConfig(
        backend=normalized_backend,
        model=normalized_model,
        reasoning_effort=normalized_reasoning,
        run_label=normalized_run_label,
    )


ACTIVE_LLM_CONFIG: LlmConfig | None = None


def get_active_llm_config() -> LlmConfig:
    global ACTIVE_LLM_CONFIG
    if ACTIVE_LLM_CONFIG is None:
        ACTIVE_LLM_CONFIG = build_llm_config(None, None, None, None)
    return ACTIVE_LLM_CONFIG


def combine_keywords(dynamic_keywords: list[str], static_keywords: list[str] | None = None) -> list[str]:
    static_keywords = static_keywords or STATIC_KEYWORDS
    cleaned = [kw.strip() for kw in static_keywords + list(dynamic_keywords) if kw.strip()]
    return ordered_dedupe(cleaned)


def resolve_artifact_paths(output_file: str) -> ArtifactPaths:
    if output_file.endswith(".jsonl"):
        base = output_file[: -len(".jsonl")]
        events_path = output_file
    else:
        base = output_file
        events_path = output_file + ".events.jsonl"
    return ArtifactPaths(
        events_path=events_path,
        synthesis_path=base + ".synthesis.json",
        report_path=base + ".report.json",
        outline_path=base + ".outline.json",
        snippets_path=base + ".snippets.jsonl",
        characters_path=base + ".characters.jsonl",
        dialogue_path=base + ".dialogue.jsonl",
        settings_path=base + ".settings.jsonl",
    )


def assess_source_text_health(text: str) -> dict[str, Any]:
    stripped = text.strip()
    lowered = stripped.lower()
    alpha_ratio = (
        sum(1 for char in stripped if char.isalpha()) / max(len(stripped), 1)
        if stripped
        else 0.0
    )
    suspicious_hits = [pattern for pattern in SUSPICIOUS_SOURCE_PATTERNS if pattern in lowered]
    looks_broken = len(stripped) < MIN_SOURCE_TEXT_CHARS
    if not looks_broken and len(stripped) < 5_000 and alpha_ratio < 0.45 and suspicious_hits:
        looks_broken = True
    return {
        "char_count": len(stripped),
        "alpha_ratio": round(alpha_ratio, 3),
        "suspicious_hits": suspicious_hits,
        "looks_broken": looks_broken,
    }


def validate_source_text(text: str, book_file: str) -> dict[str, Any]:
    health = assess_source_text_health(text)
    if health["looks_broken"]:
        hint = ", ".join(health["suspicious_hits"]) or "too short"
        raise RuntimeError(
            f"Source text for {book_file} looks broken: {health['char_count']} chars, "
            f"alpha_ratio={health['alpha_ratio']}, hits={hint}"
        )
    return health


def select_snippets_stratified(text: str, n_snippets: int = N_SNIPPETS, snippet_size: int = SNIPPET_SIZE):
    text_len = len(text)
    if text_len <= snippet_size * 2:
        return [(0.0, text)]

    snippets = [(0.0, text[:snippet_size]), (100.0, text[-snippet_size:])]
    n_middle = n_snippets - 2
    step = text_len // (n_middle + 1)
    for i in range(1, n_middle + 1):
        start = min(i * step, text_len - snippet_size)
        pct = (start / text_len) * 100
        snippets.append((pct, text[start : start + snippet_size]))
    snippets.sort(key=lambda item: item[0])
    return snippets


def build_outline_samples(text: str, n_snippets: int = N_SNIPPETS, snippet_size: int = SNIPPET_SIZE) -> list[dict[str, Any]]:
    return [
        {"position_percent": round(position, 2), "text": snippet}
        for position, snippet in select_snippets_stratified(text, n_snippets=n_snippets, snippet_size=snippet_size)
    ]


def build_outline_context(global_outline: str, max_chars: int = MAX_OUTLINE_CONTEXT_CHARS) -> str:
    sections = []
    for heading in ["## DRAMATIS PERSONAE", "## NARRATIVE STRUCTURE", "## PLOT OUTLINE"]:
        match = re.search(rf"({re.escape(heading)}.*?)(?=\n## |\Z)", global_outline, re.DOTALL)
        if match:
            sections.append(match.group(1).strip())
    context = "\n\n".join(sections).strip() or global_outline.strip()
    return context[:max_chars]


SYNTHESIS_SOURCE_SAMPLE_SIZE = 5_000


def build_source_text_sample(text: str, sample_size: int = SYNTHESIS_SOURCE_SAMPLE_SIZE) -> str:
    """Build a source text sample for synthesis grounding: opening + closing."""
    if len(text) <= sample_size:
        return text
    half = sample_size // 2
    return (
        f"[OPENING — first {half} chars]\n{text[:half]}\n\n"
        f"[CLOSING — last {half} chars]\n{text[-half:]}"
    )


def extract_character_names(outline: str) -> list[str]:
    """Extract character names from the DRAMATIS PERSONAE section of the outline."""
    match = re.search(r"## DRAMATIS PERSONAE(.*?)(?=\n## |\Z)", outline, re.DOTALL)
    if not match:
        return []
    section = match.group(1)
    names = []
    for line in section.strip().split("\n"):
        line = line.strip().lstrip("- *•")
        if not line:
            continue
        # Extract name before first parenthetical, comma, colon, or dash
        name_match = re.match(r"^\*{0,2}([A-Z][^(,:\-—\n*]+)", line)
        if name_match:
            name = name_match.group(1).strip().rstrip("*")
            if name and len(name) > 1:
                names.append(name)
    return names


def build_micro_chunk_records(text: str, start_char: int = 0) -> list[dict[str, Any]]:
    records = []
    for source_index, chunk_start in enumerate(range(0, len(text), MICRO_CHUNK_SIZE)):
        chunk_text = text[chunk_start : chunk_start + MICRO_CHUNK_SIZE]
        records.append(
            {
                "source_index": source_index,
                "start_char": start_char + chunk_start,
                "end_char": start_char + chunk_start + len(chunk_text),
                "text": chunk_text,
            }
        )
    return records


def normalize_signal(signal: str) -> str:
    return signal.strip().lower()


def validate_signal(signal: str) -> str | None:
    normalized = normalize_signal(signal)
    if normalized not in ALLOWED_SIGNALS:
        print(f"    [WARN] Skipping unknown framework signal: {signal}")
        return None
    return normalized


def require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def validate_snippet_ids(snippet_ids: Any, allowed_snippet_ids: set[str]) -> list[str]:
    if not isinstance(snippet_ids, list) or not snippet_ids:
        raise ValueError("Record missing snippet_ids")
    normalized = ordered_dedupe(
        [snippet_id.strip() for snippet_id in snippet_ids if isinstance(snippet_id, str) and snippet_id.strip()]
    )
    if not normalized:
        raise ValueError("Record missing snippet_ids")
    if not set(normalized).issubset(allowed_snippet_ids):
        raise ValueError("Record references unknown snippet_ids")
    return normalized


def validate_optional_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("aliases must be a list")
    normalized = []
    for alias in value:
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("aliases must contain only non-empty strings")
        normalized.append(alias.strip())
    return ordered_dedupe(normalized)


def validate_event(event: dict[str, Any], allowed_snippet_ids: set[str]) -> dict[str, Any]:
    event_id = event.get("event_id")
    chunk_id = event.get("chunk_id")
    snippet_ids = event.get("snippet_ids")
    event_type = event.get("type")
    summary = event.get("summary")
    framework_signals = event.get("framework_signals", [])

    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("Event missing event_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("Event missing chunk_id")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type}")
    snippet_ids = validate_snippet_ids(snippet_ids, allowed_snippet_ids)
    summary = require_non_empty_string(summary, "summary")
    if not isinstance(framework_signals, list):
        raise ValueError("framework_signals must be a list")

    raw_signals = [validate_signal(signal) for signal in framework_signals]
    n_invalid = sum(1 for s in raw_signals if s is None)
    validated_signals = ordered_dedupe([s for s in raw_signals if s is not None])
    if n_invalid > 0:
        print(f"    [WARN] Event {event_id.strip()}: {n_invalid}/{len(framework_signals)} invalid signals removed")
    return {
        "event_id": event_id.strip(),
        "chunk_id": chunk_id.strip(),
        "snippet_ids": snippet_ids,
        "type": event_type,
        "framework_signals": validated_signals,
        "summary": summary,
    }


def validate_character_record(
    chunk_id: str,
    raw_record: Any,
    allowed_snippet_ids: set[str],
    ordinal: int,
) -> dict[str, Any]:
    if not isinstance(raw_record, dict):
        raise ValueError("Character record must be an object")
    return {
        "character_id": f"{chunk_id}-char-{ordinal}",
        "chunk_id": chunk_id,
        "snippet_ids": validate_snippet_ids(raw_record.get("snippet_ids"), allowed_snippet_ids),
        "name": require_non_empty_string(raw_record.get("name"), "character.name"),
        "aliases": validate_optional_aliases(raw_record.get("aliases")),
        "role": require_non_empty_string(raw_record.get("role"), "character.role"),
        "summary": require_non_empty_string(raw_record.get("summary"), "character.summary"),
    }


def validate_dialogue_record(
    chunk_id: str,
    raw_record: Any,
    allowed_snippet_ids: set[str],
    ordinal: int,
) -> dict[str, Any]:
    if not isinstance(raw_record, dict):
        raise ValueError("Dialogue record must be an object")
    return {
        "dialogue_id": f"{chunk_id}-dlg-{ordinal}",
        "chunk_id": chunk_id,
        "snippet_ids": validate_snippet_ids(raw_record.get("snippet_ids"), allowed_snippet_ids),
        "speaker": require_non_empty_string(raw_record.get("speaker"), "dialogue.speaker"),
        "addressee": require_non_empty_string(raw_record.get("addressee"), "dialogue.addressee"),
        "speech_act": require_non_empty_string(raw_record.get("speech_act"), "dialogue.speech_act"),
        "summary": require_non_empty_string(raw_record.get("summary"), "dialogue.summary"),
    }


def validate_setting_record(
    chunk_id: str,
    raw_record: Any,
    allowed_snippet_ids: set[str],
    ordinal: int,
) -> dict[str, Any]:
    if not isinstance(raw_record, dict):
        raise ValueError("Setting record must be an object")
    return {
        "setting_id": f"{chunk_id}-set-{ordinal}",
        "chunk_id": chunk_id,
        "snippet_ids": validate_snippet_ids(raw_record.get("snippet_ids"), allowed_snippet_ids),
        "location": require_non_empty_string(raw_record.get("location"), "setting.location"),
        "time_marker": require_non_empty_string(raw_record.get("time_marker"), "setting.time_marker"),
        "social_context": require_non_empty_string(raw_record.get("social_context"), "setting.social_context"),
        "summary": require_non_empty_string(raw_record.get("summary"), "setting.summary"),
    }


def validate_substrate_payload(
    chunk_id: str,
    payload: Any,
    allowed_snippet_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ValueError("Substrate payload must be an object")

    characters_raw = payload.get("characters", [])
    dialogue_raw = payload.get("dialogue_turns", [])
    settings_raw = payload.get("settings", [])
    if not isinstance(characters_raw, list):
        raise ValueError("characters must be a list")
    if not isinstance(dialogue_raw, list):
        raise ValueError("dialogue_turns must be a list")
    if not isinstance(settings_raw, list):
        raise ValueError("settings must be a list")

    return {
        "characters": [
            validate_character_record(chunk_id, record, allowed_snippet_ids, ordinal)
            for ordinal, record in enumerate(characters_raw, start=1)
        ],
        "dialogue_turns": [
            validate_dialogue_record(chunk_id, record, allowed_snippet_ids, ordinal)
            for ordinal, record in enumerate(dialogue_raw, start=1)
        ],
        "settings": [
            validate_setting_record(chunk_id, record, allowed_snippet_ids, ordinal)
            for ordinal, record in enumerate(settings_raw, start=1)
        ],
    }


def validate_structure(value: Any, schema: Any, path: str = "analysis") -> None:
    if schema == "str":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} must be a non-empty string")
        return
    if schema == "float01":
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{path} must be a float between 0.0 and 1.0")
        return
    if isinstance(schema, list):
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list")
        for index, item in enumerate(value):
            validate_structure(item, schema[0], f"{path}[{index}]")
        return
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        missing = [key for key in schema if key not in value]
        if missing:
            raise ValueError(f"{path} missing keys: {', '.join(missing)}")
        for key, child_schema in schema.items():
            validate_structure(value[key], child_schema, f"{path}.{key}")
        return
    raise ValueError(f"Unsupported schema for {path}")


def extract_first_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON payload found")


def parse_extraction_output(output: str, chunk_id: str, allowed_snippet_ids: set[str]) -> list[dict[str, Any]]:
    lines = output.splitlines()
    start_index = 0
    for index, line in enumerate(lines):
        if line.strip() == "## JSONL":
            start_index = index + 1
            break

    raw_objects = []
    for raw_line in lines[start_index:]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        if not stripped.startswith("{"):
            continue
        raw_objects.append(json.loads(stripped))

    if not raw_objects:
        raise ValueError(f"No JSONL events parsed for {chunk_id}")

    events = []
    for index, raw_event in enumerate(raw_objects, start=1):
        if not isinstance(raw_event, dict):
            raise ValueError("Extraction output contained a non-object line")
        normalized = dict(raw_event)
        normalized["event_id"] = f"{chunk_id}-ev-{index}"
        normalized["chunk_id"] = chunk_id
        try:
            events.append(validate_event(normalized, allowed_snippet_ids))
        except ValueError as exc:
            print(f"    [WARN] Skipping malformed event {chunk_id}-ev-{index}: {exc}")

    if len(events) < MIN_EVENTS_PER_CHUNK:
        raise ValueError(f"Expected at least {MIN_EVENTS_PER_CHUNK} events for {chunk_id}, got {len(events)}")
    return events


def parse_substrate_output(output: str, chunk_id: str, allowed_snippet_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    payload = extract_first_json_value(output)
    return validate_substrate_payload(chunk_id, payload, allowed_snippet_ids)


def validate_framework_result(
    framework_name: str,
    payload: Any,
    allowed_event_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{framework_name} result must be an object")
    if "analysis" not in payload or "evidence_event_ids" not in payload:
        raise ValueError(f"{framework_name} result must contain analysis and evidence_event_ids")

    analysis = payload["analysis"]
    evidence_event_ids = payload["evidence_event_ids"]
    if not isinstance(evidence_event_ids, list) or not evidence_event_ids:
        raise ValueError(f"{framework_name} must cite at least one evidence event id")
    for event_id in evidence_event_ids:
        if not isinstance(event_id, str) or event_id not in allowed_event_ids:
            raise ValueError(f"{framework_name} cited unknown event id: {event_id}")

    validate_structure(analysis, FRAMEWORK_ANALYSIS_SCHEMAS[framework_name])
    confidence = payload.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence = max(0.0, min(1.0, float(confidence)))
    else:
        confidence = None
    result = {
        "analysis": analysis,
        "evidence_event_ids": ordered_dedupe(evidence_event_ids),
    }
    if confidence is not None:
        result["confidence"] = confidence
    return result


FRAMEWORK_FALLBACK_KEYWORDS: dict[str, list[str]] = {
    "todorov": ["equilibrium", "disruption", "recognition", "repair", "status quo", "inciting"],
    "freytag": ["exposition", "rising", "climax", "falling", "resolution", "tension"],
    "actantial": ["protagonist", "antagonist", "goal", "helper", "opponent", "quest"],
    "three_act": ["setup", "confrontation", "resolution", "act", "plot point"],
    "monomyth": ["hero", "journey", "call", "threshold", "ordeal", "return"],
    "harmon": ["comfort", "need", "unfamiliar", "adapt", "find", "price", "change"],
    "save_the_cat": ["opening", "catalyst", "debate", "midpoint", "dark night", "finale"],
    "propp": ["villain", "donor", "hero", "departure", "struggle", "victory", "wedding"],
    "kishotenketsu": ["introduction", "development", "twist", "conclusion", "juxtaposition"],
    "protocol": ["rule", "protocol", "system", "failure", "insight", "institution"],
    "genette_narrative": ["time", "duration", "frequency", "narrator", "focalization", "perspective"],
    "levi_strauss": ["opposition", "binary", "mediator", "nature", "culture"],
    "estrangement": ["familiar", "defamiliarize", "cognitive", "speculative", "shift"],
    "bakhtin": ["space", "time", "chronotope", "threshold", "intersection"],
    "aristotle": ["flaw", "reversal", "recognition", "catharsis", "tragedy"],
    "jung": ["persona", "shadow", "anima", "trickster", "archetype", "unconscious"],
    "transtextuality": ["allusion", "reference", "intertextual", "metatextual", "genre"],
}
MAX_FALLBACK_EVENTS = 30


def filter_events_for_framework(
    events: list[dict[str, Any]],
    filter_tag: str | None,
    min_events: int = MIN_TAGGED_EVENTS,
) -> tuple[list[dict[str, Any]], bool]:
    if not filter_tag:
        return events, False
    tagged = [
        event
        for event in events
        if any(signal.startswith(f"{filter_tag}:") for signal in event.get("framework_signals", []))
    ]
    if len(tagged) >= min_events:
        return tagged, False

    # Improved fallback: score events by keyword relevance instead of using all events
    keywords = FRAMEWORK_FALLBACK_KEYWORDS.get(filter_tag, [])
    if not keywords:
        return events, True

    def relevance_score(event: dict[str, Any]) -> float:
        summary = event.get("summary", "").lower()
        return sum(1 for kw in keywords if kw in summary)

    scored = [(relevance_score(e), i, e) for i, e in enumerate(events)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    # Take tagged events + top relevant events up to MAX_FALLBACK_EVENTS
    selected_ids = {id(e) for e in tagged}
    result = list(tagged)
    for score, _, event in scored:
        if id(event) not in selected_ids and len(result) < MAX_FALLBACK_EVENTS:
            result.append(event)
            selected_ids.add(id(event))
    return result, True


async def run_gemini_cli(prompt: str, timeout: int = GEMINI_TIMEOUT_SECONDS, retries: int = GEMINI_RETRIES) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        process = await asyncio.create_subprocess_exec(
            "gemini",
            "-y",
            "-p",
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            last_error = TimeoutError(f"gemini timed out after {timeout}s")
        else:
            if process.returncode == 0 and stdout:
                return stdout.decode("utf-8", errors="replace")
            last_error = RuntimeError(
                stderr.decode("utf-8", errors="replace") or f"gemini exited with code {process.returncode}"
            )
        if attempt <= retries:
            await asyncio.sleep(attempt)
    raise RuntimeError(f"Gemini CLI failed after retries: {last_error}")


def build_codex_exec_command(
    prompt: str,
    config: LlmConfig,
    output_path: str,
    workdir: str,
) -> list[str]:
    del prompt  # The prompt is passed via stdin to avoid shell length limits.
    command = [
        "codex",
        "exec",
        "-C",
        workdir,
        "--skip-git-repo-check",
        "--color",
        "never",
        "--disable",
        "apps",
        "--ephemeral",
        "-s",
        "read-only",
        "-o",
        output_path,
        "-m",
        config.model,
    ]
    if config.reasoning_effort:
        command.extend(["-c", f'model_reasoning_effort="{config.reasoning_effort}"'])
    command.append("-")
    return command


def codex_exec_prompt(prompt: str) -> str:
    return (
        "You are being used as a structured text generation backend inside the NTSMR "
        "narrative analysis pipeline. Do not use tools, do not inspect the filesystem, "
        "do not ask follow-up questions, and do not add commentary outside the requested "
        "format. Return only the requested output.\n\n"
        + prompt
    )


async def run_codex_exec_cli(
    prompt: str,
    config: LlmConfig,
    timeout: int = GEMINI_TIMEOUT_SECONDS,
    retries: int = GEMINI_RETRIES,
) -> str:
    last_error: Exception | None = None
    stdin_prompt = codex_exec_prompt(prompt)
    for attempt in range(1, retries + 2):
        with tempfile.TemporaryDirectory(prefix="ntsmr-codex-exec-") as tempdir:
            output_path = os.path.join(tempdir, "codex-last-message.txt")
            process = await asyncio.create_subprocess_exec(
                *build_codex_exec_command(
                    prompt=stdin_prompt,
                    config=config,
                    output_path=output_path,
                    workdir=tempdir,
                ),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(stdin_prompt.encode("utf-8")),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                last_error = TimeoutError(f"codex exec timed out after {timeout}s")
            else:
                if process.returncode == 0:
                    if os.path.exists(output_path):
                        output_text = Path(output_path).read_text(encoding="utf-8").strip()
                        if output_text:
                            return output_text
                    stdout_text = stdout.decode("utf-8", errors="replace").strip()
                    if stdout_text:
                        return stdout_text
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                stdout_text = stdout.decode("utf-8", errors="replace").strip()
                last_error = RuntimeError(
                    stderr_text or stdout_text or f"codex exec exited with code {process.returncode}"
                )
        if attempt <= retries:
            await asyncio.sleep(attempt)
    raise RuntimeError(f"Codex Exec failed after retries: {last_error}")


def claude_cli_prompt(prompt: str) -> str:
    return (
        "You are being used as a structured text generation backend inside the NTSMR "
        "narrative analysis pipeline. Do not use tools, do not inspect the filesystem, "
        "do not ask follow-up questions, and do not add commentary outside the requested "
        "format. Return only the requested output.\n\n"
        + prompt
    )


async def run_claude_cli(
    prompt: str,
    config: LlmConfig,
    timeout: int = GEMINI_TIMEOUT_SECONDS,
    retries: int = GEMINI_RETRIES,
) -> str:
    last_error: Exception | None = None
    stdin_prompt = claude_cli_prompt(prompt)
    env = {**os.environ, "CLAUDECODE": ""}
    for attempt in range(1, retries + 2):
        command = [
            "claude",
            "-p",
            "--model", config.model,
            "--no-session-persistence",
            "--dangerously-skip-permissions",
            "--tools", "",
            "--output-format", "json",
        ]
        if config.reasoning_effort:
            command.extend(["--effort", config.reasoning_effort])
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin_prompt.encode("utf-8")),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            last_error = TimeoutError(f"claude timed out after {timeout}s")
        else:
            if process.returncode == 0 and stdout:
                raw = stdout.decode("utf-8", errors="replace")
                try:
                    envelope = json.loads(raw)
                    result_text = envelope.get("result", raw)
                    usage = envelope.get("usage", {})
                    TOKEN_USAGE.add(
                        input_tok=usage.get("input_tokens", 0),
                        output_tok=usage.get("output_tokens", 0),
                        cache_creation=usage.get("cache_creation_input_tokens", 0),
                        cache_read=usage.get("cache_read_input_tokens", 0),
                        cost_usd=float(envelope.get("total_cost_usd", 0)),
                    )
                    return result_text
                except (json.JSONDecodeError, KeyError):
                    return raw
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            last_error = RuntimeError(
                stderr_text or stdout_text or f"claude exited with code {process.returncode}"
            )
        if attempt <= retries:
            await asyncio.sleep(attempt)
    raise RuntimeError(f"Claude CLI failed after retries: {last_error}")


async def run_llm_cli(prompt: str, timeout: int = GEMINI_TIMEOUT_SECONDS, retries: int = GEMINI_RETRIES) -> str:
    config = get_active_llm_config()
    if config.backend == "gemini":
        return await run_gemini_cli(prompt, timeout=timeout, retries=retries)
    if config.backend == "codex-exec":
        return await run_codex_exec_cli(prompt, config=config, timeout=timeout, retries=retries)
    if config.backend == "claude":
        return await run_claude_cli(prompt, config=config, timeout=timeout, retries=retries)
    raise RuntimeError(f"Unhandled LLM backend: {config.backend}")


def embedding_hosts() -> list[str]:
    primary = os.environ.get("OLLAMA_HOST")
    hosts = []
    if primary:
        hosts.append(primary)
    hosts.append("http://100.107.177.128:11434")
    hosts.append("http://localhost:11434")
    return ordered_dedupe(hosts)


def get_nomic_embedding_sync(text: str) -> list[float]:
    payload = {"model": "nomic-embed-text", "prompt": text}
    last_error = None
    for host in embedding_hosts():
        req = urllib.request.Request(
            f"{host}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS)
            embedding = json.loads(response.read()).get("embedding")
            if embedding:
                return embedding
        except Exception as exc:  # pragma: no cover - network errors are runtime only
            last_error = exc
    raise RuntimeError(f"Failed to fetch embedding: {last_error}")


async def get_nomic_embedding(text: str) -> list[float]:
    async with EMBED_SEMAPHORE:
        return await asyncio.to_thread(get_nomic_embedding_sync, text)


async def embed_many(texts: list[str]) -> list[list[float]]:
    return await asyncio.gather(*[get_nomic_embedding(text) for text in texts])


async def generate_global_outline(outline_samples: list[dict[str, Any]]) -> str:
    t0 = time.time()
    print("\n--- Phase 1: Generating Global Outline (stratified, 30 snippets) ---")
    coverage = sum(len(sample["text"]) for sample in outline_samples)
    print(f"  Snippets: {len(outline_samples)}, coverage: {coverage:,} chars")

    snippets_text = "\n\n---\n\n".join(
        f"[~{sample['position_percent']:.0f}%] {sample['text']}" for sample in outline_samples
    )
    outline = await run_llm_cli(OUTLINE_PROMPT.format(n=len(outline_samples), snippets_text=snippets_text))
    if not outline.strip():
        raise RuntimeError("Global outline generation returned empty output")
    print(f"  Outline generated in {time.time() - t0:.2f}s ({len(outline):,} chars)")
    return outline


async def extract_dynamic_keywords(outline: str) -> list[str]:
    t0 = time.time()
    print("\n--- Extracting dynamic keywords from outline ---")
    result = await run_llm_cli(KEYWORD_EXTRACTION_PROMPT.format(outline=outline))
    parsed = extract_first_json_value(result)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        print("  Warning: Could not parse dynamic keywords, falling back to static")
        return STATIC_KEYWORDS
    keywords = combine_keywords(parsed)
    print(f"  Extracted {len(keywords)} keywords in {time.time() - t0:.2f}s")
    return keywords


def chunk_cache_base(book_file: str, chunk_id: str, micro_chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256("".join(chunk["text"] for chunk in micro_chunks).encode("utf-8")).hexdigest()[:12]
    return f"{book_file}.{NTSMR_VERSION}.{CACHE_SCHEMA_VERSION}.{MICRO_CHUNK_SIZE}.{chunk_id}.{digest}"


def normalize_cached_micro_chunks(cached_chunks: Any) -> list[dict[str, Any]]:
    if not isinstance(cached_chunks, list):
        raise ValueError("Cached chunks payload must be a list")

    normalized = []
    for index, item in enumerate(cached_chunks):
        if isinstance(item, str):
            normalized.append(
                {
                    "source_index": index,
                    "start_char": None,
                    "end_char": None,
                    "text": item,
                }
            )
            continue
        if not isinstance(item, dict):
            raise ValueError("Cached chunk must be a string or object")
        text = item.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Cached chunk missing text")
        start_char = item.get("start_char")
        end_char = item.get("end_char")
        normalized.append(
            {
                "source_index": int(item.get("source_index", index)),
                "start_char": int(start_char) if start_char is not None else None,
                "end_char": int(end_char) if end_char is not None else None,
                "text": text,
            }
        )
    return normalized


def load_or_build_index(book_file: str, chunk_id: str, micro_chunks: list[dict[str, Any]]):
    cache_base = chunk_cache_base(book_file, chunk_id, micro_chunks)
    faiss_path = f"{cache_base}.faiss"
    chunks_path = f"{cache_base}.chunks.json"
    if os.path.exists(faiss_path) and os.path.exists(chunks_path):
        try:
            index = faiss.read_index(faiss_path)
            with open(chunks_path, "r", encoding="utf-8") as handle:
                raw_cache = json.load(handle)
            # Handle both old format (plain list) and new format (dict with source_file)
            if isinstance(raw_cache, dict) and "chunks" in raw_cache:
                # Verify source file matches expected book (contamination detection)
                cached_source = raw_cache.get("source_file")
                expected_source = os.path.basename(book_file)
                if cached_source and cached_source != expected_source:
                    print(f"    [WARN] Cache source mismatch for {chunk_id}: cached={cached_source}, expected={expected_source} — rebuilding index")
                    return None, None, faiss_path
                cached_chunks = normalize_cached_micro_chunks(raw_cache["chunks"])
            else:
                # Old format without source_file — accept but can't verify
                cached_chunks = normalize_cached_micro_chunks(raw_cache)
            return index, cached_chunks, faiss_path
        except Exception:
            pass
    return None, None, faiss_path


async def build_faiss_index(book_file: str, chunk_id: str, micro_chunks: list[dict[str, Any]]):
    index, cached_chunks, faiss_path = load_or_build_index(book_file, chunk_id, micro_chunks)
    if index is not None and cached_chunks is not None:
        return index, cached_chunks, faiss_path, True

    embeddings = await embed_many([chunk["text"] for chunk in micro_chunks])
    valid_pairs = [(chunk, embedding) for chunk, embedding in zip(micro_chunks, embeddings) if embedding]
    if not valid_pairs:
        raise RuntimeError(f"No embeddings generated for {chunk_id}")

    valid_micro_chunks, valid_embeddings = zip(*valid_pairs)
    embedding_array = np.array(valid_embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embedding_array.shape[1])
    faiss.normalize_L2(embedding_array)
    index.add(embedding_array)

    cache_base = chunk_cache_base(book_file, chunk_id, micro_chunks)
    faiss_path = f"{cache_base}.faiss"
    chunks_path = f"{cache_base}.chunks.json"
    faiss.write_index(index, faiss_path)
    cache_data = {"source_file": os.path.basename(book_file), "chunks": list(valid_micro_chunks)}
    with open(chunks_path, "w", encoding="utf-8") as handle:
        json.dump(cache_data, handle, ensure_ascii=False)
    return index, list(valid_micro_chunks), faiss_path, False


def select_snippets_for_chunk(
    chunk_id: str,
    valid_micro_chunks: list[dict[str, Any]],
    index: Any,
    keyword_vectors: list[tuple[str, list[float]]],
) -> list[dict[str, Any]]:
    best_by_index: dict[int, float] = {}
    for _, keyword_embedding in keyword_vectors:
        q_vec = np.array([keyword_embedding], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = index.search(q_vec, RETRIEVAL_TOP_K)
        for score, idx in zip(scores[0], indices[0]):
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= len(valid_micro_chunks):
                continue
            best_by_index[idx_int] = max(float(score), best_by_index.get(idx_int, float("-inf")))

    selected_indices = sorted(best_by_index, key=lambda idx: (-best_by_index[idx], idx))[:MAX_SNIPPETS_PER_CHUNK]
    selected_indices.sort()
    snippets = []
    for ordinal, idx in enumerate(selected_indices, start=1):
        source_chunk = valid_micro_chunks[idx]
        snippets.append(
            {
                "chunk_id": chunk_id,
                "snippet_id": f"{chunk_id}-s{ordinal}",
                "source_index": source_chunk["source_index"],
                "start_char": source_chunk.get("start_char"),
                "end_char": source_chunk.get("end_char"),
                "score": round(best_by_index[idx], 6),
                "text": source_chunk["text"],
            }
        )
    if not snippets:
        raise RuntimeError(f"No snippets retrieved for {chunk_id}")
    return snippets


def render_snippets(snippets: list[dict[str, Any]]) -> str:
    rendered = []
    for snippet in snippets:
        rendered.append(f"[{snippet['snippet_id']}]\n{snippet['text']}")
    return "\n\n---\n\n".join(rendered)


async def extract_events_for_chunk(
    chunk_id: str,
    snippets: list[dict[str, Any]],
    outline_context: str,
) -> list[dict[str, Any]]:
    prompt = f"""You are a structural analysis sub-agent analyzing {chunk_id}.

Here is the GLOBAL OUTLINE CONTEXT of the novel:
{outline_context}

Here are retrieved snippets from your specific chunk. Each snippet has a stable snippet ID:
{render_snippets(snippets)}

Task:
Extract the major structural beats from these snippets and output them as strict JSON Lines (JSONL).
Do not hallucinate events outside these snippets. Use the outline only for identity and continuity.

Each JSON line must match this schema:
{{"snippet_ids": ["{chunk_id}-s1"], "type": "action|dialogue|exposition|bureaucracy", "framework_signals": ["todorov:disruption"], "summary": "Brief 1-sentence summary"}}

Rules:
- snippet_ids must contain 1-3 IDs from the retrieved snippets above.
- framework_signals must only use this vocabulary:
{SIGNAL_VOCABULARY_TEXT}
- Include 1-5 relevant framework_signals per event when applicable.
- Output ONLY the JSONL section, no markdown code fences.

Format your response EXACTLY like this:
## JSONL
{{"snippet_ids": ["..."], "type": "...", "framework_signals": ["..."], "summary": "..."}}
"""

    allowed_snippet_ids = {snippet["snippet_id"] for snippet in snippets}
    output = await run_llm_cli(prompt)
    try:
        return parse_extraction_output(output, chunk_id, allowed_snippet_ids)
    except Exception as exc:
        repair_prompt = prompt + f"\n\nYour previous output failed validation: {exc}. Re-output corrected JSONL only."
        repair_output = await run_llm_cli(repair_prompt, retries=1)
        return parse_extraction_output(repair_output, chunk_id, allowed_snippet_ids)


async def extract_substrate_for_chunk(
    chunk_id: str,
    snippets: list[dict[str, Any]],
    outline_context: str,
) -> dict[str, list[dict[str, Any]]]:
    prompt = SUBSTRATE_EXTRACTION_PROMPT.format(
        chunk_id=chunk_id,
        outline_context=outline_context,
        snippets_text=render_snippets(snippets),
    )
    allowed_snippet_ids = {snippet["snippet_id"] for snippet in snippets}
    output = await run_llm_cli(prompt)
    try:
        return parse_substrate_output(output, chunk_id, allowed_snippet_ids)
    except Exception as exc:
        repair_prompt = prompt + f"\n\nYour previous output failed validation: {exc}. Re-output corrected JSON only."
        repair_output = await run_llm_cli(repair_prompt, retries=1)
        return parse_substrate_output(repair_output, chunk_id, allowed_snippet_ids)


async def extract_events_and_substrate_for_chunk(
    chunk_id: str,
    snippets: list[dict[str, Any]],
    outline_context: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Merged events + substrate extraction in a single LLM call to save per-call overhead."""
    prompt = f"""You are a structural analysis sub-agent analyzing {chunk_id}.

Here is the GLOBAL OUTLINE CONTEXT of the novel:
{outline_context}

Here are retrieved snippets from your specific chunk. Each snippet has a stable snippet ID:
{render_snippets(snippets)}

You have TWO tasks. Output both sections in order.

## TASK 1: STRUCTURAL EVENTS
Extract the major structural beats from these snippets as strict JSON Lines (JSONL).
Do not hallucinate events outside these snippets. Use the outline only for identity and continuity.

Each JSON line must match this schema:
{{"snippet_ids": ["{chunk_id}-s1"], "type": "action|dialogue|exposition|bureaucracy", "framework_signals": ["todorov:disruption"], "summary": "Brief 1-sentence summary"}}

Rules:
- snippet_ids must contain 1-3 IDs from the retrieved snippets above.
- framework_signals must only use this vocabulary:
{SIGNAL_VOCABULARY_TEXT}
- Include 1-5 relevant framework_signals per event when applicable.

## TASK 2: NARRATIVE SUBSTRATE
Extract reusable narrative substrate records as a single JSON object with keys "characters", "dialogue_turns", "settings".

Characters schema: {{"snippet_ids": ["{chunk_id}-s1"], "name": "Name", "aliases": [], "role": "Role", "summary": "Brief description"}}
Dialogue schema: {{"snippet_ids": ["{chunk_id}-s2"], "speaker": "Name", "addressee": "Name or unknown", "speech_act": "command|confession|debate|exposition|revelation|request|threat|warning|other", "summary": "Brief summary"}}
Settings schema: {{"snippet_ids": ["{chunk_id}-s3"], "location": "Place", "time_marker": "When", "social_context": "Context", "summary": "Brief description"}}

Rules for both tasks:
- Use only the provided snippet_ids. Do not invent names, locations, or events absent from the snippets.
- No markdown code fences. No prose outside the data sections.

Format your response EXACTLY like this:
## JSONL
{{"snippet_ids": ["..."], "type": "...", "framework_signals": ["..."], "summary": "..."}}
{{"snippet_ids": ["..."], "type": "...", "framework_signals": ["..."], "summary": "..."}}

## SUBSTRATE
{{"characters": [...], "dialogue_turns": [...], "settings": [...]}}
"""

    allowed_snippet_ids = {snippet["snippet_id"] for snippet in snippets}
    output = await run_llm_cli(prompt)

    # Split output into events section and substrate section
    events = None
    substrate = None

    # Try to parse the combined output
    parts = output.split("## SUBSTRATE")
    if len(parts) >= 2:
        events_section = parts[0]
        substrate_section = parts[1]
        try:
            events = parse_extraction_output(events_section, chunk_id, allowed_snippet_ids)
        except Exception:
            pass
        try:
            substrate = parse_substrate_output(substrate_section, chunk_id, allowed_snippet_ids)
        except Exception:
            pass

    # Fall back to individual calls for whichever part failed
    if events is None:
        print(f"  [{chunk_id}] Merged events parse failed, falling back to individual call")
        events = await extract_events_for_chunk(chunk_id, snippets, outline_context)
    if substrate is None:
        print(f"  [{chunk_id}] Merged substrate parse failed, falling back to individual call")
        substrate = await extract_substrate_for_chunk(chunk_id, snippets, outline_context)

    return events, substrate


def _empty_substrate() -> dict[str, list[dict[str, Any]]]:
    return {"characters": [], "dialogue_turns": [], "settings": []}


async def process_short_text(
    text: str,
    outline_context: str,
    skip_substrate: bool = False,
) -> dict[str, Any]:
    """Process short texts (<SHORT_TEXT_THRESHOLD chars) without FAISS retrieval.

    Uses the entire text as snippets, guaranteeing 100% coverage.
    """
    chunk_id = "chunk-1"
    chunk_start = time.time()
    print(f"[short-text] Processing {len(text):,} chars as single chunk (no FAISS)")
    snippets = []
    for i, start in enumerate(range(0, len(text), MICRO_CHUNK_SIZE)):
        chunk_text = text[start : start + MICRO_CHUNK_SIZE]
        snippets.append({
            "chunk_id": chunk_id,
            "snippet_id": f"{chunk_id}-s{i + 1}",
            "source_index": i,
            "start_char": start,
            "end_char": start + len(chunk_text),
            "score": 1.0,
            "text": chunk_text,
        })
    if skip_substrate:
        events = await extract_events_for_chunk(chunk_id, snippets, outline_context)
        substrate = _empty_substrate()
    else:
        events, substrate = await extract_events_and_substrate_for_chunk(chunk_id, snippets, outline_context)
    print(f"[short-text] Extracted {len(events)} events from {len(snippets)} snippets")
    return {
        "chunk_id": chunk_id,
        "cache_path": "short-text-bypass",
        "snippet_count": len(snippets),
        "snippets": snippets,
        "event_count": len(events),
        "events": events,
        "character_count": len(substrate["characters"]),
        "dialogue_count": len(substrate["dialogue_turns"]),
        "setting_count": len(substrate["settings"]),
        "characters": substrate["characters"],
        "dialogue_turns": substrate["dialogue_turns"],
        "settings": substrate["settings"],
        "duration_seconds": round(time.time() - chunk_start, 2),
    }


async def process_chunk(
    chunk_idx: int,
    micro_chunks: list[dict[str, Any]],
    keyword_vectors: list[tuple[str, list[float]]],
    outline_context: str,
    book_file: str,
    skip_substrate: bool = False,
) -> dict[str, Any]:
    chunk_id = f"chunk-{chunk_idx}"
    async with CHUNK_SEMAPHORE:
        chunk_start = time.time()
        print(f"[{chunk_id}] Generating or loading embeddings...")
        index, valid_micro_chunks, faiss_path, from_cache = await build_faiss_index(book_file, chunk_id, micro_chunks)
        print(
            f"[{chunk_id}] Embeddings ready in {time.time() - chunk_start:.2f}s "
            f"({'cache' if from_cache else 'fresh'}, {len(valid_micro_chunks)} micro-chunks)"
        )

        snippets = select_snippets_for_chunk(chunk_id, valid_micro_chunks, index, keyword_vectors)
        if skip_substrate:
            events = await extract_events_for_chunk(chunk_id, snippets, outline_context)
            substrate = _empty_substrate()
        else:
            events, substrate = await extract_events_and_substrate_for_chunk(chunk_id, snippets, outline_context)
        print(f"[{chunk_id}] Extracted {len(events)} validated events")
        return {
            "chunk_id": chunk_id,
            "cache_path": faiss_path,
            "snippet_count": len(snippets),
            "snippets": snippets,
            "event_count": len(events),
            "events": events,
            "character_count": len(substrate["characters"]),
            "dialogue_count": len(substrate["dialogue_turns"]),
            "setting_count": len(substrate["settings"]),
            "characters": substrate["characters"],
            "dialogue_turns": substrate["dialogue_turns"],
            "settings": substrate["settings"],
            "duration_seconds": round(time.time() - chunk_start, 2),
        }


async def synthesize_single_framework(
    framework_name: str,
    framework_config: dict[str, Any],
    events: list[dict[str, Any]],
    outline_context: str,
    source_text_sample: str = "",
    character_names: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    async with SYNTHESIS_SEMAPHORE:
        selected_events, used_fallback = filter_events_for_framework(events, framework_config.get("filter_tag"))
        event_jsonl = "\n".join(json.dumps(event, ensure_ascii=False) for event in selected_events)
        allowed_event_ids = {event["event_id"] for event in selected_events}

        source_section = ""
        if source_text_sample:
            source_section = f"\n## SOURCE TEXT SAMPLES\n{source_text_sample}\n"

        character_section = ""
        if character_names:
            names_list = ", ".join(character_names[:50])
            character_section = f"\n## VERIFIED CHARACTER NAMES\nOnly use these character names: {names_list}\nDo not reference characters not in this list. If unsure of a name, describe the character's role instead.\n"

        prompt = f"""You are a Synthesis Sub-Agent specializing in {framework_name}.

IMPORTANT: This analysis is ONLY about the work described in the CHARACTER REFERENCE below. Do not reference events, characters, themes, or plot points from any other work. All claims must be grounded in the provided source text snippets and event timeline.

## CHARACTER REFERENCE
{outline_context}
{character_section}{source_section}
## STRUCTURAL EVENT TIMELINE
{event_jsonl}

Task:
Output a single JSON object with exactly these keys:
{{
  "analysis": <the framework analysis object>,
  "evidence_event_ids": ["event ids from the timeline above"],
  "confidence": <float 0.0-1.0, how confident you are in this analysis>
}}

Rules:
- evidence_event_ids must contain 1-8 event IDs copied exactly from the timeline above.
- confidence: 1.0 = highly confident with strong textual evidence, 0.5 = moderate confidence, 0.0 = speculative.
- Only use character names that appear in the CHARACTER REFERENCE or VERIFIED CHARACTER NAMES sections. If unsure of a name, describe the character's role instead.
- Only use direct quotes that appear verbatim in the provided snippets or event timeline. If you cannot find the exact wording, paraphrase without quotation marks.
- Anchor each claim to specific events or scenes from the provided material. Do not reference plot events that are not in the event list.
- The analysis value must match this schema:
{framework_config['prompt_suffix']}
- Output JSON only. No markdown. No prose.
"""
        output = await run_llm_cli(prompt)
        try:
            payload = extract_first_json_value(output)
            validated = validate_framework_result(framework_name, payload, allowed_event_ids)
        except Exception as exc:
            # Targeted repair: send only schema + broken output + error (not full prompt)
            allowed_ids_sample = sorted(allowed_event_ids)[:10]
            repair_prompt = f"""Fix this {framework_name} synthesis output. The validation error was:
{exc}

Your broken output was:
{output[:3000]}

Required JSON schema:
{{
  "analysis": {framework_config['prompt_suffix']},
  "evidence_event_ids": {json.dumps(allowed_ids_sample)} (1-8 IDs from this set),
  "confidence": <float 0.0-1.0>
}}

Output corrected JSON only. No markdown. No prose."""
            repair_output = await run_llm_cli(repair_prompt, retries=1)
            payload = extract_first_json_value(repair_output)
            validated = validate_framework_result(framework_name, payload, allowed_event_ids)

        validated["used_full_timeline_fallback"] = used_fallback
        validated["event_count"] = len(selected_events)

        # Post-synthesis name check: log unknown names (exact match only)
        if character_names:
            analysis_text = json.dumps(validated.get("analysis", {}), ensure_ascii=False)
            names_set = set(character_names)
            for name in names_set:
                # Skip very short names that would false-positive
                if len(name) < 3:
                    continue
            # Check for capitalized words in analysis that might be character names
            words_in_analysis = set(re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b', analysis_text))
            unknown_names = [w for w in words_in_analysis if w not in names_set and len(w) > 3]
            # Filter out common framework terms
            framework_terms = {"Hero", "Shadow", "Trickster", "Self", "Journey", "Quest", "Act", "Setup",
                             "Climax", "Resolution", "Rising", "Falling", "Action", "Exposition", "Return",
                             "Departure", "Threshold", "Ordeal", "Change", "Need", "Find", "Take",
                             "Image", "Catalyst", "Debate", "Midpoint", "Finale", "Introduction",
                             "Development", "Twist", "Conclusion", "Description", "None", "True", "False"}
            unknown_names = [n for n in unknown_names if n not in framework_terms]
            if unknown_names:
                print(f"    [WARN] {framework_name}: possible unknown names: {', '.join(unknown_names[:5])}")

        return framework_name, validated


async def synthesize_frameworks(
    events: list[dict[str, Any]],
    outline_context: str,
    framework_prompts: dict[str, dict[str, Any]] | None = None,
    source_text_sample: str = "",
    character_names: list[str] | None = None,
    synthesis_cache_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    framework_prompts = framework_prompts or FRAMEWORK_SYNTHESIS_PROMPTS

    # Load partial results from a previous interrupted run
    cached_results: dict[str, dict[str, Any]] = {}
    if synthesis_cache_path and os.path.exists(synthesis_cache_path):
        try:
            with open(synthesis_cache_path, "r", encoding="utf-8") as f:
                cached_results = json.load(f)
            if not isinstance(cached_results, dict):
                cached_results = {}
        except (json.JSONDecodeError, OSError):
            cached_results = {}

    # Filter out already-completed frameworks
    remaining = {k: v for k, v in framework_prompts.items() if k not in cached_results}
    if cached_results and remaining:
        print(f"\n--- Phase 3: Synthesizing {len(remaining)} frameworks ({len(cached_results)} cached) ---")
    elif cached_results:
        print(f"\n--- Phase 3: All {len(cached_results)} frameworks cached, skipping synthesis ---")
        return cached_results
    else:
        print(f"\n--- Phase 3: Synthesizing {len(remaining)} frameworks ---")

    # Accumulate results (start from cache)
    results = dict(cached_results)

    async def _safe_synthesize(fw_name, config):
        try:
            name, payload = await synthesize_single_framework(
                fw_name, config, events, outline_context,
                source_text_sample=source_text_sample,
                character_names=character_names,
            )
            # Write partial result to disk immediately
            if payload is not None and synthesis_cache_path:
                results[name] = payload
                try:
                    with open(synthesis_cache_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                except OSError:
                    pass
            return name, payload
        except Exception as exc:
            print(f"  [ERROR] {fw_name} synthesis failed: {exc}")
            return fw_name, None

    tasks = [_safe_synthesize(fw_name, config) for fw_name, config in remaining.items()]
    pairs = await asyncio.gather(*tasks)
    failed = [name for name, payload in pairs if payload is None]
    if failed:
        print(f"  WARNING: {len(failed)} framework(s) failed: {', '.join(failed)}")

    # Merge new results with cached
    for name, payload in pairs:
        if payload is not None:
            results[name] = payload
    return results


POST_SYNTHESIS_CHECK_PROMPT = """You are a quality-control agent. Given a character outline, source text snippets, and a set of framework analyses, verify factual accuracy.

## CHARACTER OUTLINE
{outline_context}

## VERIFIED CHARACTER NAMES
{character_names_section}

## SOURCE TEXT SNIPPETS (for verification)
{source_snippets_section}

## FRAMEWORK ANALYSES
{synthesis_summary}

Check for:
1. Character names used in analyses that do NOT appear in the outline or VERIFIED CHARACTER NAMES
2. Plot events described in analyses that contradict the outline's plot summary
3. Direct quotes that do not appear verbatim in the source text snippets
4. Any claims that seem fabricated or unsupported by the outline and source text
5. References to events, characters, or themes from other works

Output a JSON object:
{{
  "quotes_verified": true or false,
  "characters_match_source": true or false,
  "no_external_references": true or false,
  "framework_terms_valid": true or false,
  "issues": [
    {{"framework": "<name>", "type": "wrong_name|contradiction|unsupported|misquote|external_reference", "detail": "<description>"}}
  ]
}}

If no issues found, return {{"quotes_verified": true, "characters_match_source": true, "no_external_references": true, "framework_terms_valid": true, "issues": []}}.
Output JSON only."""


async def post_synthesis_check(
    synthesis_payload: dict[str, dict[str, Any]],
    outline_context: str,
    snippets: list[dict[str, Any]] | None = None,
    character_names: list[str] | None = None,
) -> list[dict[str, str]]:
    """Run a structured self-check on synthesis results with source text grounding."""
    summary_parts = []
    for fw_name, fw_data in synthesis_payload.items():
        analysis = fw_data.get("analysis", {})
        summary_parts.append(f"### {fw_name}\n{json.dumps(analysis, indent=1, ensure_ascii=False)}")
    synthesis_summary = "\n\n".join(summary_parts)

    # Only run check if synthesis is substantial enough
    if len(synthesis_payload) <= 1:
        return []

    # Build source snippets section (first 20, truncated)
    source_snippets_section = "None provided."
    if snippets:
        snippet_parts = []
        for s in snippets[:20]:
            text = s.get("text", "")[:500]
            snippet_parts.append(f"[{s.get('snippet_id', '?')}] {text}")
        source_snippets_section = "\n".join(snippet_parts)

    # Build character names section
    character_names_section = "None provided."
    if character_names:
        character_names_section = ", ".join(character_names[:50])

    prompt = POST_SYNTHESIS_CHECK_PROMPT.format(
        outline_context=outline_context,
        synthesis_summary=synthesis_summary[:8000],
        source_snippets_section=source_snippets_section[:10000],
        character_names_section=character_names_section,
    )
    try:
        output = await run_llm_cli(prompt, retries=1)
        result = extract_first_json_value(output)
        issues = result.get("issues", [])
        checks = {k: result.get(k) for k in ("quotes_verified", "characters_match_source", "no_external_references", "framework_terms_valid") if k in result}
        failed_checks = [k for k, v in checks.items() if v is False]
        if issues:
            print(f"  Post-synthesis check found {len(issues)} issues (failed: {', '.join(failed_checks) or 'none'}):")
            for issue in issues[:5]:
                print(f"    [{issue.get('type')}] {issue.get('framework')}: {issue.get('detail', '')[:100]}")
        else:
            print(f"  Post-synthesis check: no issues found")
        return issues
    except Exception as exc:
        print(f"  Post-synthesis check failed (non-fatal): {exc}")
        return []


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_json_safe(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}")
            rows.append(payload)
    return rows


def framework_summary(synthesis_payload: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        framework_name: {
            "event_count": payload["event_count"],
            "used_full_timeline_fallback": payload["used_full_timeline_fallback"],
        }
        for framework_name, payload in synthesis_payload.items()
    }


def select_framework_prompts(score_only: bool = False, framework_filter: list[str] | None = None) -> dict[str, dict[str, Any]]:
    if score_only:
        return {"Quadrant Scores": FRAMEWORK_SYNTHESIS_PROMPTS["Quadrant Scores"]}
    if framework_filter:
        selected = {}
        for name in framework_filter:
            if name in FRAMEWORK_SYNTHESIS_PROMPTS:
                selected[name] = FRAMEWORK_SYNTHESIS_PROMPTS[name]
            else:
                print(f"  Warning: unknown framework '{name}', skipping")
        if not selected:
            raise ValueError(f"No valid frameworks in filter: {framework_filter}")
        return selected
    return FRAMEWORK_SYNTHESIS_PROMPTS


def build_report(
    book_file: str,
    artifact_paths: ArtifactPaths,
    keywords: list[str],
    global_outline: str,
    outline_context: str,
    outline_samples: list[dict[str, Any]],
    chunk_reports: list[dict[str, Any]],
    synthesis_payload: dict[str, dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    llm_config = get_active_llm_config()
    all_events = [event for chunk in chunk_reports for event in chunk["events"]]
    total_snippets = sum(chunk["snippet_count"] for chunk in chunk_reports)
    total_characters = sum(chunk.get("character_count", 0) for chunk in chunk_reports)
    total_dialogue = sum(chunk.get("dialogue_count", 0) for chunk in chunk_reports)
    total_settings = sum(chunk.get("setting_count", 0) for chunk in chunk_reports)
    return to_json_safe({
        "ntsmr_version": NTSMR_VERSION,
        "ntsmr_run_label": llm_config.run_label,
        "book_file": book_file,
        "llm": {
            "backend": llm_config.backend,
            "model": llm_config.model,
            "reasoning_effort": llm_config.reasoning_effort,
        },
        "artifacts": artifact_paths.__dict__,
        "keywords": keywords,
        "global_outline": global_outline,
        "outline_context": outline_context,
        "outline_samples": outline_samples,
        "chunk_count": len(chunk_reports),
        "event_count": len(all_events),
        "substrate": {
            "outline_sample_count": len(outline_samples),
            "snippet_count": total_snippets,
            "character_count": total_characters,
            "dialogue_count": total_dialogue,
            "setting_count": total_settings,
        },
        "frameworks": framework_summary(synthesis_payload),
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "cache_path": chunk["cache_path"],
                "snippet_count": chunk["snippet_count"],
                "event_count": chunk["event_count"],
                "character_count": chunk.get("character_count", 0),
                "dialogue_count": chunk.get("dialogue_count", 0),
                "setting_count": chunk.get("setting_count", 0),
                "duration_seconds": chunk["duration_seconds"],
                "snippets": chunk["snippets"],
            }
            for chunk in chunk_reports
        ],
        "elapsed_seconds": round(elapsed_seconds, 2),
        "token_usage": TOKEN_USAGE.snapshot(),
    })


def build_reused_report(
    source_report: dict[str, Any],
    artifact_paths: ArtifactPaths,
    synthesis_payload: dict[str, dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    llm_config = get_active_llm_config()
    report = dict(source_report)
    report["ntsmr_version"] = NTSMR_VERSION
    report["ntsmr_run_label"] = llm_config.run_label
    report["llm"] = {
        "backend": llm_config.backend,
        "model": llm_config.model,
        "reasoning_effort": llm_config.reasoning_effort,
    }
    report["artifacts"] = artifact_paths.__dict__
    report["frameworks"] = framework_summary(synthesis_payload)
    report["elapsed_seconds"] = round(elapsed_seconds, 2)
    report["token_usage"] = TOKEN_USAGE.snapshot()
    report["source_run_label"] = source_report.get("ntsmr_run_label")
    report["source_artifacts"] = source_report.get("artifacts")
    return to_json_safe(report)


def copy_artifact_if_present(source_path: str, destination_path: str) -> None:
    if source_path == destination_path or not os.path.exists(source_path):
        return
    shutil.copyfile(source_path, destination_path)


async def main():
    parser = argparse.ArgumentParser(description="NTSMR Pipeline — Narrative Topology Semantic Map Reduce")
    parser.add_argument("book_file", help="Path to the raw text file")
    parser.add_argument("output_file", help="Path to save the events JSONL or artifact base path")
    parser.add_argument("--llm-backend", choices=sorted(SUPPORTED_LLM_BACKENDS), default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-reasoning-effort", choices=sorted(SUPPORTED_REASONING_EFFORTS | CLAUDE_REASONING_EFFORTS), default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--reuse-from", default=None, help="Reuse existing outline/events artifacts and rerun synthesis only")
    parser.add_argument("--score-only", action="store_true", help="Only synthesize the Quadrant Scores block")
    parser.add_argument("--frameworks", type=str, default=None, help="Comma-separated list of framework names to synthesize (e.g., 'Quadrant Scores,Todorov')")
    parser.add_argument("--skip-substrate", action="store_true", help="Skip substrate extraction (characters, dialogue, settings) — events only")
    args = parser.parse_args()

    global ACTIVE_LLM_CONFIG
    ACTIVE_LLM_CONFIG = build_llm_config(
        backend=args.llm_backend,
        model=args.llm_model,
        reasoning_effort=args.llm_reasoning_effort,
        run_label=args.run_label,
    )

    TOKEN_USAGE.reset()
    start_time = time.time()
    print(
        f"LLM backend: {ACTIVE_LLM_CONFIG.backend}, model: {ACTIVE_LLM_CONFIG.model}, "
        f"reasoning: {ACTIVE_LLM_CONFIG.reasoning_effort or 'default'}, "
        f"run label: {ACTIVE_LLM_CONFIG.run_label}"
    )
    framework_filter = [f.strip() for f in args.frameworks.split(",")] if args.frameworks else None
    framework_prompts = select_framework_prompts(score_only=args.score_only, framework_filter=framework_filter)

    artifact_paths = resolve_artifact_paths(args.output_file)
    for path_str in [
        artifact_paths.events_path,
        artifact_paths.synthesis_path,
        artifact_paths.report_path,
        artifact_paths.outline_path,
        artifact_paths.snippets_path,
        artifact_paths.characters_path,
        artifact_paths.dialogue_path,
        artifact_paths.settings_path,
    ]:
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_from:
        source_artifacts = resolve_artifact_paths(args.reuse_from)
        outline_payload = json.loads(Path(source_artifacts.outline_path).read_text(encoding="utf-8"))
        source_report = json.loads(Path(source_artifacts.report_path).read_text(encoding="utf-8"))
        all_events = read_jsonl(source_artifacts.events_path)
        outline_context = outline_payload["outline_context"]
        global_outline = outline_payload.get("global_outline", outline_context)
        character_names = extract_character_names(global_outline)

        # Load source text sample if book file is available
        source_text_sample = ""
        book_file = source_report.get("book_file", args.book_file)
        if book_file and os.path.exists(book_file):
            with open(book_file, "r", encoding="utf-8") as handle:
                source_text = handle.read()
            source_text_sample = build_source_text_sample(source_text)

        synthesis_payload = await synthesize_frameworks(
            all_events,
            outline_context,
            framework_prompts=framework_prompts,
            source_text_sample=source_text_sample,
            character_names=character_names,
            synthesis_cache_path=artifact_paths.synthesis_path,
        )

        for source_path, destination_path in [
            (source_artifacts.outline_path, artifact_paths.outline_path),
            (source_artifacts.snippets_path, artifact_paths.snippets_path),
            (source_artifacts.characters_path, artifact_paths.characters_path),
            (source_artifacts.dialogue_path, artifact_paths.dialogue_path),
            (source_artifacts.settings_path, artifact_paths.settings_path),
            (source_artifacts.events_path, artifact_paths.events_path),
        ]:
            copy_artifact_if_present(source_path, destination_path)

        # Load snippets from reused artifacts for self-check grounding
        reuse_snippets = []
        if os.path.exists(source_artifacts.snippets_path):
            reuse_snippets = read_jsonl(source_artifacts.snippets_path)
        check_issues = await post_synthesis_check(synthesis_payload, outline_context, snippets=reuse_snippets, character_names=character_names)

        with open(artifact_paths.synthesis_path, "w", encoding="utf-8") as handle:
            json.dump(synthesis_payload, handle, indent=2, ensure_ascii=False)

        report_payload = build_reused_report(
            source_report=source_report,
            artifact_paths=artifact_paths,
            synthesis_payload=synthesis_payload,
            elapsed_seconds=time.time() - start_time,
        )
        if check_issues:
            report_payload["post_synthesis_issues"] = check_issues
        report_payload["score_only"] = args.score_only
        with open(artifact_paths.report_path, "w", encoding="utf-8") as handle:
            json.dump(report_payload, handle, indent=2, ensure_ascii=False)
        usage = TOKEN_USAGE.snapshot()
        print(f"Reused artifacts from {args.reuse_from}; synthesized {len(synthesis_payload)} frameworks.")
        if usage["llm_calls"] > 0:
            print(f"Token usage: {usage['input_tokens']:,} in + {usage['output_tokens']:,} out — ${usage['total_cost_usd']:.4f}")
        return

    with open(args.book_file, "r", encoding="utf-8") as handle:
        text = handle.read()
    validate_source_text(text, args.book_file)

    is_short_text = len(text) < SHORT_TEXT_THRESHOLD
    if is_short_text:
        print(f"Loaded {args.book_file}: {len(text):,} characters (short text mode).")
    else:
        macro_chunks = [
            {
                "start_char": index,
                "end_char": min(index + MACRO_CHUNK_SIZE, len(text)),
                "text": text[index : index + MACRO_CHUNK_SIZE],
            }
            for index in range(0, len(text), MACRO_CHUNK_SIZE)
        ]
        print(f"Loaded {args.book_file}: {len(text):,} characters, {len(macro_chunks)} macro-chunks.")

    outline_samples = build_outline_samples(text)

    # Content-addressed outline cache: skip LLM call if source text unchanged
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    outline_cache_hit = False
    if os.path.exists(artifact_paths.outline_path):
        try:
            with open(artifact_paths.outline_path, "r", encoding="utf-8") as f:
                cached_outline = json.load(f)
            if cached_outline.get("source_hash") == source_hash:
                global_outline = cached_outline["global_outline"]
                outline_context = cached_outline["outline_context"]
                outline_cache_hit = True
                print(f"\n--- Phase 1: Outline cache hit (hash={source_hash}) ---")
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    if not outline_cache_hit:
        global_outline = await generate_global_outline(outline_samples)
        outline_context = build_outline_context(global_outline)

    if is_short_text:
        keywords = STATIC_KEYWORDS
        print(f"  Short text mode: skipping keyword extraction and FAISS retrieval")
        print(f"\n--- Phase 2: Short Text Direct Extraction ---")
        short_report = await process_short_text(text, outline_context, skip_substrate=args.skip_substrate)
        chunk_reports = [short_report]
    else:
        dynamic_keywords = await extract_dynamic_keywords(global_outline)
        keywords = combine_keywords(dynamic_keywords)
        print(f"  Keyword set size: {len(keywords)}")

        keyword_embeddings = await embed_many(keywords)
        keyword_vectors = [
            (keyword, embedding)
            for keyword, embedding in zip(keywords, keyword_embeddings)
            if embedding
        ]
        if not keyword_vectors:
            raise RuntimeError("No keyword embeddings could be generated")

        print(f"\n--- Phase 2: Parallel Fan-Out Chunk Extraction ({len(macro_chunks)} chunks) ---")
        chunk_reports = await asyncio.gather(
            *[
                process_chunk(
                    chunk_idx + 1,
                    build_micro_chunk_records(macro_chunk["text"], start_char=macro_chunk["start_char"]),
                    keyword_vectors,
                    outline_context,
                    args.book_file,
                    skip_substrate=args.skip_substrate,
                )
                for chunk_idx, macro_chunk in enumerate(macro_chunks)
            ]
        )

    all_events = [event for chunk in chunk_reports for event in chunk["events"]]
    print(f"Phase 2 complete: {len(all_events)} total validated events.")

    source_text_sample = build_source_text_sample(text)
    character_names = extract_character_names(global_outline)

    synthesis_payload = await synthesize_frameworks(
        all_events,
        outline_context,
        framework_prompts=framework_prompts,
        source_text_sample=source_text_sample,
        character_names=character_names,
        synthesis_cache_path=artifact_paths.synthesis_path,
    )
    all_snippets = [snippet for chunk in chunk_reports for snippet in chunk["snippets"]]
    all_characters = [record for chunk in chunk_reports for record in chunk["characters"]]
    all_dialogue = [record for chunk in chunk_reports for record in chunk["dialogue_turns"]]
    all_settings = [record for chunk in chunk_reports for record in chunk["settings"]]

    outline_payload = {
        "ntsmr_version": NTSMR_VERSION,
        "ntsmr_run_label": ACTIVE_LLM_CONFIG.run_label,
        "book_file": args.book_file,
        "source_hash": source_hash,
        "llm": {
            "backend": ACTIVE_LLM_CONFIG.backend,
            "model": ACTIVE_LLM_CONFIG.model,
            "reasoning_effort": ACTIVE_LLM_CONFIG.reasoning_effort,
        },
        "outline_samples": outline_samples,
        "global_outline": global_outline,
        "outline_context": outline_context,
        "keywords": keywords,
    }
    with open(artifact_paths.outline_path, "w", encoding="utf-8") as handle:
        json.dump(to_json_safe(outline_payload), handle, indent=2, ensure_ascii=False)

    if is_short_text:
        check_issues = []
        print("  Skipping post-synthesis self-check for short text")
    else:
        check_issues = await post_synthesis_check(synthesis_payload, outline_context, snippets=all_snippets, character_names=character_names)

    write_jsonl(artifact_paths.snippets_path, all_snippets)
    write_jsonl(artifact_paths.characters_path, all_characters)
    write_jsonl(artifact_paths.dialogue_path, all_dialogue)
    write_jsonl(artifact_paths.settings_path, all_settings)
    write_jsonl(artifact_paths.events_path, all_events)

    with open(artifact_paths.synthesis_path, "w", encoding="utf-8") as handle:
        json.dump(synthesis_payload, handle, indent=2, ensure_ascii=False)

    report_payload = build_report(
        args.book_file,
        artifact_paths,
        keywords,
        global_outline,
        outline_context,
        outline_samples,
        chunk_reports,
        synthesis_payload,
        time.time() - start_time,
    )
    if check_issues:
        report_payload["post_synthesis_issues"] = check_issues
    report_payload["score_only"] = args.score_only
    with open(artifact_paths.report_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False)

    usage = TOKEN_USAGE.snapshot()
    print(f"\nDone! Pipeline completed in {time.time() - start_time:.2f} seconds.")
    if usage["llm_calls"] > 0:
        print(f"Token usage: {usage['input_tokens']:,} in + {usage['output_tokens']:,} out "
              f"({usage['cache_creation_tokens']:,} cache-create, {usage['cache_read_tokens']:,} cache-read) "
              f"across {usage['llm_calls']} LLM calls — ${usage['total_cost_usd']:.4f}")
    print(f"Outline: {artifact_paths.outline_path}")
    print(f"Snippets: {artifact_paths.snippets_path}")
    print(f"Characters: {artifact_paths.characters_path}")
    print(f"Dialogue: {artifact_paths.dialogue_path}")
    print(f"Settings: {artifact_paths.settings_path}")
    print(f"Events: {artifact_paths.events_path}")
    print(f"Synthesis: {artifact_paths.synthesis_path}")
    print(f"Report: {artifact_paths.report_path}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

NTSMR_VERSION = "2.2"

MACRO_CHUNK_SIZE = 150_000
MICRO_CHUNK_SIZE = 4_000
SNIPPET_SIZE = 1_500
N_SNIPPETS = 30
MAX_SNIPPETS_PER_CHUNK = 12
RETRIEVAL_TOP_K = 2
MIN_EVENTS_PER_CHUNK = 1
MIN_TAGGED_EVENTS = 2
MAX_OUTLINE_CONTEXT_CHARS = 5_000

EMBED_CONCURRENCY = int(os.environ.get("NTSMR_EMBED_CONCURRENCY", "8"))
CHUNK_CONCURRENCY = int(os.environ.get("NTSMR_CHUNK_CONCURRENCY", "2"))
SYNTHESIS_CONCURRENCY = int(os.environ.get("NTSMR_SYNTHESIS_CONCURRENCY", "2"))
GEMINI_TIMEOUT_SECONDS = int(os.environ.get("NTSMR_GEMINI_TIMEOUT_SECONDS", "180"))
GEMINI_RETRIES = int(os.environ.get("NTSMR_GEMINI_RETRIES", "2"))
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("NTSMR_OLLAMA_TIMEOUT_SECONDS", "10"))

EVENT_TYPES = {"action", "dialogue", "exposition", "bureaucracy"}

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
        "prompt_suffix": """Output exactly 6 floats between 0.0 and 1.0 for these metrics based on the timeline's pacing, conflict types, and plot structure:\n- time_linearity: 0.0=Linear, 1.0=Fractured\n- pacing_velocity: 0.0=Action-Driven, 1.0=Observational\n- threat_scale: 0.0=Individual, 1.0=Systemic\n- protagonist_fate: 0.0=Victory, 1.0=Assimilation\n- conflict_style: 0.0=Western Combat, 1.0=Kishotenketsu\n- price_type: 0.0=Physical, 1.0=Ideological\n\nScoring guidance:\n- Score the dominant narrative logic of the whole work, not just the loudest climax.\n- For time_linearity, documentary flashbacks, embedded records, recurring memory loops, and frame narration that repeatedly interrupts the forward plot should push upward even if the surface chronology is mostly sequential.\n- For pacing_velocity, sustained observation, reflection, investigation, and environmental interpretation should push upward even if a few scenes are violent.\n- For threat_scale, institutions, class systems, biopolitical sorting, civilizational infrastructures, and other social logics count as systemic even when the immediate scene is intimate.\n- For conflict_style, ask whether meaning is driven mainly by combat/opposition or by revelation, juxtaposition, atmosphere, and perspective shift.\n- Isolated violent climaxes should not outweigh a largely contemplative or exploratory structure.\n- For price_type, costs paid in identity, autonomy, status, belief, memory, relational bonds, or moral worldview should push toward ideological even when bodies are also at risk.\n\nThe analysis object must be:\n{\n  \"time_linearity\": 0.0,\n  \"pacing_velocity\": 0.0,\n  \"threat_scale\": 0.0,\n  \"protagonist_fate\": 0.0,\n  \"conflict_style\": 0.0,\n  \"price_type\": 0.0\n}""",
    },
    "The Freytag Pyramid": {
        "filter_tag": "freytag",
        "prompt_suffix": """Map the narrative arc to Freytag's five dramatic stages.\n\nThe analysis object must be:\n{\n  \"exposition\": \"Setup of world, characters, initial situation\",\n  \"rising_action\": \"Key complications and escalation toward the turning point\",\n  \"climax\": \"The decisive turning point or moment of highest tension\",\n  \"falling_action\": \"Consequences and unwinding after the climax\",\n  \"denouement\": \"Final resolution and the new state of affairs\"\n}""",
    },
    "The Three-Act Structure": {
        "filter_tag": "three_act",
        "prompt_suffix": """Map the narrative to the three-act structure, identifying the two key plot points.\n\nThe analysis object must be:\n{\n  \"act_1_setup\": \"The world, characters, and status quo before the inciting incident\",\n  \"plot_point_1\": \"The event that launches the protagonist into the central conflict\",\n  \"act_2_confrontation\": \"The escalating obstacles, complications, and midpoint reversal\",\n  \"plot_point_2\": \"The crisis that forces the final confrontation\",\n  \"act_3_resolution\": \"The climax and its aftermath\"\n}""",
    },
    "The Monomyth": {
        "filter_tag": "monomyth",
        "prompt_suffix": """Analyze how this narrative relates to Campbell's Hero's Journey. Focus on how the work subverts or departs from the monomyth template.\n\nThe analysis object must be:\n{\n  \"applicable_stages\": [\"List the monomyth stages that appear in this narrative\"],\n  \"subversions\": \"How does this work depart from or subvert the traditional Hero's Journey?\"\n}""",
    },
    "Dan Harmon's Story Circle": {
        "filter_tag": "harmon",
        "prompt_suffix": """Analyze the narrative through Dan Harmon's 8-step Story Circle. Focus on 'The Take' -- the price paid for the journey.\n\nThe analysis object must be:\n{\n  \"circle_stages\": {\n    \"you\": \"Character in comfort zone\",\n    \"need\": \"What they want/need\",\n    \"go\": \"Entering unfamiliar territory\",\n    \"search\": \"Adapting to the new situation\",\n    \"find\": \"Getting what they wanted\",\n    \"take\": \"The price paid\",\n    \"return\": \"Going back to familiar territory\",\n    \"change\": \"How they have changed\"\n  },\n  \"the_take\": \"The specific price paid -- what was lost or sacrificed\"\n}""",
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
        "prompt_suffix": """Identify the Jungian archetypes present in the narrative.\n\nThe analysis object must be:\n{\n  \"persona\": \"The public mask or social role characters present\",\n  \"shadow\": \"The repressed, dark, or denied aspects\",\n  \"anima_animus\": \"The contrasexual inner figure\",\n  \"trickster\": \"The agent of chaos, boundary-crossing, or transformation\"\n}""",
    },
    "Genette's Transtextuality": {
        "filter_tag": "transtextuality",
        "prompt_suffix": """Analyze the transtextual relationships -- how this text relates to other texts.\n\nThe analysis object must be:\n{\n  \"intertextuality\": \"Direct quotations, allusions, or references to other works\",\n  \"paratextuality\": \"How titles, epigraphs, prefaces, or cover art frame meaning\",\n  \"metatextuality\": \"How the text comments on or critiques other texts or its own genre\"\n}""",
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
        "the_take": "str",
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

## THEMATIC TENSIONS
2-4 central binary oppositions driving the narrative.

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

EMBED_SEMAPHORE = asyncio.Semaphore(EMBED_CONCURRENCY)
CHUNK_SEMAPHORE = asyncio.Semaphore(CHUNK_CONCURRENCY)
SYNTHESIS_SEMAPHORE = asyncio.Semaphore(SYNTHESIS_CONCURRENCY)


@dataclass(frozen=True)
class ArtifactPaths:
    events_path: str
    synthesis_path: str
    report_path: str


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
    )


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


def build_outline_context(global_outline: str, max_chars: int = MAX_OUTLINE_CONTEXT_CHARS) -> str:
    sections = []
    for heading in ["## DRAMATIS PERSONAE", "## NARRATIVE STRUCTURE", "## PLOT OUTLINE"]:
        match = re.search(rf"({re.escape(heading)}.*?)(?=\n## |\Z)", global_outline, re.DOTALL)
        if match:
            sections.append(match.group(1).strip())
    context = "\n\n".join(sections).strip() or global_outline.strip()
    return context[:max_chars]


def normalize_signal(signal: str) -> str:
    return signal.strip().lower()


def validate_signal(signal: str) -> str:
    normalized = normalize_signal(signal)
    if normalized not in ALLOWED_SIGNALS:
        raise ValueError(f"Unknown framework signal: {signal}")
    return normalized


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
    if not isinstance(snippet_ids, list) or not snippet_ids:
        raise ValueError("Event missing snippet_ids")
    if not set(snippet_ids).issubset(allowed_snippet_ids):
        raise ValueError("Event references unknown snippet_ids")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type}")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Event missing summary")
    if not isinstance(framework_signals, list):
        raise ValueError("framework_signals must be a list")

    validated_signals = ordered_dedupe([validate_signal(signal) for signal in framework_signals])
    return {
        "event_id": event_id.strip(),
        "chunk_id": chunk_id.strip(),
        "snippet_ids": ordered_dedupe([snippet_id for snippet_id in snippet_ids if isinstance(snippet_id, str)]),
        "type": event_type,
        "framework_signals": validated_signals,
        "summary": summary.strip(),
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
        events.append(validate_event(normalized, allowed_snippet_ids))

    if len(events) < MIN_EVENTS_PER_CHUNK:
        raise ValueError(f"Expected at least {MIN_EVENTS_PER_CHUNK} events for {chunk_id}, got {len(events)}")
    return events


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
    return {
        "analysis": analysis,
        "evidence_event_ids": ordered_dedupe(evidence_event_ids),
    }


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
    return events, True


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


async def generate_global_outline(text: str) -> str:
    t0 = time.time()
    print("\n--- Phase 1: Generating Global Outline (stratified, 30 snippets) ---")
    snippets = select_snippets_stratified(text)
    coverage = sum(len(snippet) for _, snippet in snippets)
    print(f"  Snippets: {len(snippets)}, coverage: {coverage:,} chars ({coverage / len(text) * 100:.1f}%)")

    snippets_text = "\n\n---\n\n".join(
        f"[~{pct:.0f}%] {snippet_text}" for pct, snippet_text in snippets
    )
    outline = await run_gemini_cli(OUTLINE_PROMPT.format(n=len(snippets), snippets_text=snippets_text))
    if not outline.strip():
        raise RuntimeError("Global outline generation returned empty output")
    print(f"  Outline generated in {time.time() - t0:.2f}s ({len(outline):,} chars)")
    return outline


async def extract_dynamic_keywords(outline: str) -> list[str]:
    t0 = time.time()
    print("\n--- Extracting dynamic keywords from outline ---")
    result = await run_gemini_cli(KEYWORD_EXTRACTION_PROMPT.format(outline=outline))
    parsed = extract_first_json_value(result)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        print("  Warning: Could not parse dynamic keywords, falling back to static")
        return STATIC_KEYWORDS
    keywords = combine_keywords(parsed)
    print(f"  Extracted {len(keywords)} keywords in {time.time() - t0:.2f}s")
    return keywords


def chunk_cache_base(book_file: str, chunk_id: str, micro_chunks: list[str]) -> str:
    digest = hashlib.sha256("".join(micro_chunks).encode("utf-8")).hexdigest()[:12]
    return f"{book_file}.{NTSMR_VERSION}.{MICRO_CHUNK_SIZE}.{chunk_id}.{digest}"


def load_or_build_index(book_file: str, chunk_id: str, micro_chunks: list[str]):
    cache_base = chunk_cache_base(book_file, chunk_id, micro_chunks)
    faiss_path = f"{cache_base}.faiss"
    chunks_path = f"{cache_base}.chunks.json"
    if os.path.exists(faiss_path) and os.path.exists(chunks_path):
        index = faiss.read_index(faiss_path)
        with open(chunks_path, "r", encoding="utf-8") as handle:
            cached_chunks = json.load(handle)
        return index, cached_chunks, faiss_path
    return None, None, faiss_path


async def build_faiss_index(book_file: str, chunk_id: str, micro_chunks: list[str]):
    index, cached_chunks, faiss_path = load_or_build_index(book_file, chunk_id, micro_chunks)
    if index is not None and cached_chunks is not None:
        return index, cached_chunks, faiss_path, True

    embeddings = await embed_many(micro_chunks)
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
    with open(chunks_path, "w", encoding="utf-8") as handle:
        json.dump(list(valid_micro_chunks), handle)
    return index, list(valid_micro_chunks), faiss_path, False


def select_snippets_for_chunk(
    chunk_id: str,
    valid_micro_chunks: list[str],
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
        snippets.append(
            {
                "snippet_id": f"{chunk_id}-s{ordinal}",
                "source_index": idx,
                "score": round(best_by_index[idx], 6),
                "text": valid_micro_chunks[idx],
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
    output = await run_gemini_cli(prompt)
    try:
        return parse_extraction_output(output, chunk_id, allowed_snippet_ids)
    except Exception as exc:
        repair_prompt = prompt + f"\n\nYour previous output failed validation: {exc}. Re-output corrected JSONL only."
        repair_output = await run_gemini_cli(repair_prompt, retries=1)
        return parse_extraction_output(repair_output, chunk_id, allowed_snippet_ids)


async def process_chunk(
    chunk_idx: int,
    micro_chunks: list[str],
    keyword_vectors: list[tuple[str, list[float]]],
    outline_context: str,
    book_file: str,
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
        events = await extract_events_for_chunk(chunk_id, snippets, outline_context)
        print(f"[{chunk_id}] Extracted {len(events)} validated events")
        return {
            "chunk_id": chunk_id,
            "cache_path": faiss_path,
            "snippet_count": len(snippets),
            "snippets": snippets,
            "event_count": len(events),
            "events": events,
            "duration_seconds": round(time.time() - chunk_start, 2),
        }


async def synthesize_single_framework(
    framework_name: str,
    framework_config: dict[str, Any],
    events: list[dict[str, Any]],
    outline_context: str,
) -> tuple[str, dict[str, Any]]:
    async with SYNTHESIS_SEMAPHORE:
        selected_events, used_fallback = filter_events_for_framework(events, framework_config.get("filter_tag"))
        event_jsonl = "\n".join(json.dumps(event, ensure_ascii=False) for event in selected_events)
        allowed_event_ids = {event["event_id"] for event in selected_events}
        prompt = f"""You are a Synthesis Sub-Agent specializing in {framework_name}.

## CHARACTER REFERENCE
{outline_context}

## STRUCTURAL EVENT TIMELINE
{event_jsonl}

Task:
Output a single JSON object with exactly these keys:
{{
  "analysis": <the framework analysis object>,
  "evidence_event_ids": ["event ids from the timeline above"]
}}

Rules:
- evidence_event_ids must contain 1-8 event IDs copied exactly from the timeline above.
- The analysis value must match this schema:
{framework_config['prompt_suffix']}
- Output JSON only. No markdown. No prose.
"""
        output = await run_gemini_cli(prompt)
        try:
            payload = extract_first_json_value(output)
            validated = validate_framework_result(framework_name, payload, allowed_event_ids)
        except Exception as exc:
            repair_prompt = prompt + f"\n\nYour previous output failed validation: {exc}. Re-output corrected JSON only."
            repair_output = await run_gemini_cli(repair_prompt, retries=1)
            payload = extract_first_json_value(repair_output)
            validated = validate_framework_result(framework_name, payload, allowed_event_ids)

        validated["used_full_timeline_fallback"] = used_fallback
        validated["event_count"] = len(selected_events)
        return framework_name, validated


async def synthesize_frameworks(
    events: list[dict[str, Any]],
    outline_context: str,
) -> dict[str, dict[str, Any]]:
    print(f"\n--- Phase 3: Synthesizing {len(FRAMEWORK_SYNTHESIS_PROMPTS)} frameworks ---")
    tasks = [
        synthesize_single_framework(framework_name, config, events, outline_context)
        for framework_name, config in FRAMEWORK_SYNTHESIS_PROMPTS.items()
    ]
    pairs = await asyncio.gather(*tasks)
    return {framework_name: payload for framework_name, payload in pairs}


def build_report(
    book_file: str,
    artifact_paths: ArtifactPaths,
    keywords: list[str],
    global_outline: str,
    outline_context: str,
    chunk_reports: list[dict[str, Any]],
    synthesis_payload: dict[str, dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    all_events = [event for chunk in chunk_reports for event in chunk["events"]]
    routed_frameworks = {
        framework_name: {
            "event_count": payload["event_count"],
            "used_full_timeline_fallback": payload["used_full_timeline_fallback"],
        }
        for framework_name, payload in synthesis_payload.items()
    }
    return to_json_safe({
        "ntsmr_version": NTSMR_VERSION,
        "book_file": book_file,
        "artifacts": artifact_paths.__dict__,
        "keywords": keywords,
        "global_outline": global_outline,
        "outline_context": outline_context,
        "chunk_count": len(chunk_reports),
        "event_count": len(all_events),
        "frameworks": routed_frameworks,
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "cache_path": chunk["cache_path"],
                "snippet_count": chunk["snippet_count"],
                "event_count": chunk["event_count"],
                "duration_seconds": chunk["duration_seconds"],
                "snippets": chunk["snippets"],
            }
            for chunk in chunk_reports
        ],
        "elapsed_seconds": round(elapsed_seconds, 2),
    })


async def main():
    parser = argparse.ArgumentParser(description="NTSMR Pipeline — Narrative Topology Semantic Map Reduce")
    parser.add_argument("book_file", help="Path to the raw text file")
    parser.add_argument("output_file", help="Path to save the events JSONL or artifact base path")
    args = parser.parse_args()

    start_time = time.time()
    with open(args.book_file, "r", encoding="utf-8") as handle:
        text = handle.read()

    macro_chunks = [
        text[index : index + MACRO_CHUNK_SIZE]
        for index in range(0, len(text), MACRO_CHUNK_SIZE)
    ]
    print(f"Loaded {args.book_file}: {len(text):,} characters, {len(macro_chunks)} macro-chunks.")

    global_outline = await generate_global_outline(text)
    outline_context = build_outline_context(global_outline)
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
                [macro_chunk[i : i + MICRO_CHUNK_SIZE] for i in range(0, len(macro_chunk), MICRO_CHUNK_SIZE)],
                keyword_vectors,
                outline_context,
                args.book_file,
            )
            for chunk_idx, macro_chunk in enumerate(macro_chunks)
        ]
    )
    all_events = [event for chunk in chunk_reports for event in chunk["events"]]
    print(f"Phase 2 complete: {len(all_events)} total validated events.")

    synthesis_payload = await synthesize_frameworks(all_events, outline_context)

    artifact_paths = resolve_artifact_paths(args.output_file)
    for path_str in [artifact_paths.events_path, artifact_paths.synthesis_path, artifact_paths.report_path]:
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    with open(artifact_paths.events_path, "w", encoding="utf-8") as handle:
        for event in all_events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    with open(artifact_paths.synthesis_path, "w", encoding="utf-8") as handle:
        json.dump(synthesis_payload, handle, indent=2, ensure_ascii=False)

    report_payload = build_report(
        args.book_file,
        artifact_paths,
        keywords,
        global_outline,
        outline_context,
        chunk_reports,
        synthesis_payload,
        time.time() - start_time,
    )
    with open(artifact_paths.report_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False)

    print(f"\nDone! Pipeline completed in {time.time() - start_time:.2f} seconds.")
    print(f"Events: {artifact_paths.events_path}")
    print(f"Synthesis: {artifact_paths.synthesis_path}")
    print(f"Report: {artifact_paths.report_path}")


if __name__ == "__main__":
    asyncio.run(main())

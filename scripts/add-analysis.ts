#!/usr/bin/env tsx
import { z } from "zod";
import * as fs from "fs";
import * as path from "path";

// --- Schema ---

const ScoresSchema = z.object({
  time_linearity: z.number().min(0).max(1),
  pacing_velocity: z.number().min(0).max(1),
  threat_scale: z.number().min(0).max(1),
  protagonist_fate: z.number().min(0).max(1),
  conflict_style: z.number().min(0).max(1),
  price_type: z.number().min(0).max(1),
});

const AddAnalysisSchema = z.object({
  title: z.string().describe("Title of the book"),
  author: z.string().describe("Author of the book"),
  year: z.number().describe("Year of publication"),
  quadrant_scores: ScoresSchema.describe(
    "Scores between 0.0 and 1.0 for the 3 visual quadrant charts",
  ),
  llm_memory_quadrant_scores: ScoresSchema.optional().describe(
    "Optional memory-based scores for A/B testing",
  ),
  minilm_rag_scores: ScoresSchema.optional(),
  nomic_rag_scores: ScoresSchema.optional(),
  hybrid_index_scores: ScoresSchema.optional(),
  sliding_window_scores: ScoresSchema.optional(),
  ntsmr_scores: ScoresSchema.optional(),
  medium: z.string().default("book").describe("Medium of the work"),
  genre_tags: z.array(z.string()).describe("Array of genre tags"),
  tropes: z.array(z.string()).describe("Array of tropes"),
  todorov_stages: z
    .object({
      equilibrium: z.string(),
      disruption: z.string(),
      recognition: z.string(),
      repair: z.string(),
      new_equilibrium: z.string(),
    })
    .describe("The 5 stages of Todorov's Equilibrium Model"),
  frameworks: z
    .object({
      protocol_fiction_mapping: z.object({
        rule: z.string(),
        failure_mode: z.string(),
        human_insight: z.string(),
      }),
      actantial_model: z.object({
        subject: z.string(),
        object: z.string(),
        sender: z.string(),
        receiver: z.string(),
        helper: z.string(),
        opponent: z.string(),
      }),
      freytag_pyramid: z.object({
        exposition: z.string(),
        climax: z.string(),
      }),
      propps_morphology: z.string().describe("Applicable narratemes"),
      genettes_narrative: z.object({
        order: z.string(),
        duration: z.string(),
        focalization: z.string(),
      }),
      monomyth: z.string().describe("Subversions of the hero's journey"),
      dan_harmon: z.string().describe("The Take (The Price Paid)"),
      save_the_cat: z.string().describe("Pacing deviations"),
      kishotenketsu: z.object({
        applicability: z.string(),
        ki: z.string().optional(),
        sho: z.string().optional(),
        ten: z.string().optional(),
        ketsu: z.string().optional(),
      }),
      three_act_structure: z.object({
        plot_points: z.string(),
      }),
    })
    .describe("Content for the markdown headers (11 frameworks)."),
});

// --- Helpers ---

function escapeYaml(s: string): string {
  return s.replace(/"/g, '\\"');
}

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^\w_]/g, "");
}

function renderScoresBlock(
  key: string,
  scores: z.infer<typeof ScoresSchema> | undefined,
): string {
  if (!scores) return "";
  return `${key}:
  time_linearity: ${scores.time_linearity}
  pacing_velocity: ${scores.pacing_velocity}
  threat_scale: ${scores.threat_scale}
  protagonist_fate: ${scores.protagonist_fate}
  conflict_style: ${scores.conflict_style}
  price_type: ${scores.price_type}`;
}

// --- Templating ---

function buildFile(data: z.infer<typeof AddAnalysisSchema>): string {
  const optionalScores = [
    renderScoresBlock(
      "llm_memory_quadrant_scores",
      data.llm_memory_quadrant_scores,
    ),
    renderScoresBlock("minilm_rag_scores", data.minilm_rag_scores),
    renderScoresBlock("nomic_rag_scores", data.nomic_rag_scores),
    renderScoresBlock("hybrid_index_scores", data.hybrid_index_scores),
    renderScoresBlock("sliding_window_scores", data.sliding_window_scores),
    renderScoresBlock("ntsmr_scores", data.ntsmr_scores),
  ]
    .filter(Boolean)
    .join("\n");

  const yaml = `---
title: "${escapeYaml(data.title)}"
author: "${escapeYaml(data.author)}"
year: ${data.year}
frameworks_mapped: ["Protocol Fiction Mapping", "Actantial Model", "Todorov's Equilibrium", "The Freytag Pyramid", "Propp's Morphology", "Genette's Narrative", "The Monomyth", "Dan Harmon", "Save the Cat", "Kishōtenketsu", "The Three-Act Structure"]
todorov_stages:
  equilibrium: "${escapeYaml(data.todorov_stages.equilibrium)}"
  disruption: "${escapeYaml(data.todorov_stages.disruption)}"
  recognition: "${escapeYaml(data.todorov_stages.recognition)}"
  repair: "${escapeYaml(data.todorov_stages.repair)}"
  new_equilibrium: "${escapeYaml(data.todorov_stages.new_equilibrium)}"
quadrant_scores:
  time_linearity: ${data.quadrant_scores.time_linearity}
  pacing_velocity: ${data.quadrant_scores.pacing_velocity}
  threat_scale: ${data.quadrant_scores.threat_scale}
  protagonist_fate: ${data.quadrant_scores.protagonist_fate}
  conflict_style: ${data.quadrant_scores.conflict_style}
  price_type: ${data.quadrant_scores.price_type}
${optionalScores}medium: "${data.medium}"
genre_tags: [${data.genre_tags.map((t) => `"${escapeYaml(t)}"`).join(", ")}]
tropes: [${data.tropes.map((t) => `"${escapeYaml(t)}"`).join(", ")}]
analysis_metadata:
  generated_by: "cli-agent"
  date: "${new Date().toISOString().split("T")[0]}"
  validation_status: "pending"
---`;

  const k = data.frameworks.kishotenketsu;
  const kishotenketsuDetails = [
    k.ki ? `- **Ki (Introduction):** ${k.ki}` : "",
    k.sho ? `- **Shō (Development):** ${k.sho}` : "",
    k.ten ? `- **Ten (Twist):** ${k.ten}` : "",
    k.ketsu ? `- **Ketsu (Resolution):** ${k.ketsu}` : "",
  ]
    .filter(Boolean)
    .join("\n");

  const markdown = `
# Structural Analysis

## 1. Protocol Fiction Mapping (Summer of Protocols)
- **Render a Rule:** ${data.frameworks.protocol_fiction_mapping.rule}
- **Rehearse a Failure Mode:** ${data.frameworks.protocol_fiction_mapping.failure_mode}
- **Reveal a Human Insight:** ${data.frameworks.protocol_fiction_mapping.human_insight}

## 2. Actantial Model (A.J. Greimas)
- **Subject:** ${data.frameworks.actantial_model.subject}
- **Object:** ${data.frameworks.actantial_model.object}
- **Sender (Destinator):** ${data.frameworks.actantial_model.sender}
- **Receiver (Destinatee):** ${data.frameworks.actantial_model.receiver}
- **Helper:** ${data.frameworks.actantial_model.helper}
- **Opponent:** ${data.frameworks.actantial_model.opponent}

## 3. Todorov's Equilibrium Model
- *See YAML Frontmatter for stage breakdown.*

## 4. The Freytag Pyramid
- **Exposition:** ${data.frameworks.freytag_pyramid.exposition}
- **Climax:** ${data.frameworks.freytag_pyramid.climax}

## 5. Propp's Morphology of the Folktale
- **Applicable Narratemes:** ${data.frameworks.propps_morphology}

## 6. Genette's Narrative Discourse
- **Order:** ${data.frameworks.genettes_narrative.order}
- **Duration:** ${data.frameworks.genettes_narrative.duration}
- **Focalization:** ${data.frameworks.genettes_narrative.focalization}

## 7. The Monomyth / Hero's Journey
- **Subversions:** ${data.frameworks.monomyth}

## 8. Dan Harmon's Story Circle
- **The Take (The Price Paid):** ${data.frameworks.dan_harmon}

## 9. Save the Cat! Beat Sheet
- **Pacing Deviations:** ${data.frameworks.save_the_cat}

## 10. Kishōtenketsu (Four-Act Structure)
- **Applicability:** ${data.frameworks.kishotenketsu.applicability}
${kishotenketsuDetails ? kishotenketsuDetails + "\n" : ""}
## 11. The Three-Act Structure
- **Plot Points:** ${data.frameworks.three_act_structure.plot_points}`;

  return yaml + "\n" + markdown.trim() + "\n";
}

// --- Validation ---

const REQUIRED_KEYS = [
  "title",
  "author",
  "year",
  "quadrant_scores",
  "frameworks_mapped",
  "medium",
  "genre_tags",
];

function validateWrittenFile(filePath: string): void {
  const written = fs.readFileSync(filePath, "utf-8");
  const fmMatch = written.match(/^---\n([\s\S]*?)\n---/);
  if (!fmMatch) {
    fs.unlinkSync(filePath);
    throw new Error(
      "Validation failed: written file has no valid YAML frontmatter. File deleted.",
    );
  }

  const rawYaml = fmMatch[1];
  const missingKeys = REQUIRED_KEYS.filter((k) => !rawYaml.includes(`${k}:`));
  if (missingKeys.length > 0) {
    fs.unlinkSync(filePath);
    throw new Error(
      `Validation failed: missing keys [${missingKeys.join(", ")}]. File deleted.`,
    );
  }
}

// --- Main ---

async function main() {
  // Read JSON from stdin
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  const input = Buffer.concat(chunks).toString("utf-8").trim();

  if (!input) {
    console.error("Usage: echo '<json>' | tsx scripts/add-analysis.ts");
    console.error("       tsx scripts/add-analysis.ts < input.json");
    process.exit(1);
  }

  let raw: unknown;
  try {
    raw = JSON.parse(input);
  } catch {
    console.error("Error: invalid JSON input");
    process.exit(1);
  }

  // Validate with Zod
  const result = AddAnalysisSchema.safeParse(raw);
  if (!result.success) {
    console.error("Validation error:");
    for (const issue of result.error.issues) {
      console.error(`  ${issue.path.join(".")}: ${issue.message}`);
    }
    process.exit(1);
  }

  const data = result.data;
  const slug = `${slugify(data.author)}_${slugify(data.title)}`;
  const filePath = path.resolve(__dirname, `../../data/${slug}.md`);

  if (fs.existsSync(filePath)) {
    console.error(
      `Error: ${filePath} already exists. Delete it first to overwrite.`,
    );
    process.exit(1);
  }

  // Write file
  const content = buildFile(data);
  fs.writeFileSync(filePath, content, "utf-8");

  // Validate
  validateWrittenFile(filePath);

  console.log(`data/${slug}.md`);
}

main().catch((err) => {
  console.error("Error:", err.message);
  process.exit(1);
});

#!/usr/bin/env tsx
/**
 * Deterministic body re-render from committed synthesis artifacts
 * (Nartopo-f38 backfill, zero LLM calls).
 *
 * Reads a TSV of `slug<TAB>artifact-run-label[<TAB>lenient]` (produced by
 * the classifier audit in the Nartopo repo), loads each work's
 * artifacts/<slug>.md.<label>.synthesis.json, renders the markdown body
 * with the exact section template add-analysis.ts uses, and splices it
 * under the work's existing YAML frontmatter — which is preserved
 * byte-for-byte (the synthesis holds raw run scores; the frontmatter holds
 * the corrected, site-label-space values and must not regress).
 *
 * Strict by design: a work with a missing block or field is skipped and
 * reported, not rendered with blanks — it stays on the site's honest
 * "analysis pending" note.
 *
 * Usage: npx tsx scripts/render-bodies-from-synthesis.ts <stubs.tsv>
 */
import fs from "fs";
import path from "path";

const TOOLS_ROOT = path.resolve(__dirname, "..");
const ARTIFACTS = path.join(TOOLS_ROOT, "artifacts");
const DATA_DIR = path.resolve(TOOLS_ROOT, "..", "data");

type Block = Record<string, unknown>;
type Synthesis = Record<string, { analysis?: Block } | undefined>;

class FieldError extends Error {}

/** Fetch a required string field; collapse internal newlines. */
function field(syn: Synthesis, block: string, key: string): string {
  const a = syn[block]?.analysis;
  if (!a) throw new FieldError(`missing block "${block}"`);
  const v = a[key];
  if (typeof v === "string" && v.trim()) {
    return v.replace(/\s*\n\s*/g, " ").trim();
  }
  if (Array.isArray(v) && v.length) {
    return v.map((x) => String(x).replace(/\s*\n\s*/g, " ").trim()).join("; ");
  }
  throw new FieldError(`missing/empty field "${block}.${key}"`);
}

function optionalField(syn: Synthesis, block: string, key: string): string {
  try {
    return field(syn, block, key);
  } catch {
    return "";
  }
}

/** The Take: prefer an explicit the_take, else the circle's take stage. */
function danHarmonTake(syn: Synthesis): string {
  const a = syn["Dan Harmon's Story Circle"]?.analysis;
  if (!a) throw new FieldError(`missing block "Dan Harmon's Story Circle"`);
  const explicit = a["the_take"];
  if (typeof explicit === "string" && explicit.trim()) {
    return explicit.replace(/\s*\n\s*/g, " ").trim();
  }
  const stages = a["circle_stages"];
  if (stages && typeof stages === "object" && !Array.isArray(stages)) {
    const take = (stages as Block)["take"];
    if (typeof take === "string" && take.trim()) {
      return take.replace(/\s*\n\s*/g, " ").trim();
    }
  }
  throw new FieldError(`no the_take or circle_stages.take`);
}

/** Mirrors the body template in nartopo-tools/scripts/add-analysis.ts. */
function renderBody(syn: Synthesis, lenient = false): string {
  const ki = optionalField(syn, "Kishotenketsu", "ki");
  const sho = optionalField(syn, "Kishotenketsu", "sho");
  const ten = optionalField(syn, "Kishotenketsu", "ten");
  const ketsu = optionalField(syn, "Kishotenketsu", "ketsu");
  const kishotenketsuDetails = [
    ki ? `- **Ki (Introduction):** ${ki}` : "",
    sho ? `- **Shō (Development):** ${sho}` : "",
    ten ? `- **Ten (Twist):** ${ten}` : "",
    ketsu ? `- **Ketsu (Resolution):** ${ketsu}` : "",
  ]
    .filter(Boolean)
    .join("\n");

  const plotPoints = `PP1: ${field(syn, "The Three-Act Structure", "plot_point_1")} PP2: ${field(syn, "The Three-Act Structure", "plot_point_2")}`;

  // Lenient mode (opt-in per work via TSV flag): a synthesis missing the
  // Lévi-Strauss block renders an honest pending note for that one section
  // instead of skipping the whole work. Every other block stays strict.
  const leviStrauss = (() => {
    try {
      return [
        `- **Primary Binary:** ${field(syn, "Levi-Strauss's Binary Oppositions", "primary_binary")}`,
        `- **Secondary Binary:** ${field(syn, "Levi-Strauss's Binary Oppositions", "secondary_binary")}`,
        `- **The Mediator:** ${field(syn, "Levi-Strauss's Binary Oppositions", "mediator")}`,
      ].join("\n");
    } catch (e) {
      if (lenient && e instanceof FieldError) {
        return "- *Analysis pending for this framework.*";
      }
      throw e;
    }
  })();

  const markdown = `
# Structural Analysis

## 1. Protocol Fiction Mapping (Summer of Protocols)
- **Render a Rule:** ${field(syn, "Protocol Fiction Mapping", "rule")}
- **Rehearse a Failure Mode:** ${field(syn, "Protocol Fiction Mapping", "failure_mode")}
- **Reveal a Human Insight:** ${field(syn, "Protocol Fiction Mapping", "human_insight")}

## 2. Actantial Model (A.J. Greimas)
- **Subject:** ${field(syn, "Actantial Model", "subject")}
- **Object:** ${field(syn, "Actantial Model", "object")}
- **Sender (Destinator):** ${field(syn, "Actantial Model", "sender")}
- **Receiver (Destinatee):** ${field(syn, "Actantial Model", "receiver")}
- **Helper:** ${field(syn, "Actantial Model", "helper")}
- **Opponent:** ${field(syn, "Actantial Model", "opponent")}

## 3. Todorov's Equilibrium Model
- *See YAML Frontmatter for stage breakdown.*

## 4. The Freytag Pyramid
- **Exposition:** ${field(syn, "The Freytag Pyramid", "exposition")}
- **Climax:** ${field(syn, "The Freytag Pyramid", "climax")}

## 5. Propp's Morphology of the Folktale
- **Applicable Narratemes:** ${field(syn, "Propp's Morphology", "applicable_narratemes")}

## 6. Genette's Narrative Discourse
- **Order:** ${field(syn, "Genette's Narrative Discourse", "order")}
- **Duration:** ${field(syn, "Genette's Narrative Discourse", "duration")}
- **Focalization:** ${field(syn, "Genette's Narrative Discourse", "focalization")}

## 7. The Monomyth / Hero's Journey
- **Subversions:** ${field(syn, "The Monomyth", "subversions")}

## 8. Dan Harmon's Story Circle
- **The Take (The Price Paid):** ${danHarmonTake(syn)}

## 9. Save the Cat! Beat Sheet
- **Pacing Deviations:** ${field(syn, "Save the Cat! Beat Sheet", "pacing_deviations")}

## 10. Kishōtenketsu (Four-Act Structure)
- **Applicability:** ${field(syn, "Kishotenketsu", "applicability")}
${kishotenketsuDetails ? kishotenketsuDetails + "\n" : ""}
## 11. The Three-Act Structure
- **Plot Points:** ${plotPoints}

## 12. Lévi-Strauss's Binary Oppositions
${leviStrauss}

## 13. Cognitive Estrangement (Suvin / Shklovsky)
- **The Familiar Concept:** ${field(syn, "Cognitive Estrangement", "familiar_concept")}
- **The Estranging Mechanism:** ${field(syn, "Cognitive Estrangement", "estranging_mechanism")}
- **The Cognitive Shift:** ${field(syn, "Cognitive Estrangement", "cognitive_shift")}

## 14. Bakhtin's Chronotope
- **The Spatial Matrix:** ${field(syn, "Bakhtin's Chronotope", "spatial_matrix")}
- **The Temporal Flow:** ${field(syn, "Bakhtin's Chronotope", "temporal_flow")}
- **The Point of Intersection:** ${field(syn, "Bakhtin's Chronotope", "intersection")}

## 15. Aristotelian Poetics
- **Hamartia:** ${field(syn, "Aristotelian Poetics", "hamartia")}
- **Peripeteia:** ${field(syn, "Aristotelian Poetics", "peripeteia")}
- **Anagnorisis:** ${field(syn, "Aristotelian Poetics", "anagnorisis")}

## 16. Jungian Archetypal Analysis
- **The Persona:** ${field(syn, "Jungian Archetypal Analysis", "persona")}
- **The Shadow:** ${field(syn, "Jungian Archetypal Analysis", "shadow")}
- **The Anima/Animus:** ${field(syn, "Jungian Archetypal Analysis", "anima_animus")}
- **The Trickster:** ${field(syn, "Jungian Archetypal Analysis", "trickster")}

## 17. Genette's Transtextuality
- **Intertextuality:** ${field(syn, "Genette's Transtextuality", "intertextuality")}
- **Paratextuality:** ${field(syn, "Genette's Transtextuality", "paratextuality")}
- **Metatextuality:** ${field(syn, "Genette's Transtextuality", "metatextuality")}`;

  return markdown.trim() + "\n";
}

/** Split a data file into (frontmatter incl. both --- fences, rest). */
function splitFrontmatter(raw: string): [string, string] {
  const m = raw.match(/^(---\n[\s\S]*?\n---\n)/);
  if (!m) throw new Error("no YAML frontmatter found");
  return [m[1], raw.slice(m[1].length)];
}

const tsvPath = process.argv[2];
if (!tsvPath) {
  console.error("usage: render-bodies-from-synthesis.ts <stubs.tsv>");
  process.exit(2);
}

const rows = fs
  .readFileSync(tsvPath, "utf8")
  .trim()
  .split("\n")
  .map((l) => l.split("\t") as [string, string, string?]);

let rendered = 0;
const skipped: string[] = [];
for (const [slug, label, flag] of rows) {
  const artifactPath = path.join(ARTIFACTS, `${slug}.md.${label}.synthesis.json`);
  const dataPath = path.join(DATA_DIR, `${slug}.md`);
  try {
    const syn = JSON.parse(fs.readFileSync(artifactPath, "utf8")) as Synthesis;
    const body = renderBody(syn, flag === "lenient");
    const raw = fs.readFileSync(dataPath, "utf8");
    const [frontmatter] = splitFrontmatter(raw);
    fs.writeFileSync(dataPath, frontmatter + "\n" + body);
    // parse-back: frontmatter unchanged byte-for-byte, body present
    const [fm2, body2] = splitFrontmatter(fs.readFileSync(dataPath, "utf8"));
    if (fm2 !== frontmatter) throw new Error("frontmatter changed on write");
    if (!body2.includes("# Structural Analysis")) throw new Error("body missing heading");
    rendered++;
  } catch (e) {
    skipped.push(`${slug}: ${(e as Error).message}`);
  }
}

console.log(`rendered: ${rendered} | skipped: ${skipped.length}`);
skipped.forEach((s) => console.log("  SKIP " + s));
process.exit(skipped.length ? 1 : 0);

#!/usr/bin/env tsx
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import * as fs from "fs";
import * as path from "path";

const server = new Server(
  {
    name: "nartopo-mcp",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  },
);

// Define the schema for the add_analysis tool
const AddAnalysisSchema = z.object({
  title: z.string().describe("Title of the book"),
  author: z.string().describe("Author of the book"),
  year: z.number().describe("Year of publication"),
  quadrant_scores: z
    .object({
      time_linearity: z.number().min(0).max(1),
      pacing_velocity: z.number().min(0).max(1),
      threat_scale: z.number().min(0).max(1),
      protagonist_fate: z.number().min(0).max(1),
      conflict_style: z.number().min(0).max(1),
      price_type: z.number().min(0).max(1),
    })
    .describe("Scores between 0.0 and 1.0 for the 3 visual quadrant charts"),
  llm_memory_quadrant_scores: z
    .object({
      time_linearity: z.number().min(0).max(1),
      pacing_velocity: z.number().min(0).max(1),
      threat_scale: z.number().min(0).max(1),
      protagonist_fate: z.number().min(0).max(1),
      conflict_style: z.number().min(0).max(1),
      price_type: z.number().min(0).max(1),
    })
    .optional()
    .describe("Optional memory-based scores for A/B testing"),
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
    .describe(
      "Content for the markdown headers. Do NOT include Corporate Vampire Arc.",
    ),
});

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "add_analysis",
        description:
          "Add a complete structural analysis for a new book to the Nartopo database. Strictly enforces schema, calculates the slug, and writes the correct YAML/Markdown file.",
        inputSchema: zodToJsonSchema(AddAnalysisSchema),
      },
    ],
  };
});

// Helper to convert Zod schema to JSON schema for MCP
function zodToJsonSchema(schema: z.ZodType<any, any, any>) {
  // simplified implementation for standard types
  return {
    type: "object",
    properties: {
      title: { type: "string", description: "Title of the book" },
      author: { type: "string", description: "Author of the book" },
      year: { type: "number", description: "Year of publication" },
      quadrant_scores: {
        type: "object",
        properties: {
          time_linearity: { type: "number", minimum: 0, maximum: 1 },
          pacing_velocity: { type: "number", minimum: 0, maximum: 1 },
          threat_scale: { type: "number", minimum: 0, maximum: 1 },
          protagonist_fate: { type: "number", minimum: 0, maximum: 1 },
          conflict_style: { type: "number", minimum: 0, maximum: 1 },
          price_type: { type: "number", minimum: 0, maximum: 1 },
        },
        required: [
          "time_linearity",
          "pacing_velocity",
          "threat_scale",
          "protagonist_fate",
          "conflict_style",
          "price_type",
        ],
      },
      llm_memory_quadrant_scores: {
        type: "object",
        properties: {
          time_linearity: { type: "number", minimum: 0, maximum: 1 },
          pacing_velocity: { type: "number", minimum: 0, maximum: 1 },
          threat_scale: { type: "number", minimum: 0, maximum: 1 },
          protagonist_fate: { type: "number", minimum: 0, maximum: 1 },
          conflict_style: { type: "number", minimum: 0, maximum: 1 },
          price_type: { type: "number", minimum: 0, maximum: 1 },
        },
      },
      medium: { type: "string" },
      genre_tags: { type: "array", items: { type: "string" } },
      tropes: { type: "array", items: { type: "string" } },
      todorov_stages: {
        type: "object",
        properties: {
          equilibrium: { type: "string" },
          disruption: { type: "string" },
          recognition: { type: "string" },
          repair: { type: "string" },
          new_equilibrium: { type: "string" },
        },
        required: [
          "equilibrium",
          "disruption",
          "recognition",
          "repair",
          "new_equilibrium",
        ],
      },
      frameworks: {
        type: "object",
        properties: {
          protocol_fiction_mapping: {
            type: "object",
            properties: {
              rule: { type: "string" },
              failure_mode: { type: "string" },
              human_insight: { type: "string" },
            },
            required: ["rule", "failure_mode", "human_insight"],
          },
          actantial_model: {
            type: "object",
            properties: {
              subject: { type: "string" },
              object: { type: "string" },
              sender: { type: "string" },
              receiver: { type: "string" },
              helper: { type: "string" },
              opponent: { type: "string" },
            },
            required: [
              "subject",
              "object",
              "sender",
              "receiver",
              "helper",
              "opponent",
            ],
          },
          freytag_pyramid: {
            type: "object",
            properties: {
              exposition: { type: "string" },
              climax: { type: "string" },
            },
            required: ["exposition", "climax"],
          },
          propps_morphology: { type: "string" },
          genettes_narrative: {
            type: "object",
            properties: {
              order: { type: "string" },
              duration: { type: "string" },
              focalization: { type: "string" },
            },
            required: ["order", "duration", "focalization"],
          },
          monomyth: { type: "string" },
          dan_harmon: { type: "string" },
          save_the_cat: { type: "string" },
          kishotenketsu: {
            type: "object",
            properties: {
              applicability: { type: "string" },
              ki: { type: "string" },
              sho: { type: "string" },
              ten: { type: "string" },
              ketsu: { type: "string" },
            },
            required: ["applicability"],
          },
          three_act_structure: {
            type: "object",
            properties: { plot_points: { type: "string" } },
            required: ["plot_points"],
          },
        },
        required: [
          "protocol_fiction_mapping",
          "actantial_model",
          "freytag_pyramid",
          "propps_morphology",
          "genettes_narrative",
          "monomyth",
          "dan_harmon",
          "save_the_cat",
          "kishotenketsu",
          "three_act_structure",
        ],
      },
    },
    required: [
      "title",
      "author",
      "year",
      "quadrant_scores",
      "medium",
      "genre_tags",
      "tropes",
      "todorov_stages",
      "frameworks",
    ],
  };
}

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "add_analysis") {
    throw new McpError(
      ErrorCode.MethodNotFound,
      `Unknown tool: ${request.params.name}`,
    );
  }

  try {
    const data = AddAnalysisSchema.parse(request.params.arguments);

    const slugAuthor = data.author
      .toLowerCase()
      .replace(/\s+/g, "_")
      .replace(/[^\w_]/g, "");
    const slugTitle = data.title
      .toLowerCase()
      .replace(/\s+/g, "_")
      .replace(/[^\w_]/g, "");
    const slug = `${slugAuthor}_${slugTitle}`;

    const filePath = path.resolve(__dirname, `../data/${slug}.md`);

    // Serialize YAML
    const yaml = `---
title: "${data.title.replace(/"/g, '"')}"
author: "${data.author.replace(/"/g, '"')}"
year: ${data.year}
frameworks_mapped: ["Protocol Fiction Mapping", "Actantial Model", "Todorov's Equilibrium", "The Freytag Pyramid", "Propp's Morphology", "Genette’s Narrative", "The Monomyth", "Dan Harmon", "Save the Cat", "Kishōtenketsu", "The Three-Act Structure"]
todorov_stages:
  equilibrium: "${data.todorov_stages.equilibrium.replace(/"/g, '"')}"
  disruption: "${data.todorov_stages.disruption.replace(/"/g, '"')}"
  recognition: "${data.todorov_stages.recognition.replace(/"/g, '"')}"
  repair: "${data.todorov_stages.repair.replace(/"/g, '"')}"
  new_equilibrium: "${data.todorov_stages.new_equilibrium.replace(/"/g, '"')}"
quadrant_scores:
  time_linearity: ${data.quadrant_scores.time_linearity}
  pacing_velocity: ${data.quadrant_scores.pacing_velocity}
  threat_scale: ${data.quadrant_scores.threat_scale}
  protagonist_fate: ${data.quadrant_scores.protagonist_fate}
  conflict_style: ${data.quadrant_scores.conflict_style}
  price_type: ${data.quadrant_scores.price_type}
${data.llm_memory_quadrant_scores ? `llm_memory_quadrant_scores:
  time_linearity: ${data.llm_memory_quadrant_scores.time_linearity}
  pacing_velocity: ${data.llm_memory_quadrant_scores.pacing_velocity}
  threat_scale: ${data.llm_memory_quadrant_scores.threat_scale}
  protagonist_fate: ${data.llm_memory_quadrant_scores.protagonist_fate}
  conflict_style: ${data.llm_memory_quadrant_scores.conflict_style}
  price_type: ${data.llm_memory_quadrant_scores.price_type}` : ""}
medium: "${data.medium}"
genre_tags: [${data.genre_tags.map((t) => `"${t.replace(/"/g, '"')}"`).join(", ")}]
tropes: [${data.tropes.map((t) => `"${t.replace(/"/g, '"')}"`).join(", ")}]
analysis_metadata:
  generated_by: "mcp-agent"
  date: "${new Date().toISOString().split("T")[0]}"
  validation_status: "approved"
---`;

    // Serialize Markdown Body
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

## 6. Genette’s Narrative Discourse
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
${
  data.frameworks.kishotenketsu.ki
    ? `- **Ki (Introduction):** ${data.frameworks.kishotenketsu.ki}
`
    : ""
}${
      data.frameworks.kishotenketsu.sho
        ? `- **Shō (Development):** ${data.frameworks.kishotenketsu.sho}
`
        : ""
    }${
      data.frameworks.kishotenketsu.ten
        ? `- **Ten (Twist):** ${data.frameworks.kishotenketsu.ten}
`
        : ""
    }${
      data.frameworks.kishotenketsu.ketsu
        ? `- **Ketsu (Resolution):** ${data.frameworks.kishotenketsu.ketsu}
`
        : ""
    }
## 11. The Three-Act Structure
- **Plot Points:** ${data.frameworks.three_act_structure.plot_points}
`;

    fs.writeFileSync(filePath, yaml + "\n" + markdown.trim(), "utf-8");

    return {
      content: [
        {
          type: "text",
          text: `Analysis successfully written to data/${slug}.md`,
        },
      ],
    };
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      return {
        isError: true,
        content: [{ type: "text", text: `Validation error: ${error.message}` }],
      };
    }
    throw new McpError(ErrorCode.InternalError, error.message);
  }
});

async function run() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Nartopo MCP server running on stdio");
}

run().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});

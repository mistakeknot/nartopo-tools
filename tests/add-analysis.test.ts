import assert from "node:assert/strict";
import test from "node:test";

import { validateWrittenFileContent } from "../scripts/add-analysis";

const validContent = `---
title: "Example Book"
author: "Example Author"
year: 2024
frameworks_mapped: ["Protocol Fiction Mapping"]
todorov_stages:
  equilibrium: "A stable world"
  disruption: "A disruption occurs"
  recognition: "The protagonist notices"
  repair: "The protagonist responds"
  new_equilibrium: "A changed world"
quadrant_scores:
  time_linearity: 0.1
  pacing_velocity: 0.2
  threat_scale: 0.3
  protagonist_fate: 0.4
  conflict_style: 0.5
  price_type: 0.6
medium: "book"
genre_tags: ["science_fiction"]
tropes: ["first_contact"]
analysis_metadata:
  generated_by: "cli-agent"
  date: "2026-03-05"
  validation_status: "pending"
---
# Structural Analysis

## 1. Protocol Fiction Mapping (Summer of Protocols)
- **Render a Rule:** Example rule
`;

test("validateWrittenFileContent parses frontmatter structurally", () => {
  const parsed = validateWrittenFileContent(validContent);
  assert.equal(parsed.title, "Example Book");
  assert.equal(parsed.quadrant_scores.price_type, 0.6);
});

test("validateWrittenFileContent rejects malformed structural frontmatter", () => {
  const malformed = `---
title: "Example Book"
author: "Example Author"
year: nope
quadrant_scores: invalid
frameworks_mapped: ["Protocol Fiction Mapping"]
medium: "book"
genre_tags: ["science_fiction"]
---
# Structural Analysis
`;

  assert.throws(() => validateWrittenFileContent(malformed));
});

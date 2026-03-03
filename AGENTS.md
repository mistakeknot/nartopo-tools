# nartopo-tools: Agent Guide

## Overview

This repo contains the ingestion pipeline for the [Nartopo](https://nartopo.com) narrative topology engine. It is a companion to the main Nartopo repo, which holds the data files and the Next.js presentation layer.

## Directory Layout

```
Nartopo/                    # Parent repo (Next.js app + data)
├── data/                   # Analysis files (source of truth)
│   ├── _frameworks.md      # The 11-point framework reference
│   ├── _template.md        # Canonical analysis template
│   └── *.md                # Individual work analyses
├── src/                    # Next.js application
└── nartopo-tools/          # This repo (gitignored by parent)
    ├── scripts/
    │   ├── ingest.ts       # CLI: download + extract + scaffold
    │   └── mcp-server.ts   # MCP server: validated analysis writing
    └── package.json

~/projects/books/           # Downloaded EPUBs/text (separate, untracked)
├── {Author Name}/
│   ├── *.epub
│   └── *.txt
└── .env                    # ANNAS_ARCHIVE_API_KEY, ANNAS_ARCHIVE_DOMAIN
```

## The Ingestion Pipeline

### Core Rules
1. **Never write files to `../data/` manually.** Always use the Nartopo MCP server (`add_analysis`) to prevent schema drift and YAML/Markdown formatting errors.
2. **Do NOT commit raw text.** All EPUBs and extracted `.txt` files must stay in `../../books/`.
3. **Run build after adding.** Always run `npm run build` in the parent Nartopo repo to verify the generated Next.js application parses the new work correctly.

### Step 1: Automated Download and Extraction

```bash
npm run ingest -- --author "Author Name" --title "Book Title" --year YYYY
```

What this does:
- Triggers the local `annas-archive-mcp` to search for the book.
- Downloads the highest-rated EPUB directly to `../../books/Author Name/`.
- Unzips the EPUB and extracts all chapters to a raw `Book Title.txt` file.
- Generates a blank markdown template in `../data/` with the correct slug.

### Step 2A: Context Loading and Analysis (Standard)

The AI agent must read the extracted text file located at `../../books/{Author Name}/{Book Title}.txt` into its context window. Based on this raw text, the agent will analyze the book against the 11 frameworks (see `../data/_frameworks.md`) and score its six quadrant axes (Time Linearity, Pacing Velocity, Threat Scale, Protagonist Fate, Conflict Style, Price Type) on a 0.0–1.0 float scale.

### Step 2B: Map-Reduce Pipeline (For Exceptionally Large Books)

If a book text is too large to fit effectively into a single context window or you need to ensure rigorous, highly-localized structural extraction across a massive epic, use the `map_reduce.sh` script.

```bash
./scripts/map_reduce.sh "../../books/{Author Name}/{Book Title}.txt" "/tmp/findings.txt"
```

What this does:
- Splits the massive raw text file into ~150k character chunks.
- Spawns parallel, headless `gemini` CLI sub-agents for each chunk.
- Each sub-agent extracts dense structural data, plot events, and character shifts strictly from their assigned local text.
- Synthesizes all parallel findings into a single `findings.txt` log, which the primary agent then reads to synthesize the final 11-framework JSON.

### Step 3: Writing the Analysis via Nartopo MCP

The AI agent MUST NOT write to `../data/` directly. Instead, use the local Nartopo MCP server:

1. **Check Tools:**
   ```bash
   mcp tools npm run mcp
   ```
2. **Submit Analysis:**
   ```bash
   mcp call add_analysis --params '{ "title": "...", "author": "...", ... }' npm run mcp
   ```

The MCP server strictly validates the schema using Zod, templates the Markdown headers, and writes the file to `../data/`.

### Step 4: Verification and Deployment

1. Run `npm run build` in the parent Nartopo repo. This triggers `scripts/generate-works-json.ts` which reads the newly generated Markdown files and rebuilds the `works.json` database, followed by compiling the static Next.js pages.
2. Commit the new markdown file in the Nartopo repo: `git add data/{file}.md && git commit -m "data: add {Author} - {Title} structural analysis"`.

## Bypassing Cloudflare 403 Errors

If `npm run ingest` fails because the `annas-archive-mcp` returns a network error or Cloudflare 403 block, ensure the custom mirror domain is set:

```bash
# Check or set the working domain in ../../books/.env
echo 'ANNAS_ARCHIVE_DOMAIN=annas-archive.gl' >> ../../books/.env
```

The patched MCP client reads this environment variable and routes requests through the unblocked mirror.

## Legal and Ethical Scope

**CRITICAL MANDATE:** Nartopo is strictly an **academic, non-profit, comparative literature project**.

1. **Fair Use:** Textual extraction workflows are exclusively designed to generate transformative, statistical, and structural metadata.
2. **No Text Distribution:** Agents must **never** commit, publish, or publicly expose raw text or EPUB files. Only analytical metadata (YAML frontmatter and structural markdown) is permitted in the repository.
3. **Local Ephemerality:** Raw text and EPUBs live in `../../books/` (separate directory). The `.gitignore` blocks `*.epub`, `*.pdf`, `*.txt` as a safety net.

# nartopo-tools

Ingestion pipeline for the [Nartopo](https://nartopo.com) narrative topology engine. Downloads books, extracts text, and writes structured analyses via an MCP server.

## Setup

This repo lives inside the Nartopo directory as a gitignored subdirectory:

```
Nartopo/
├── data/               # Analysis files (source of truth)
├── src/                # Next.js app
└── nartopo-tools/      # This repo (separate git, .gitignore'd by parent)
    └── scripts/

~/projects/books/       # Downloaded EPUBs (separate, untracked)
```

```bash
cd /path/to/Nartopo/nartopo-tools
npm install
```

Requires a sibling `books/` directory at `~/projects/books/` with a `.env` containing `ANNAS_ARCHIVE_API_KEY`.

## Usage

### Ingest a new book

```bash
npm run ingest -- --author "Author Name" --title "Book Title" --year 2020
```

This searches Anna's Archive, downloads the EPUB, extracts text, and scaffolds an analysis template in `../data/`.

### MCP server for writing analyses

```bash
npm run mcp
```

Starts a stdio MCP server with an `add_analysis` tool that validates and writes structured analysis files to `../data/`.

## See Also

- `AGENTS.md` — Full ingestion workflow guide for AI agents
- `../data/_frameworks.md` — The 11-point framework reference
- `../data/_template.md` — Analysis template

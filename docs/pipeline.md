# Nartopo Ingestion Pipeline Runbook

This document serves as the master runbook for the end-to-end ingestion pipeline. It covers how a book goes from being an idea to becoming a structured, 11-framework analysis plotted on the interactive Nartopo website.

---

## 1. Automated Download and Extraction

Instead of manually searching for files or writing extraction scripts, use the automated Node CLI script to download and prep the book.

```bash
cd nartopo-tools
npm run ingest -- --author "Author Name" --title "Book Title" --year YYYY
```

**What this does:**
1. Triggers the local `annas-archive-mcp` to search for the book via its MD5 hash.
2. Downloads the highest-rated EPUB directly to `../../books/{Author Name}/`.
3. Unzips the EPUB and extracts all chapters to a raw `Book Title.txt` file.
4. Generates a blank markdown template in the main Nartopo repo (`../data/`) with the correct slug.

*Note: If `npm run ingest` fails due to a Cloudflare 403 block, ensure `ANNAS_ARCHIVE_DOMAIN=annas-archive.gl` is set in `../../books/.env`.*

---

## 2. Context Loading and Analysis

Nartopo requires a rigorous structural analysis across 11 specific frameworks. The method of analysis depends on the length of the extracted `.txt` file.

### Option A: Standard Pipeline (For shorter works)
If the book fits comfortably within the AI agent's standard context window:
1. The AI agent reads the full `../../books/{Author Name}/{Book Title}.txt` file directly into memory.
2. The agent synthesizes the 11-framework JSON payload and determines the 0.0–1.0 quadrant scores.

### Option B: Map-Reduce Pipeline (For exceptionally large books)
If a book text is too massive to fit into a single context window without degrading reasoning quality, use the parallel map-reduce script to orchestrate multiple Gemini sub-agents.

```bash
./scripts/map_reduce.sh "../../books/{Author Name}/{Book Title}.txt" "/tmp/findings.txt"
```

**What this does:**
1. Splits the massive raw text file into `~150k` character chunks (based on byte size) so each fits perfectly within optimal context bounds.
2. Spawns parallel, headless `gemini -y -p` CLI sub-agents for each chunk.
3. Each sub-agent extracts dense structural data, plot events, pacing metrics, and character shifts strictly from their assigned local text fragment.
4. Waits for all sub-agents to complete and synthesizes their parallel findings into a single `findings.txt` log.
5. The primary agent then reads the synthesized log to generate the final 11-framework JSON payload.

---

## 3. Writing the Analysis via Nartopo MCP

To prevent schema drift and YAML formatting errors, agents must **never** write files to the `../data/` folder directly. Instead, use the local Nartopo MCP server to construct the file.

1. **Start the MCP Server:**
   ```bash
   npm run mcp
   ```
2. **Submit Analysis:**
   Use the `add_analysis` tool via the MCP client, passing the full structured JSON analysis as arguments. 
   ```bash
   mcp call add_analysis --params '{ "title": "...", "author": "...", ... }' npm run mcp
   ```

The MCP server will strictly validate the schema using Zod, automatically calculate the file slug, template the Markdown headers correctly, and write the file securely to the parent Nartopo repository.

---

## 4. Verification and Deployment

Once the Markdown file has been safely injected into the database:

1. **Rebuild the JSON state:**
   Navigate back to the parent repository and run the Next.js build. This triggers `scripts/generate-works-json.ts`, which reads the newly generated Markdown file, parses its YAML, and regenerates the `works.json` database.
   ```bash
   cd ..
   npm run build
   ```
2. **Commit the data:**
   Stage and commit the new analysis file to the repository.
   ```bash
   git add data/{file}.md 
   git commit -m "data: add {Author} - {Title} structural analysis"
   ```
3. **Deploy:**
   Push the changes to the `main` branch. 
   ```bash
   git push
   ```
   Vercel will automatically detect the push, run the build script, and deploy the updated corpus—complete with the new interactive scatter plots and radar charts—live to `nartopo.com`.
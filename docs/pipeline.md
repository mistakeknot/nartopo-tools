# Map-Reduce Ingestion Pipeline

If a book text is too large to fit effectively into a single context window or you need to ensure rigorous, highly-localized structural extraction across a massive epic, use the `map_reduce.sh` script to orchestrate multiple Gemini sub-agents.

## Usage

```bash
./scripts/map_reduce.sh "../../books/{Author Name}/{Book Title}.txt" "/tmp/findings.txt"
```

## How It Works

1. **Partitioning:** The script splits the massive raw text file into `~150k` character chunks (based on byte size, `split -C 150k`) so each fits comfortably within standard optimal context bounds.
2. **Parallel Sub-Agents:** It spawns parallel, headless `gemini` CLI sub-agents for each chunk using the `gemini -y -p` command.
3. **Local Extraction:** Each sub-agent is prompted to act as a structural analysis expert, extracting dense structural data, plot events, pacing metrics, and character shifts strictly from their assigned local text fragment.
4. **Synthesis:** It waits for all sub-agents to complete and synthesizes all parallel findings into a single `findings.txt` log.

## Next Steps
Once the `findings.txt` log is generated, the primary agent should read that log into its context to synthesize the final 11-framework JSON and inject it using the `nartopo-mcp` server.
# 2B. Map-Reduce Pipeline

If a book text is too large to fit effectively into a single context window or you need to ensure rigorous, highly-localized structural extraction across a massive epic, use the parallel map-reduce script to orchestrate multiple Gemini sub-agents.

```bash
./scripts/map_reduce.sh "../../books/{Author Name}/{Book Title}.txt" "/tmp/findings.txt"
```

**What this does:**
1. Splits the massive raw text file into `~150k` character chunks (based on byte size) so each fits perfectly within optimal context bounds.
2. Spawns parallel, headless `gemini -y -p` CLI sub-agents for each chunk.
3. Each sub-agent extracts dense structural data, plot events, pacing metrics, and character shifts strictly from their assigned local text fragment.
4. Waits for all sub-agents to complete and synthesizes their parallel findings into a single `findings.txt` log.
5. The primary agent then reads the synthesized log to generate the final 11-framework JSON payload.

## Next Step
Once the `findings.txt` log is generated and read by the primary agent, proceed to [MCP Injection](03-mcp-injection.md).
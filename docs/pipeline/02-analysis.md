# 2. Context Loading and Analysis

Nartopo requires a rigorous structural analysis across 11 specific frameworks. The method of analysis depends on the length of the extracted `.txt` file.

### Option A: Standard Pipeline (For shorter works)
If the book fits comfortably within the AI agent's standard context window:
1. The AI agent reads the full `../../books/{Author Name}/{Book Title}.txt` file directly into memory.
2. The agent synthesizes the 11-framework JSON payload and determines the 0.0–1.0 quadrant scores.

### Option B: Map-Reduce Pipeline (For exceptionally large books)
If a book text is too massive to fit into a single context window without degrading reasoning quality, use the parallel map-reduce script to orchestrate multiple Gemini sub-agents.

See [Map-Reduce Pipeline](02b-map-reduce.md) for detailed execution instructions.

## Next Step
Once the structural synthesis is complete, proceed to [MCP Injection](03-mcp-injection.md).
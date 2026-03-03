# Nartopo Ingestion Pipeline Runbook

This directory serves as the master runbook for the end-to-end ingestion pipeline. It covers how a book goes from being an idea to becoming a structured, 11-framework analysis plotted on the interactive Nartopo website.

Please proceed through the steps sequentially:

1. [Automated Download and Extraction](01-download-and-extraction.md)
2. [Context Loading and Analysis](02-analysis.md)
   * *(Optional)* [Map-Reduce Pipeline for Massive Texts](02b-map-reduce.md)
3. [Writing the Analysis via Nartopo MCP](03-mcp-injection.md)
4. [Verification and Deployment](04-deployment.md)

---
*For optimizing context usage on massive texts, see [Pipeline Efficiency & Grand Unification](05-efficiency.md).*
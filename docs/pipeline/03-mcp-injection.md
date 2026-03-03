# 3. Writing the Analysis via Nartopo MCP

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

## Next Step
Once injected, proceed to [Verification and Deployment](04-deployment.md).
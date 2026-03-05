# 3. Writing the Analysis via CLI

To prevent schema drift and YAML formatting errors, agents must **never** write files to the `../data/` folder directly. Instead, pipe the analysis JSON through the `add-analysis` CLI.

```bash
echo '{ "title": "...", "author": "...", ... }' | npm run add-analysis
```

The CLI validates the input schema using Zod, calculates the file slug, templates the Markdown headers correctly, writes the file, and then performs a structural parse-back validation of the written frontmatter before returning. On success it prints the relative path (`data/{slug}.md`).

## Next Step
Once injected, proceed to [Verification and Deployment](04-deployment.md).

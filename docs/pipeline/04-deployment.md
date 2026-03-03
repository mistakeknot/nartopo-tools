# 4. Verification and Deployment

Once the Markdown file has been safely injected into the database via the MCP server:

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
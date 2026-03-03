#!/usr/bin/env tsx
import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import AdmZip from "adm-zip";
import { convert } from "html-to-text";
import dotenv from "dotenv";

// Load environment variables from ../books/.env
const ENV_PATH = path.resolve(__dirname, "../../books/.env");
if (fs.existsSync(ENV_PATH)) {
  dotenv.config({ path: ENV_PATH });
}

// Ensure required variables are set
const API_KEY = process.env.ANNAS_ARCHIVE_API_KEY;
if (!API_KEY) {
  console.error(
    "Error: ANNAS_ARCHIVE_API_KEY is not set in ../books/.env or the environment."
  );
  process.exit(1);
}

const argv = yargs(hideBin(process.argv))
  .option("author", {
    alias: "a",
    type: "string",
    description: "The author's full name",
    demandOption: true,
  })
  .option("title", {
    alias: "t",
    type: "string",
    description: "The book title",
    demandOption: true,
  })
  .option("year", {
    alias: "y",
    type: "number",
    description: "The publication year (for the template)",
    default: new Date().getFullYear(),
  })
  .parseSync();

async function main() {
  const { author, title, year } = argv;
  const query = `${author} ${title} epub`;

  console.log(`\n🔍 Searching Anna's Archive for: "${query}"...`);

  // We use the mcp tool locally
  let searchResult;
  try {
    const searchCmd = `mcp call search --params '{"query": "${query}"}' -f json annas-archive-mcp`;
    const searchOutput = execSync(searchCmd, {
      encoding: "utf-8",
      env: { ...process.env }, // pass current env which has ANNAS_ARCHIVE_DOMAIN if set
    });

    const parsedOutput = JSON.parse(searchOutput);
    if (parsedOutput.isError || !parsedOutput.content || parsedOutput.content.length === 0) {
      console.error("MCP Search failed:", parsedOutput);
      process.exit(1);
    }
    
    searchResult = JSON.parse(parsedOutput.content[0].text);
  } catch (error: any) {
    console.error("Error executing mcp search command:", error.message);
    process.exit(1);
  }

  if (!searchResult.results || searchResult.results.length === 0) {
    console.error("No results found on Anna's Archive for this query.");
    process.exit(1);
  }

  // Find the first EPUB result
  const epubResult = searchResult.results.find(
    (r: any) => r.format.toLowerCase() === "epub"
  );
  if (!epubResult) {
    console.error("No EPUB format found in the top search results.");
    process.exit(1);
  }

  const md5 = epubResult.md5;
  console.log(`✅ Found EPUB. MD5: ${md5}`);
  console.log(`🔗 Requesting fast download URL...`);

  let downloadUrl;
  try {
    const downloadCmd = `mcp call get_download_url --params '{"md5": "${md5}"}' -f json annas-archive-mcp`;
    const downloadOutput = execSync(downloadCmd, { encoding: "utf-8", env: { ...process.env, ANNAS_ARCHIVE_API_KEY: API_KEY } });
    
    const parsedOutput = JSON.parse(downloadOutput);
    if (parsedOutput.isError) {
      console.error("MCP Download URL failed:", parsedOutput);
      process.exit(1);
    }
    const downloadResult = JSON.parse(parsedOutput.content[0].text);
    downloadUrl = downloadResult.download_url;
  } catch (error: any) {
    console.error("Error getting download URL:", error.message);
    process.exit(1);
  }

  if (!downloadUrl) {
    console.error("Failed to extract download_url from response.");
    process.exit(1);
  }

  const authorDir = path.resolve(__dirname, "../../books", author);
  if (!fs.existsSync(authorDir)) {
    fs.mkdirSync(authorDir, { recursive: true });
  }

  const epubPath = path.join(authorDir, `${title}.epub`);
  const txtPath = path.join(authorDir, `${title}.txt`);

  console.log(`📥 Downloading to ${epubPath}...`);
  try {
    execSync(`curl -sL -o "${epubPath}" "${downloadUrl}"`);
    console.log(`✅ Download complete.`);
  } catch (error: any) {
    console.error("Failed to download file with curl:", error.message);
    process.exit(1);
  }

  console.log(`📄 Extracting text from EPUB to ${txtPath}...`);
  try {
    const zip = new AdmZip(epubPath);
    const zipEntries = zip.getEntries();
    let fullText = "";

    // Basic heuristic: read HTML/XHTML files
    const htmlEntries = zipEntries.filter(
      (entry) =>
        entry.name.endsWith(".html") ||
        entry.name.endsWith(".xhtml") ||
        entry.name.endsWith(".htm")
    );

    // Sort by name or entry path to maintain relative chapter order
    htmlEntries.sort((a, b) => a.entryName.localeCompare(b.entryName));

    for (const entry of htmlEntries) {
      const htmlContent = zip.readAsText(entry);
      const textContent = convert(htmlContent, {
        wordwrap: 130,
        preserveNewlines: true,
      });
      fullText += textContent + "\n\n";
    }

    fs.writeFileSync(txtPath, fullText, "utf-8");
    console.log(`✅ Extracted ${fullText.length} characters of text.`);
  } catch (error: any) {
    console.error("Error extracting text from EPUB:", error.message);
    console.warn("You may need to extract it manually.");
  }

  // Generate markdown template
  console.log(`📝 Scaffolding Markdown template...`);
  const slugAuthor = author
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^\w_]/g, "");
  const slugTitle = title
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^\w_]/g, "");
  const slug = `${slugAuthor}_${slugTitle}`;
  const templatePath = path.resolve(__dirname, "../data/_template.md");
  const targetPath = path.resolve(__dirname, `../data/${slug}.md`);

  if (fs.existsSync(targetPath)) {
    console.warn(`⚠️  Template already exists at data/${slug}.md. Skipping generation.`);
  } else {
    try {
      let templateContent = fs.readFileSync(templatePath, "utf-8");
      templateContent = templateContent.replace('title: "Title of the Book"', `title: "${title}"`);
      templateContent = templateContent.replace('author: "Author Name"', `author: "${author}"`);
      templateContent = templateContent.replace("year: YYYY", `year: ${year}`);
      
      fs.writeFileSync(targetPath, templateContent, "utf-8");
      console.log(`✅ Created analysis template at data/${slug}.md`);
    } catch (error: any) {
      console.error("Error scaffolding template:", error.message);
    }
  }

  console.log(`\n🎉 Ingestion complete!`);
  console.log(`1. Book text is available at: ${txtPath}`);
  console.log(`2. Analysis file ready at: data/${slug}.md`);
}

main().catch((err) => {
  console.error("Unhandled error:", err);
  process.exit(1);
});

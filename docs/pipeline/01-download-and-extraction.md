# 1. Automated Download and Extraction

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

## Next Step
Proceed to [Context Loading and Analysis](02-analysis.md).
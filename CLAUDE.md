# nartopo-tools

Ingestion pipeline for Nartopo — downloads books, extracts text, writes structural analyses.

## Quick Reference

- **What:** CLI tools for adding new works to the Nartopo database
- **Location:** Lives inside `Nartopo/` as a gitignored subdirectory (separate git repo)
- **Scripts:** `npm run ingest` (download + extract), `npm run add-analysis` (validated analysis writing via stdin JSON)
- **Data target:** Writes to `../data/` (the parent Nartopo repo's data directory)
- **Books:** Downloads to `../../books/` (sibling of Nartopo at `~/projects/books/`)
- **Env:** `../../books/.env` must have `ANNAS_ARCHIVE_API_KEY`
- **Framework ref:** `../data/_frameworks.md` — the 11-point analysis framework
- **Template:** `../data/_template.md` — canonical analysis template

## Git

- Trunk-based. Commit to `main`.
- This is a separate repo from Nartopo — has its own remote at `mistakeknot/nartopo-tools`.

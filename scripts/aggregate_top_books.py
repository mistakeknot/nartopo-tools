import json
import urllib.request
import re
from collections import Counter
from bs4 import BeautifulSoup
import os

# We will scrape a few well-known aggregate lists from the web
# 1. NPR Top 100 Sci-Fi/Fantasy
# 2. Goodreads Top Sci-Fi
# 3. Hugo Award Best Novel Winners (Wikipedia)

# Since direct scraping of dynamic sites is hard without playwright, we'll use Wikipedia
# lists which are structured and easy to parse, plus some open APIs.

# Let's get Hugo Winners
print("Fetching Hugo Award winners...")
hugo_url = "https://en.wikipedia.org/wiki/Hugo_Award_for_Best_Novel"
req = urllib.request.Request(hugo_url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
soup = BeautifulSoup(html, 'html.parser')

books = []

# Find all tables with class wikitable
tables = soup.find_all('table', class_='wikitable')
for table in tables:
    rows = table.find_all('tr')
    for row in rows:
        # Looking for rows where the first cell is a year (or has a span with an id)
        # and the row is highlighted as a winner (usually has a blue background, but we can just check if it's the main entry)
        cols = row.find_all(['th', 'td'])
        if len(cols) >= 3:
            # Usually: Year, Author, Novel
            author_cell = cols[1]
            novel_cell = cols[2]
            
            author = author_cell.get_text(strip=True).replace('*', '').split('[')[0]
            novel = novel_cell.get_text(strip=True).replace('*', '').split('[')[0]
            
            if len(novel) > 2 and len(author) > 2 and "Author" not in author and "Novel" not in novel:
                # Clean up year if it's in the first col
                year_text = cols[0].get_text(strip=True)[:4]
                if year_text.isdigit():
                    books.append((author, novel))

# Let's get Nebula Winners
print("Fetching Nebula Award winners...")
nebula_url = "https://en.wikipedia.org/wiki/Nebula_Award_for_Best_Novel"
req = urllib.request.Request(nebula_url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
soup = BeautifulSoup(html, 'html.parser')

tables = soup.find_all('table', class_='wikitable')
for table in tables:
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all(['th', 'td'])
        if len(cols) >= 3:
            author_cell = cols[1]
            novel_cell = cols[2]
            
            author = author_cell.get_text(strip=True).replace('*', '').split('[')[0]
            novel = novel_cell.get_text(strip=True).replace('*', '').split('[')[0]
            
            if len(novel) > 2 and len(author) > 2 and "Author" not in author and "Novel" not in novel:
                year_text = cols[0].get_text(strip=True)[:4]
                if year_text.isdigit():
                    books.append((author, novel))

# Let's get Locus Award for Best Sci-Fi Novel
print("Fetching Locus Award winners...")
locus_url = "https://en.wikipedia.org/wiki/Locus_Award_for_Best_Science_Fiction_Novel"
req = urllib.request.Request(locus_url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
soup = BeautifulSoup(html, 'html.parser')

tables = soup.find_all('table', class_='wikitable')
for table in tables:
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all(['th', 'td'])
        if len(cols) >= 3:
            # Locus table: Year, Winner, Author
            novel_cell = cols[1]
            author_cell = cols[2]
            
            author = author_cell.get_text(strip=True).replace('*', '').split('[')[0]
            novel = novel_cell.get_text(strip=True).replace('*', '').split('[')[0]
            
            if len(novel) > 2 and len(author) > 2 and "Author" not in author and "Winner" not in novel:
                year_text = cols[0].get_text(strip=True)[:4]
                if year_text.isdigit():
                    books.append((author, novel))

# Clean and normalize
def normalize_string(s):
    # Remove quotes, punctuation, lowercase
    s = re.sub(r'[^\w\s]', '', s).lower().strip()
    # Remove common prepositions to match slightly different titles
    s = re.sub(r'^(the|a|an)\s+', '', s)
    return s

normalized_books = []
for author, title in books:
    norm_author = normalize_string(author)
    norm_title = normalize_string(title)
    # Fix some common author name variations
    if "ursula k le guin" in norm_author or "ursula le guin" in norm_author: norm_author = "ursula k le guin"
    if "iain m banks" in norm_author or "iain banks" in norm_author: norm_author = "iain m banks"
    if "stanislaw lem" in norm_author: norm_author = "stanislaw lem"
    if "william gibson" in norm_author: norm_author = "william gibson"
    if "arthur c clarke" in norm_author: norm_author = "arthur c clarke"
    if "philip k dick" in norm_author: norm_author = "philip k dick"
    
    normalized_books.append((norm_author, norm_title, author, title))

# Count frequencies (how many lists the book appears on)
counter = Counter((a, t) for a, t, orig_a, orig_t in normalized_books)

# Get original names mapping
original_names = {}
for a, t, orig_a, orig_t in normalized_books:
    if (a, t) not in original_names:
        original_names[(a, t)] = (orig_a, orig_t)

# Load existing database to filter out what we already have
existing_works = []
nartopo_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
for filename in os.listdir(nartopo_data_dir):
    if filename.endswith(".md") and filename != "_template.md" and filename != "_frameworks.md":
        with open(os.path.join(nartopo_data_dir, filename), "r") as f:
            content = f.read()
            title_match = re.search(r'title:\s*"([^"]+)"', content)
            author_match = re.search(r'author:\s*"([^"]+)"', content)
            if title_match and author_match:
                existing_works.append((normalize_string(author_match.group(1)), normalize_string(title_match.group(1))))

print(f"Found {len(existing_works)} works already in the Nartopo database.")

# Rank and filter
ranked_todo = []
for (norm_author, norm_title), count in counter.most_common():
    if (norm_author, norm_title) not in existing_works:
        orig_author, orig_title = original_names[(norm_author, norm_title)]
        ranked_todo.append({
            "author": orig_author,
            "title": orig_title,
            "awards_count": count
        })

# Save to JSON
out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../docs/research/prioritized_backlog.json'))
with open(out_path, 'w') as f:
    json.dump(ranked_todo[:100], f, indent=2)

print(f"Saved top 100 missing books to {out_path}")

# Print top 10 to stdout
print("\n--- TOP 10 HIGH-PRIORITY TARGETS ---")
for i, book in enumerate(ranked_todo[:10]):
    print(f"{i+1}. {book['title']} by {book['author']} (Won {book['awards_count']} major awards)")


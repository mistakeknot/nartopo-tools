import json
import urllib.request
import re
from collections import Counter
from bs4 import BeautifulSoup
import os

# We will scrape a few well-known aggregate lists from the web
# 1. Hugo Award Best Novel Winners (Wikipedia)
# 2. Nebula Award Best Novel Winners (Wikipedia)
# 3. Locus Award for Best Science Fiction Novel (Wikipedia)
# 4. Arthur C. Clarke Award Winners (Wikipedia)
# 5. BSFA Award for Best Novel (Wikipedia)
# 6. NPR Top 100 Sci-Fi/Fantasy (Hardcoded knowledge)
# 7. Locus All-Time Best SF Novel Poll (Hardcoded knowledge)

books = []

print("Fetching Hugo Award winners...")
hugo_url = "https://en.wikipedia.org/wiki/Hugo_Award_for_Best_Novel"
req = urllib.request.Request(hugo_url, headers={'User-Agent': 'Mozilla/5.0'})
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
                    # Only append winners (the first row for each year usually, or we can just append all and let the frequency counter handle it, 
                    # but to keep it strictly 'winners/highly acclaimed', we will append all since even nominations are huge).
                    # Actually, for these Wikipedia tables, the winner is typically the first row for the year.
                    # We'll just grab everything; being nominated for a Hugo + Nebula + Locus is just as strong a signal.
                    books.append((author, novel))

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
            novel_cell = cols[1]
            author_cell = cols[2]
            author = author_cell.get_text(strip=True).replace('*', '').split('[')[0]
            novel = novel_cell.get_text(strip=True).replace('*', '').split('[')[0]
            if len(novel) > 2 and len(author) > 2 and "Author" not in author and "Winner" not in novel:
                year_text = cols[0].get_text(strip=True)[:4]
                if year_text.isdigit():
                    books.append((author, novel))

print("Fetching Arthur C. Clarke Award winners...")
clarke_url = "https://en.wikipedia.org/wiki/Arthur_C._Clarke_Award"
req = urllib.request.Request(clarke_url, headers={'User-Agent': 'Mozilla/5.0'})
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
                 books.append((author, novel))

print("Fetching BSFA Award winners...")
bsfa_url = "https://en.wikipedia.org/wiki/BSFA_Award_for_Best_Novel"
req = urllib.request.Request(bsfa_url, headers={'User-Agent': 'Mozilla/5.0'})
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
                 books.append((author, novel))

print("Adding NPR Top Sci-Fi & Locus All-Time Best SF Novel Poll lists...")

npr_locus_data = [
    # NPR Top 100 Sci-Fi/Fantasy
    ("J.R.R. Tolkien", "The Lord of the Rings"),
    ("Douglas Adams", "The Hitchhiker's Guide to the Galaxy"),
    ("Orson Scott Card", "Ender's Game"),
    ("Frank Herbert", "Dune"),
    ("George R.R. Martin", "A Song of Ice and Fire"),
    ("George Orwell", "1984"),
    ("Ray Bradbury", "Fahrenheit 451"),
    ("Isaac Asimov", "The Foundation Trilogy"),
    ("Aldous Huxley", "Brave New World"),
    ("Neil Gaiman", "American Gods"),
    ("William Goldman", "The Princess Bride"),
    ("Robert Jordan", "The Wheel of Time"),
    ("George Orwell", "Animal Farm"),
    ("William Gibson", "Neuromancer"),
    ("Alan Moore", "Watchmen"),
    ("Isaac Asimov", "I, Robot"),
    ("Robert A. Heinlein", "Stranger in a Strange Land"),
    ("Patrick Rothfuss", "The Kingkiller Chronicle"),
    ("Kurt Vonnegut", "Slaughterhouse-Five"),
    ("Mary Shelley", "Frankenstein"),
    ("Philip K. Dick", "Do Androids Dream of Electric Sheep?"),
    ("Margaret Atwood", "The Handmaid's Tale"),
    ("Stephen King", "The Dark Tower"),
    ("Stephen King", "The Stand"),
    ("Neal Stephenson", "Snow Crash"),
    ("Ray Bradbury", "The Martian Chronicles"),
    ("Kurt Vonnegut", "Cat's Cradle"),
    ("Neil Gaiman", "The Sandman"),
    ("Anthony Burgess", "A Clockwork Orange"),
    ("Robert A. Heinlein", "Starship Troopers"),
    ("Kurt Vonnegut", "Watership Down"),
    ("Richard Adams", "The Dragonriders of Pern"),
    ("Ursula K. Le Guin", "The Left Hand of Darkness"),
    ("Robert A. Heinlein", "The Moon is a Harsh Mistress"),
    ("Larry Niven", "Ringworld"),
    ("Marion Zimmer Bradley", "The Mists of Avalon"),
    ("T.H. White", "The Once and Future King"),
    ("Arthur C. Clarke", "Childhood's End"),
    ("Carl Sagan", "Contact"),
    ("Ursula K. Le Guin", "The Dispossessed"),
    
    # Locus All-Time Best SF Novel Poll
    ("Frank Herbert", "Dune"),
    ("Ursula K. Le Guin", "The Left Hand of Darkness"),
    ("Orson Scott Card", "Ender's Game"),
    ("Isaac Asimov", "The Foundation Trilogy"),
    ("Alfred Bester", "The Stars My Destination"),
    ("Robert A. Heinlein", "The Moon is a Harsh Mistress"),
    ("Joe Haldeman", "The Forever War"),
    ("William Gibson", "Neuromancer"),
    ("Arthur C. Clarke", "Childhood's End"),
    ("Larry Niven", "Ringworld"),
    ("Ursula K. Le Guin", "The Dispossessed"),
    ("Arthur C. Clarke", "Rendezvous with Rama"),
    ("Robert A. Heinlein", "Starship Troopers"),
    ("Neal Stephenson", "Snow Crash"),
    ("Walter M. Miller Jr.", "A Canticle for Leibowitz"),
    ("George Orwell", "1984"),
    ("Robert A. Heinlein", "Stranger in a Strange Land"),
    ("H.G. Wells", "The Time Machine"),
    ("Ray Bradbury", "The Martian Chronicles"),
    ("Philip K. Dick", "Do Androids Dream of Electric Sheep?"),
    ("H.G. Wells", "The War of the Worlds"),
    ("Isaac Asimov", "I, Robot"),
    ("Stanislaw Lem", "Solaris"),
    ("Philip K. Dick", "Ubik"),
    ("Alfred Bester", "The Demolished Man"),
    ("Kurt Vonnegut", "The Sirens of Titan"),
    ("Frederik Pohl", "Gateway"),
    ("Larry Niven and Jerry Pournelle", "The Mote in God's Eye"),
    ("John Brunner", "Stand on Zanzibar"),
    ("Arthur C. Clarke", "The City and the Stars"),
    ("Gene Wolfe", "The Book of the New Sun"),
    ("Roger Zelazny", "Lord of Light"),
    ("Vernor Vinge", "A Fire Upon the Deep"),
    ("Orson Scott Card", "Speaker for the Dead"),
    ("Neal Stephenson", "The Diamond Age"),
    ("Dan Simmons", "Hyperion"),
    ("Kim Stanley Robinson", "Red Mars"),
    ("Isaac Asimov", "The Gods Themselves"),
    ("Greg Bear", "A Fire Upon the Deep"),
    ("Iain M. Banks", "The Player of Games"),
    ("Iain M. Banks", "Use of Weapons")
]

for author, title in npr_locus_data:
    books.append((author, title))

# Clean and normalize
def normalize_string(s):
    s = re.sub(r'[^\w\s]', '', s).lower().strip()
    s = re.sub(r'^(the|a|an)\s+', '', s)
    return s

normalized_books = []
for author, title in books:
    norm_author = normalize_string(author)
    norm_title = normalize_string(title)
    if "ursula k le guin" in norm_author or "ursula le guin" in norm_author: norm_author = "ursula k le guin"
    if "iain m banks" in norm_author or "iain banks" in norm_author: norm_author = "iain m banks"
    if "stanislaw lem" in norm_author: norm_author = "stanislaw lem"
    if "william gibson" in norm_author: norm_author = "william gibson"
    if "arthur c clarke" in norm_author: norm_author = "arthur c clarke"
    if "philip k dick" in norm_author: norm_author = "philip k dick"
    if "heinlein" in norm_author: norm_author = "robert a heinlein"
    
    normalized_books.append((norm_author, norm_title, author, title))

counter = Counter((a, t) for a, t, orig_a, orig_t in normalized_books)

original_names = {}
for a, t, orig_a, orig_t in normalized_books:
    if (a, t) not in original_names:
        original_names[(a, t)] = (orig_a, orig_t)

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

ranked_todo = []
for (norm_author, norm_title), count in counter.most_common():
    if (norm_author, norm_title) not in existing_works:
        orig_author, orig_title = original_names[(norm_author, norm_title)]
        ranked_todo.append({
            "author": orig_author,
            "title": orig_title,
            "list_appearances": count
        })

out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../docs/research/prioritized_backlog.json'))
with open(out_path, 'w') as f:
    json.dump(ranked_todo, f, indent=2)

print(f"Saved {len(ranked_todo)} missing books to {out_path}")

print("\n--- TOP 10 HIGH-PRIORITY TARGETS (Awards + Locus/NPR) ---")
for i, book in enumerate(ranked_todo[:10]):
    print(f"{i+1}. {book['title']} by {book['author']} (Appeared on {book['list_appearances']} lists)")
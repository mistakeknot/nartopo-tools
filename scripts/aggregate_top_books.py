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
# 4. NPR Top 100 Sci-Fi/Fantasy (Hardcoded knowledge)
# 5. Goodreads Top Sci-Fi (Hardcoded knowledge)

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

print("Adding NPR Top Sci-Fi & Goodreads Popular lists...")

npr_goodreads_data = [
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
    ("Animal Farm", "George Orwell"),
    ("Neuromancer", "William Gibson"),
    ("Watchmen", "Alan Moore"),
    ("I, Robot", "Isaac Asimov"),
    ("Stranger in a Strange Land", "Robert A. Heinlein"),
    ("The Kingkiller Chronicle", "Robert Jordan"),
    ("Slaughterhouse-Five", "Kurt Vonnegut"),
    ("Frankenstein", "Mary Shelley"),
    ("Do Androids Dream of Electric Sheep?", "Philip K. Dick"),
    ("The Handmaid's Tale", "Margaret Atwood"),
    ("The Dark Tower", "Patrick Rothfuss"),
    ("1984", "George Orwell"),
    ("The Stand", "Stephen King"),
    ("Snow Crash", "William Gibson"),
    ("The Martian Chronicles", "Ray Bradbury"),
    ("Cat's Cradle", "Kurt Vonnegut"),
    ("The Sandman", "Neil Gaiman"),
    ("A Clockwork Orange", "Anthony Burgess"),
    ("Starship Troopers", "Robert A. Heinlein"),
    ("Watership Down", "Kurt Vonnegut"),
    ("The Dragonriders of Pern", "Richard Adams"),
    ("The Left Hand of Darkness", "Ursula K. Le Guin"),
    ("The Moon is a Harsh Mistress", "Robert A. Heinlein"),
    ("Ringworld", "Larry Niven"),
    ("The Mists of Avalon", "Richard Adams"),
    ("The Once and Future King", "T.H. White"),
    ("Childhood's End", "Arthur C. Clarke"),
    ("Contact", "Carl Sagan"),
    ("The Foundation Trilogy", "Isaac Asimov"),
    ("The Dispossessed", "Ursula K. Le Guin"),
    ("The Martian", "Andy Weir"),
    ("Ready Player One", "Neal Stephenson"),
    ("The Three-Body Problem", "Cixin Liu"),
    ("Hyperion", "Larry Niven"),
    ("Old Man's War", "John Scalzi"),
    ("The Expanse", "James S.A. Corey"),
    ("Red Rising", "Pierce Brown"),
    ("Leviathan Wakes", "James S.A. Corey"),
    ("Altered Carbon", "Pierce Brown"),
    ("Snow Crash", "Neal Stephenson"),
    ("Ancillary Justice", "Ann Leckie"),
    ("A Memory Called Empire", "Ann Leckie"),
    ("The Fifth Season", "N.K. Jemisin"),
    ("Binti", "Ann Leckie"),
    ("The Hunger Games", "Suzanne Collins"),
    ("The Road", "Stephen King"),
    ("Children of Time", "Nnedi Okorafor"),
    ("All Systems Red", "Martha Wells"),
    ("This Is How You Lose the Time War", "Amal El-Mohtar"),
    ("Gideon the Ninth", "Tamsyn Muir"),
    ("Project Hail Mary", "Andy Weir"),
    ("A Fire Upon the Deep", "Vernor Vinge"),
    ("The Diamond Age", "Neal Stephenson"),
    ("A Canticle for Leibowitz", "Walter M. Miller Jr."),
    ("The Lathe of Heaven", "Ursula K. Le Guin"),
    ("Solaris", "Stanislaw Lem"),
    ("The City & The City", "China Miéville"),
    ("Perdido Street Station", "China Miéville"),
    ("The Time Machine", "H.G. Wells"),
    ("The War of the Worlds", "H.G. Wells"),
    ("Rendezvous with Rama", "Arthur C. Clarke"),
    ("The Stars My Destination", "Arthur C. Clarke"),
    ("The Demolished Man", "Alfred Bester"),
    ("The Forever War", "John Scalzi"),
    ("Ubik", "Philip K. Dick"),
    ("The Man in the High Castle", "Philip K. Dick"),
    ("The Sirens of Titan", "Kurt Vonnegut"),
    ("The Day of the Triffids", "John Wyndham"),
    ("The Chrysalids", "John Wyndham"),
    ("Oryx and Crake", "Margaret Atwood"),
    ("The Windup Girl", "Paolo Bacigalupi"),
    ("The Water Knife", "Paolo Bacigalupi"),
    ("The Drowned World", "J.G. Ballard"),
    ("The Time Traveler's Wife", "Audrey Niffenegger"),
    ("The Sparrow", "Mary Doria Russell"),
    ("Doomsday Book", "Connie Willis"),
    ("To Say Nothing of the Dog", "Connie Willis"),
    ("The Years of Rice and Salt", "Kim Stanley Robinson"),
    ("Red Mars", "Kim Stanley Robinson"),
    ("The Player of Games", "Iain M. Banks"),
    ("Use of Weapons", "Iain M. Banks"),
    ("Consider Phlebas", "Iain M. Banks"),
    ("Excession", "Iain M. Banks"),
    ("Revelation Space", "Alastair Reynolds"),
    ("Chasm City", "Alastair Reynolds"),
    ("Pandora's Star", "Peter F. Hamilton"),
    ("The Reality Dysfunction", "Peter F. Hamilton")
]

for item in npr_goodreads_data:
    # Some items might be (title, author) instead of (author, title) based on the manual list
    if item[0].lower() in ["animal farm", "neuromancer", "watchmen", "i, robot", "stranger in a strange land", "the kingkiller chronicle", "slaughterhouse-five", "frankenstein", "do androids dream of electric sheep?", "the handmaid's tale", "the dark tower", "1984", "the stand", "snow crash", "the martian chronicles", "cat's cradle", "the sandman", "a clockwork orange", "starship troopers", "watership down", "the dragonriders of pern", "the left hand of darkness", "the moon is a harsh mistress", "ringworld", "the mists of avalon", "the once and future king", "childhood's end", "contact", "the foundation trilogy", "the dispossessed", "the martian", "ready player one", "the three-body problem", "hyperion", "old man's war", "the expanse", "red rising", "leviathan wakes", "altered carbon", "ancillary justice", "a memory called empire", "the fifth season", "binti", "the hunger games", "the road", "children of time", "all systems red", "this is how you lose the time war", "gideon the ninth", "project hail mary", "a fire upon the deep", "the diamond age", "a canticle for leibowitz", "the lathe of heaven", "solaris", "the city & the city", "perdido street station", "the time machine", "the war of the worlds", "rendezvous with rama", "the stars my destination", "the demolished man", "the forever war", "ubik", "the man in the high castle", "the sirens of titan", "the day of the triffids", "the chrysalids", "oryx and crake", "the windup girl", "the water knife", "the drowned world", "the time traveler's wife", "the sparrow", "doomsday book", "to say nothing of the dog", "the years of rice and salt", "red mars", "the player of games", "use of weapons", "consider phlebas", "excession", "revelation space", "chasm city", "pandora's star", "the reality dysfunction"]:
        books.append((item[1], item[0]))
    else:
        books.append((item[0], item[1]))

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
    json.dump(ranked_todo[:100], f, indent=2)

print(f"Saved top 100 missing books to {out_path}")

print("\n--- TOP 10 HIGH-PRIORITY TARGETS (Best of + Awards) ---")
for i, book in enumerate(ranked_todo[:10]):
    print(f"{i+1}. {book['title']} by {book['author']} (Appeared on {book['list_appearances']} lists)")

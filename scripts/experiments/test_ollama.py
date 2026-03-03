import json
import urllib.request
import sys

payload = {"model": "nomic-embed-text", "prompt": "test"}
req = urllib.request.Request(
    "http://100.107.177.128:11434/api/embeddings",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
try:
    response = urllib.request.urlopen(req, timeout=10)
    print("Success. Received embedding of length:", len(json.loads(response.read())["embedding"]))
except Exception as e:
    print("Error:", e)

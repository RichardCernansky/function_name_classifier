import ndjson
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = ndjson.load(f)

for entry in data:
    if entry.get("tag") == sys.argv[2]: 
        print(entry["source_code"])

#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, sys
root = Path(__file__).resolve().parents[1]
manifest = root / "FILE_MANIFEST.csv"

def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

errors = []
with manifest.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        p = root / row["path"]
        if not p.exists():
            errors.append(("missing", row["path"]))
            continue
        if p.stat().st_size != int(row["size_bytes"]):
            errors.append(("size", row["path"]))
        if sha256(p) != row["sha256"]:
            errors.append(("sha256", row["path"]))
if errors:
    print("FAIL manifest verification")
    for e in errors[:50]:
        print(e)
    sys.exit(1)
print("PASS manifest verification")

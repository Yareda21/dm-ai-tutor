# preprocess_and_embed.py
import os, json
from pathlib import Path

DOCS_DIR = Path("docs")   # put your doc.txt files here
OUT = Path("chunks.jsonl")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks=[]
    start=0
    L=len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, chunk))
        if end >= L:
            break
        start = end - overlap
    return chunks

def main():
    rows=[]
    for p in sorted(DOCS_DIR.glob("*.txt")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for offset, chunk in chunk_text(txt):
            rows.append({"id": f"{p.stem}_{offset}", "source": p.name, "text": chunk, "chunk_len": len(chunk)})
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Wrote", OUT, "with", len(rows), "chunks")

if __name__ == "__main__":
    main()

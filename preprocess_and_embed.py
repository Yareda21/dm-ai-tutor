# preprocess_and_embed.py
import os, json, uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DOCS_DIR = Path("docs")   # put doc.txt here
OUT_FILE = Path("chunks.jsonl")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
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
    model = SentenceTransformer(EMBED_MODEL)
    rows = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        txt = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(txt)
        for offset, chunk in chunks:
            chunk_id = f"{path.stem}_{offset}"
            rows.append({"id": chunk_id, "source": path.name, "text": chunk})

    # compute embeddings in batches
    texts = [r["text"] for r in rows]
    batch_size = 64
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True)
        for e in embs:
            embeddings.append(e.astype("float32").tolist())

    # write jsonl
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r, emb in zip(rows, embeddings):
            r_out = {
                "id": r["id"],
                "source": r["source"],
                "text": r["text"],
                "chunk_len": len(r["text"]),
                "embedding": emb
            }
            f.write(json.dumps(r_out, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_FILE} with {len(rows)} chunks.")

if __name__ == "__main__":
    main()

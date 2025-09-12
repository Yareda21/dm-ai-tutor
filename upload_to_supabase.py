# upload_to_supabase.py
import os, json, time, requests
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from pgvector import Vector

load_dotenv()
DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
EMBED_SERVICE_URL = os.getenv("EMBED_SERVICE_URL", "http://localhost:8001/embed")
CHUNKS_FILE = os.getenv("CHUNKS_JSONL", "chunks.jsonl")

def connect():
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    register_vector(conn)
    return conn

def get_embedding(text):
    """Get embedding from your embedding service"""
    try:
        resp = requests.post(EMBED_SERVICE_URL, json={"texts": [text]}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"][0]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def upsert(conn, row, emb=None):
    cur = conn.cursor()
    sql = """
    INSERT INTO rag_chunks (id, source, chunk_text, chunk_len, embedding)
    VALUES (%s,%s,%s,%s,%s)
    ON CONFLICT (id) DO UPDATE SET
      chunk_text = EXCLUDED.chunk_text,
      chunk_len = EXCLUDED.chunk_len,
      embedding = COALESCE(EXCLUDED.embedding, rag_chunks.embedding),
      source = EXCLUDED.source;
    """
    emb_val = Vector(emb) if emb is not None else None
    cur.execute(sql, (row["id"], row.get("source"), row["text"], row.get("chunk_len",len(row["text"])), emb_val))
    conn.commit()
    cur.close()

def main():
    conn = connect()
    cnt = 0
    error_count = 0
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total_lines = len(lines)
    
    print(f"Processing {total_lines} chunks...")
    
    for line in lines:
        row = json.loads(line)
        
        # Get embedding for this chunk
        embedding = get_embedding(row["text"])
        
        if embedding:
            upsert(conn, row, emb=embedding)
            cnt += 1
            if cnt % 10 == 0:
                print(f"Processed {cnt}/{total_lines} chunks")
        else:
            print(f"Failed to get embedding for chunk {row['id']}")
            error_count += 1
        
        # Add a small delay to avoid overwhelming the embedding service
        time.sleep(0.1)
    
    conn.close()
    print(f"Done! Processed {cnt} chunks, {error_count} errors")

if __name__ == "__main__":
    main()
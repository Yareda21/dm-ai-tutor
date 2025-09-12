# upload_to_supabase.py
import os, json
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from pgvector import Vector

load_dotenv()
DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
CHUNKS_FILE = os.getenv("CHUNKS_JSONL","chunks.jsonl")

def connect():
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    register_vector(conn)
    return conn

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
    cnt=0
    with open(CHUNKS_FILE,"r",encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            upsert(conn, row, emb=None)  # emb=None for now
            cnt += 1
            if cnt % 100 == 0:
                print("Inserted", cnt)
    conn.close()
    print("Done", cnt)

if __name__ == "__main__":
    main()

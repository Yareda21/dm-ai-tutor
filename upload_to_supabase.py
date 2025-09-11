# upload_to_supabase.py
import os, json
import psycopg2
from pgvector.psycopg2 import register_vector
from pgvector import Vector
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")  # from Supabase settings
# print("Using database:", DATABASE_URL)
def connect():
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    register_vector(conn)
    return conn

def insert_row(conn, row):
    cur = conn.cursor()
    sql = """
    INSERT INTO rag_chunks (id, source, chunk_text, chunk_len, embedding)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        chunk_text = EXCLUDED.chunk_text,
        chunk_len = EXCLUDED.chunk_len,
        embedding = EXCLUDED.embedding,
        source = EXCLUDED.source;
    """
    emb = Vector(row["embedding"])
    cur.execute(sql, (row["id"], row["source"], row["text"], row["chunk_len"], emb))
    conn.commit()
    cur.close()

def main():
    from urllib.parse import urlparse
    # after load_dotenv() and DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
    if not DATABASE_URL:
        raise SystemExit("SUPABASE_DATABASE_URL not found in environment. Please set it in .env")
    print("Using SUPABASE_DATABASE_URL host:", urlparse(DATABASE_URL).hostname)
    conn = connect()
    count = 0
    with open("chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            insert_row(conn, row)
            count += 1
            if count % 100 == 0:
                print("Inserted", count)
    conn.close()
    print("Done. Inserted total:", count)

if __name__ == "__main__":
    main()

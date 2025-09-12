# rag_api/supabase_rag.py
import os
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from pgvector import Vector

load_dotenv()
DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("SUPABASE_DATABASE_URL not set in environment")

def retrieve_topk(query_embedding: list, k: int = 4):
    """
    Opens a fresh DB connection for each call, queries the top-k nearest vectors,
    and closes the connection. Safer for development and avoids 'connection closed' errors.
    """
    conn = None
    try:
        # create a new connection (sslmode required for Supabase)
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        # register vector adapter for this connection
        register_vector(conn)
        cur = conn.cursor()
        sql = """
        SELECT id, chunk_text, embedding <-> %s as distance
        FROM rag_chunks
        ORDER BY distance
        LIMIT %s;
        """
        cur.execute(sql, (Vector(query_embedding), k))
        rows = cur.fetchall()
        cur.close()
        results = [{"id": r[0], "text": r[1], "score": float(r[2])} for r in rows]
        return results
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

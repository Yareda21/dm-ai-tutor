# supabase_rag.py
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from pgvector.psycopg2 import register_vector
from pgvector import Vector

load_dotenv()

DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "5"))

_pool = None

def init_db_pool():
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(1, DB_POOL_MAX, dsn=DATABASE_URL, sslmode="require")
        # register vector adapter on one conn
        conn = _pool.getconn()
        register_vector(conn)
        _pool.putconn(conn)
    return _pool

def get_conn():
    if _pool is None:
        raise RuntimeError("DB pool not initialized - call init_db_pool() first")
    return _pool.getconn()

def put_conn(conn):
    _pool.putconn(conn)

def retrieve_topk(query_embedding: list, k: int = 4):
    """
    Returns list of dicts: {id, source, text, score}
    Uses pgvector '<->' L2 operator. Lower distance = more similar.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        sql = """
        SELECT id, source, chunk_text, embedding <-> %s as distance
        FROM rag_chunks
        ORDER BY distance
        LIMIT %s;
        """
        cur.execute(sql, (Vector(query_embedding), k))
        rows = cur.fetchall()
        cur.close()
    finally:
        put_conn(conn)

    results = []
    for r in rows:
        results.append({"id": r[0], "source": r[1], "text": r[2], "score": float(r[3])})
    return results

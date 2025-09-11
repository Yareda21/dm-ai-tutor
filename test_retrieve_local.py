# test_retrieve_local.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from pgvector import Vector

load_dotenv()
DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("Please set SUPABASE_DATABASE_URL in .env")

# init
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
conn = psycopg2.connect(DATABASE_URL, sslmode="require")
register_vector(conn)

def compute_query_embedding(text):
    emb = model.encode([text], convert_to_numpy=True)[0].astype("float32")
    return emb.tolist()

def retrieve_topk(query_emb, k=4):
    cur = conn.cursor()
    sql = """
    SELECT id, source, chunk_text, embedding <-> %s as distance
    FROM rag_chunks
    ORDER BY distance
    LIMIT %s;
    """
    cur.execute(sql, (Vector(query_emb), k))
    rows = cur.fetchall()
    cur.close()
    return rows

if __name__ == "__main__":
    q = "What is digital marketing?"
    q_emb = compute_query_embedding(q)
    rows = retrieve_topk(q_emb, k=4)
    for r in rows:
        print("ID:", r[0], "SRC:", r[1], "DIST:", r[3])
        print("TEXT SNIPPET:", r[2][:200].replace("\n"," "), "\n---\n")

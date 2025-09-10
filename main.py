
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from starlette.middleware.cors import CORSMiddleware
# from starlette.concurrency import run_in_threadpool
# import os
# import traceback

# # your LLM import (keeps the same as your original)
# from langchain_openai import ChatOpenAI

# load_dotenv()

# # initialize LLM (same settings you used)
# llm = ChatOpenAI(
#     model="openai/gpt-oss-20b:free",
#     openai_api_key=os.getenv("SECOND_OPEN_AI_KEY"),
#     openai_api_base="https://openrouter.ai/api/v1",
#     temperature=0.0,
#     max_tokens=4096,
# )

# app = FastAPI(title="AI Agent API")

# # CORS: set ALLOWED_ORIGINS in Render to your frontend origin(s) like https://yoursite.vercel.app
# raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
# if raw_origins.strip() == "*" or raw_origins.strip() == "":
#     origins = ["*"]
# else:
#     origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     prompt: str
#     session_id: str | None = "anon_user"

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.post("/ai/chat")
# async def ai_chat(req: ChatRequest):
#     try:
#         # run the (possibly blocking) llm.invoke in a thread so the server stays responsive
#         result = await run_in_threadpool(llm.invoke, req.prompt)

#         # best-effort extraction of a human-friendly text answer
#         answer = None
#         if isinstance(result, dict):
#             answer = result.get("answer") or result.get("content") or result.get("response")
#         if not answer:
#             answer = str(result)

#         return {"answer": answer, "raw": result}
#     except Exception as e:
#         # return useful debug info (strip in prod if needed)
#         return {"error": str(e), "traceback": traceback.format_exc()}

# rag_fastapi.py
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import os, glob, pickle, traceback

# LLM import - keep your original
from langchain_openai import ChatOpenAI

# Embedding + index
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

# -------------------------
# initialize LLM (your existing)
# -------------------------
llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    openai_api_key=os.getenv("SECOND_OPEN_AI_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=4096,
)

# -------------------------
# Config
# -------------------------
DOCS_DIR = os.getenv("RAG_DOCS_DIR", "docs")     # put .txt (or pre-extracted text) here
INDEX_PATH = os.getenv("RAG_INDEX_PATH", "faiss_index.ivf")  # file to persist index
META_PATH = os.getenv("RAG_META_PATH", "faiss_meta.pkl")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))   # characters per chunk
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "3000"))  # rough trimming

# -------------------------
# Simple chunker
# -------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, chunk))
        start = max(end - overlap, end) if end < L else end
    return chunks

# -------------------------
# Build / load index (very small, simple)
# -------------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
# We'll store metadata list aligned with faiss vectors
index = None
metadata = []

def build_index_from_docs(docs_dir=DOCS_DIR):
    global index, metadata
    texts = []
    metadata = []

    file_paths = glob.glob(os.path.join(docs_dir, "*.txt"))
    if not file_paths:
        print("No txt files found in docs/ - index will be empty.")
        # create empty index with expected dim
        dim = embed_model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        return

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            full = f.read()
        chunks = chunk_text(full)
        for start, chunk in chunks:
            metadata.append({
                "source": os.path.basename(path),
                "offset": start,
                "text": chunk
            })
            texts.append(chunk)

    # compute embeddings (in threadpool because it's blocking)
    embs = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # normalize for inner-product similarity
    faiss.normalize_L2(embs)

    dim = embs.shape[1]
    # simple flat index (fast and ok for small corpora)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # persist
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Built index with {index.ntotal} vectors.")

def load_index():
    global index, metadata
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
        print(f"Loaded index with {index.ntotal} vectors.")
    else:
        build_index_from_docs()

# Build/load on startup
load_index()

# -------------------------
# Retrieval helper
# -------------------------
def retrieve(query: str, top_k=TOP_K):
    if index is None or index.ntotal == 0:
        return []

    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx]
        results.append({
            "score": float(score),
            "source": m["source"],
            "offset": m["offset"],
            "text": m["text"]
        })
    return results

# -------------------------
# Prompt assembly
# -------------------------
SYSTEM_PROMPT = (
    "You are an assistant that must answer using the provided CONTEXT only. "
    "If the answer is not in the context, say 'I don't know' or provide a best effort and clearly note uncertainty. "
    "Cite sources by filename in square brackets like [source.txt]."
)

def build_prompt(question: str, retrieved: list):
    if not retrieved:
        return f"{SYSTEM_PROMPT}\n\nUser: {question}\n\nContext: <no results available>"

    # join context pieces with separators and source labels
    pieces = []
    total_chars = 0
    for r in retrieved:
        text = r["text"].strip()
        src = r["source"]
        part = f"[{src}]\n{text}\n---\n"
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            break
        pieces.append(part)
        total_chars += len(part)

    context = "\n".join(pieces)
    prompt = (
        f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nUser question: {question}\n\n"
        "Answer using only the CONTEXT above and cite the sources used. "
        "If the context does not contain the answer, say you don't know."
    )
    return prompt

# -------------------------
# FastAPI (your original config)
# -------------------------
app = FastAPI(title="AI Agent API - RAG enabled")

raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
if raw_origins.strip() == "*" or raw_origins.strip() == "":
    origins = ["*"]
else:
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    session_id: str | None = "anon_user"
    use_rag: bool | None = True   # if False, fall back to plain LLM

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ai/chat")
async def ai_chat(req: ChatRequest):
    try:
        if req.use_rag:
            # 1) retrieve
            retrieved = await run_in_threadpool(retrieve, req.prompt, TOP_K)
            # 2) build prompt
            prompt = build_prompt(req.prompt, retrieved)
            # 3) call llm in threadpool (your pattern)
            result = await run_in_threadpool(llm.invoke, prompt)
            # extract answer
            answer = None
            if isinstance(result, dict):
                answer = result.get("answer") or result.get("content") or result.get("response")
            if not answer:
                answer = str(result)
            return {
                "answer": answer,
                "raw": result,
                "sources": [{"source": r["source"], "score": r["score"]} for r in retrieved]
            }
        else:
            # fallback to plain LLM
            result = await run_in_threadpool(llm.invoke, req.prompt)
            answer = None
            if isinstance(result, dict):
                answer = result.get("answer") or result.get("content") or result.get("response")
            if not answer:
                answer = str(result)
            return {"answer": answer, "raw": result, "sources": []}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

# optional: endpoint to rebuild index (admin)
@app.post("/ai/reindex")
async def reindex():
    try:
        await run_in_threadpool(build_index_from_docs)
        return {"status": "ok", "total_vectors": index.ntotal if index else 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

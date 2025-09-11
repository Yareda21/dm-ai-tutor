
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# import os

# load_dotenv()

# llm = ChatOpenAI(
#     model="openai/gpt-oss-20b:free",
#     openai_api_key=os.getenv("SECOND_OPEN_AI_KEY"),
#     openai_api_base="https://openrouter.ai/api/v1",
#     temperature=0.0,
#     max_tokens=4096,
# )

# result = llm.invoke("hi who are you")
# print(result)

# main.py
import os
import traceback
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# LLM import (as you used)
from langchain_openai import ChatOpenAI

# our helper
from supabase_rag import init_db_pool, retrieve_topk

load_dotenv()

# CONFIG
SECOND_OPEN_AI_KEY = os.getenv("SECOND_OPEN_AI_KEY")
SUPABASE_DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
TOP_K = int(os.getenv("TOP_K", "4"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1536"))

# LLM init (OpenRouter base)
llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    openai_api_key=SECOND_OPEN_AI_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=2048,
)

# FastAPI app
app = FastAPI(title="RAG API (Supabase + OpenRouter)")

# CORS setup
if ALLOWED_ORIGINS.strip() == "*" or ALLOWED_ORIGINS.strip() == "":
    origins = ["*"]
else:
    origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    prompt: str
    session_id: str | None = "anon_user"
    use_rag: bool | None = True

@app.on_event("startup")
async def startup():
    # Init DB pool (blocking) in threadpool
    await run_in_threadpool(init_db_pool)
    print("DB pool initialized.")

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Embeddings via OpenRouter (blocking -> run_in_threadpool) ----------
def get_query_embedding_via_openrouter(text: str):
    """
    Sends a request to OpenRouter embeddings endpoint.
    Adjust payload/endpoint if you use another provider.
    """
    if not SECOND_OPEN_AI_KEY:
        raise RuntimeError("SECOND_OPEN_AI_KEY not set")

    url = "https://openrouter.ai/v1/embeddings"
    headers = {"Authorization": f"Bearer {SECOND_OPEN_AI_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": text}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # unwrap depending on provider shape:
    # OpenRouter-like: data["data"][0]["embedding"]
    if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
        emb = data["data"][0].get("embedding") or data["data"][0].get("vector") or None
        if emb is None:
            raise RuntimeError("Unexpected embeddings response structure: " + str(data))
        return emb
    # fallback: OpenAI-like
    if "data" in data and len(data["data"]) > 0 and "embedding" in data["data"][0]:
        return data["data"][0]["embedding"]
    raise RuntimeError("Could not parse embedding response: " + str(data))

# ---------- Prompt builder ----------
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use ONLY the provided CONTEXT to answer. "
    "Cite sources by filename in square brackets like [doc.txt]. If the answer is not in the context, say 'I don't know.'"
)

def build_prompt(question: str, retrieved: list):
    pieces = []
    total_chars = 0
    MAX_CONTEXT_CHARS = 3000
    for r in retrieved:
        part = f"[{r['source']}] {r['text']}\n---\n"
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            break
        pieces.append(part)
        total_chars += len(part)
    context = "\n".join(pieces) if pieces else "<no context available>"
    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return prompt

# ---------- API endpoint ----------
@app.post("/ai/chat")
async def ai_chat(req: ChatRequest):
    try:
        if req.use_rag:
            # 1) compute query embedding (use run_in_threadpool because it blocks w/ requests)
            query_emb = await run_in_threadpool(get_query_embedding_via_openrouter, req.prompt)

            # 2) retrieve top-k from Supabase (blocking DB IO)
            retrieved = await run_in_threadpool(retrieve_topk, query_emb, TOP_K)

            # 3) build prompt & call LLM (blocking)
            prompt = build_prompt(req.prompt, retrieved)
            result = await run_in_threadpool(llm.invoke, prompt)

            # 4) extract answer
            answer = None
            if isinstance(result, dict):
                answer = result.get("content") or result.get("answer") or result.get("response")
            if not answer:
                answer = str(result)

            return {"answer": answer, "raw": result, "sources": retrieved}
        else:
            # non-RAG path
            result = await run_in_threadpool(llm.invoke, req.prompt)
            answer = None
            if isinstance(result, dict):
                answer = result.get("content") or result.get("answer") or result.get("response")
            if not answer:
                answer = str(result)
            return {"answer": answer, "raw": result, "sources": []}
    except Exception as e:


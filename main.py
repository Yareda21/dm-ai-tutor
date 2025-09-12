# rag_api/main.py
import os
import re
import requests
import traceback
import json
from typing import Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool

from supabase_rag import retrieve_topk

load_dotenv()
EMBED_SERVICE_URL = os.getenv("EMBED_SERVICE_URL", "http://localhost:8001/embed")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free") 
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2000")) 
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "800"))

SAFE_TOP_K = int(os.getenv("TOP_K", "4"))

app = FastAPI(title="RAG API")
print("RAG API starting with EMBED_SERVICE_URL", EMBED_SERVICE_URL)
class ChatRequest(BaseModel):
    prompt: str
    use_rag: bool = True

@app.get("/health")
def health():
    return {"status":"ok"}

def sanitize_text(s: str) -> str:
    """Remove control characters and collapse excessive whitespace."""
    if not isinstance(s, str):
        return ""
    # remove non-printable/control chars except newline/tab
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u0080-\uFFFF]+", " ", s)
    # collapse multiple whitespace/newlines
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def build_trimmed_context(retrieved, max_total=MAX_CONTEXT_CHARS, per_chunk=MAX_CHUNK_CHARS):
    """
    Sanitize and trim retrieved list of {"text":...} into a single context string
    not exceeding max_total chars. We include chunks up to the limit.
    """
    parts = []
    total = 0
    for r in retrieved:
        txt = sanitize_text(r.get("text",""))
        if not txt:
            continue
        if len(txt) > per_chunk:
            txt = txt[:per_chunk] + " …(truncated)"
        # if adding this would exceed total, add truncated remainder and break
        if total + len(txt) > max_total:
            remaining = max(0, max_total - total)
            if remaining <= 10:
                break
            parts.append(txt[:remaining] + " …(truncated)")
            total += remaining
            break
        parts.append(txt)
        total += len(txt)
    return "\n\n".join(parts)

def get_query_embedding_via_embed_service(text: str):
    resp = requests.post(EMBED_SERVICE_URL, json={"texts":[text]}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    emb = data["embeddings"][0]
    return emb

def _post_and_debug(url: str, headers: dict, payload: dict, timeout: int = 60) -> Tuple[requests.Response, str]:
    """Post and return (response, raw_text). No exception raised here."""
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error calling OpenRouter: {e}") from e
    text = r.text or ""
    return r, text

def call_openrouter_llm(prompt: str, max_tokens: int = 1024, temperature: float = 0.0):
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_KEY not set")

    url = f"{OPENROUTER_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful tutor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # Debug info: size and a trimmed preview of prompt/payload
    prompt_len = len(prompt)
    try:
        payload_preview = json.dumps(payload)[:2000]
    except Exception:
        payload_preview = "<could not json.dumps payload>"

    print("[OpenRouter debug] POST", url)
    print("[OpenRouter debug] model:", OPENROUTER_MODEL)
    print(f"[OpenRouter debug] prompt_len: {prompt_len} chars")
    print("[OpenRouter debug] payload_preview:", payload_preview)

    r, raw_text = _post_and_debug(url, headers, payload)
    # If response is not ok, print debug and raise clear error
    if not r.ok:
        print("[OpenRouter debug] status:", r.status_code)
        print("[OpenRouter debug] response body (first 2000 chars):", raw_text[:2000])
        # include both in raised error for FastAPI to return (so you can see it)
        raise RuntimeError(f"OpenRouter returned {r.status_code}. Body: {raw_text[:2000]}")
    # parse and extract text
    try:
        data = r.json()
    except Exception:
        # if provider returned non-JSON despite 200, return raw_text
        return raw_text

    # Typical shape: choices[0].message.content or choices[0].text
    choices = data.get("choices", [])
    if choices and isinstance(choices, list) and len(choices) > 0:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message") or {}
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
            if first.get("text"):
                return first.get("text")
    # Fallbacks for other shapes
    for k in ("response", "text", "content"):
        if isinstance(data.get(k), str):
            return data.get(k)
    # Last resort return JSON string
    return json.dumps(data)

@app.post("/ai/chat")
async def ai_chat(req: ChatRequest):
    try:
        if req.use_rag:
            # 1) embed query
            query_emb = await run_in_threadpool(get_query_embedding_via_embed_service, req.prompt)
            print(f"Query embedding length: {len(query_emb)}") 
            # 2) retrieve (use SAFE_TOP_K)
            retrieved = await run_in_threadpool(retrieve_topk, query_emb, SAFE_TOP_K)
            print("Retrieved chunks:", retrieved)
            # 3) build sanitized, trimmed context
            context = build_trimmed_context(retrieved, max_total=MAX_CONTEXT_CHARS, per_chunk=MAX_CHUNK_CHARS)
            # if context empty, fall back to no-rag LLM call (avoids sending empty placeholders)
            if not context:
                prompt = req.prompt
                full_prompt_is_rag = False
            else:
                prompt = (
                    "You are a helpful tutor. Use ONLY the context below to answer the question. If not present, say \"I don't know.\"\n\n"
                    f"CONTEXT:\n{context}\n\nQuestion: {req.prompt}\n\nAnswer:"
                )
                full_prompt_is_rag = True

            # 4) call LLM with debug wrapper
            answer = await run_in_threadpool(call_openrouter_llm, prompt)
            return {"answer": answer}
        else:
            answer = await run_in_threadpool(call_openrouter_llm, req.prompt)
            return {"answer": answer}
    except Exception as e:
        # raise 500 for client
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

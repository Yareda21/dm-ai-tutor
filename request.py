# hf_model_probe_fixed.py
import os, requests, json
from dotenv import load_dotenv
load_dotenv()

HF_KEY = os.getenv("HUGGINGFACE_API_KEY")
MODEL = os.getenv("HUGGINGFACE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
if not HF_KEY:
    raise SystemExit("Set HUGGINGFACE_API_KEY in .env")

BASE = f"https://api-inference.huggingface.co/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_KEY}", "Content-Type": "application/json"}

candidates = [
    {"inputs": "hello world"},
    {"inputs": ["hello world"]},
    {"inputs": {"sentences": ["hello world"]}},
    {"inputs": {"sentence": "hello world"}},
    {"inputs": {"text": "hello world"}},
    {"inputs": {"text": ["hello world"]}},
    {"inputs": {"source": "hello world"}},
]

def try_extract_vector(obj):
    # Try common shapes, return vector list or None
    # 1) flat list of numbers
    if isinstance(obj, list) and obj and all(isinstance(x, (int, float)) for x in obj):
        return obj
    # 2) outer list with first element a flat list
    if isinstance(obj, list) and obj and isinstance(obj[0], list) and all(isinstance(x, (int, float)) for x in obj[0]):
        return obj[0]
    # 3) nested token vectors -> average across token vectors
    if isinstance(obj, list) and obj and isinstance(obj[0], list) and isinstance(obj[0][0], list):
        # average token vectors without numpy
        token_vecs = obj[0]  # list of token vectors
        L = len(token_vecs)
        if L == 0:
            return None
        dim = len(token_vecs[0])
        acc = [0.0] * dim
        for tv in token_vecs:
            for i, v in enumerate(tv):
                acc[i] += float(v)
        return [x / L for x in acc]
    # 4) dict with embedding-like keys
    if isinstance(obj, dict):
        for k in ("embedding","vector","embeddings","features"):
            if k in obj:
                return try_extract_vector(obj[k])
    return None

def deep_scan_for_vector(j):
    # iterative stack-based DFS that finds first list-of-numbers
    stack = [j]
    while stack:
        node = stack.pop()
        vec = try_extract_vector(node)
        if vec:
            return vec
        if isinstance(node, dict):
            for v in node.values():
                stack.append(v)
        elif isinstance(node, list):
            for el in node:
                stack.append(el)
    return None

def run_probe():
    last_resp = None
    for payload in candidates:
        print("\nTRY payload:", json.dumps(payload))
        try:
            r = requests.post(BASE, headers=HEADERS, json=payload, timeout=30)
        except Exception as e:
            print("REQUEST ERROR:", e)
            continue
        print("STATUS:", r.status_code)
        body = (r.text or "")[:1200]
        print("BODY SNIPPET:", body[:500])
        last_resp = r
        # try parse JSON
        try:
            j = r.json()
        except Exception as e:
            print("JSON parse error:", e)
            continue
        # Try direct extraction
        vec = try_extract_vector(j)
        if vec:
            print("OK! extracted vector length:", len(vec))
            print("Vector sample (first 8):", vec[:8])
            return
        # deep scan
        found = deep_scan_for_vector(j)
        if found:
            print("OK (scanned) vector length:", len(found))
            print("sample:", found[:8])
            return
        print("No vector found for this payload. Response JSON type/summary:", type(j))
    # nothing worked
    if last_resp is not None:
        print("\nNo payload produced an embedding. Last response status:", last_resp.status_code)
        print("Last response body (first 1000 chars):")
        print((last_resp.text or "")[:1000])
    else:
        print("\nNo response obtained from server for any payload.")

if __name__ == "__main__":
    run_probe()

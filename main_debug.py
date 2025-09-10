# main_debug.py
import os
import traceback
from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# Dummy placeholders for your real index/llm code
index = None
metadata = []

app = FastAPI(title="AI Agent API - Debug Mode")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    print("APP STARTUP fired")
    # IMPORTANT: do NOT do heavy blocking work here synchronously.
    # If you must run blocking code, use run_in_threadpool or an async wrapper.
    try:
        # Example: call your load_index() in a thread so startup doesn't block event loop
        # await run_in_threadpool(load_index)   # uncomment and replace with your loader
        print("Pretend to load index here (in real app call load_index in a thread).")
    except Exception as e:
        print("Error during startup:", e)
        traceback.print_exc()

@app.on_event("shutdown")
async def on_shutdown():
    print("APP SHUTDOWN fired")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Got request: {request.method} {request.url.path}")
    try:
        resp = await call_next(request)
        print(f"Request completed: {request.method} {request.url.path} -> {resp.status_code}")
        return resp
    except Exception as e:
        print("Unhandled exception in request:", e)
        traceback.print_exc()
        raise

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ai/chat")
async def ai_chat(payload: dict):
    # Minimal dummy response; replace with your actual handler.
    return {"answer": "dummy", "raw": {}, "sources": []}

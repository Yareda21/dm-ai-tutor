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
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import os
import traceback

# your LLM import (keeps the same as your original)
from langchain_openai import ChatOpenAI

load_dotenv()

# initialize LLM (same settings you used)
llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    openai_api_key=os.getenv("SECOND_OPEN_AI_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=4096,
)

app = FastAPI(title="AI Agent API")

# CORS: set ALLOWED_ORIGINS in Render to your frontend origin(s) like https://yoursite.vercel.app
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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ai/chat")
async def ai_chat(req: ChatRequest):
    try:
        # run the (possibly blocking) llm.invoke in a thread so the server stays responsive
        result = await run_in_threadpool(llm.invoke, req.prompt)

        # best-effort extraction of a human-friendly text answer
        answer = None
        if isinstance(result, dict):
            answer = result.get("answer") or result.get("content") or result.get("response")
        if not answer:
            answer = str(result)

        return {"answer": answer, "raw": result}
    except Exception as e:
        # return useful debug info (strip in prod if needed)
        return {"error": str(e), "traceback": traceback.format_exc()}

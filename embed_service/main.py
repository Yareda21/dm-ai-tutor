# embed_service/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="Embed Service")
print("Loading embed model:", MODEL_NAME)
EMB_MODEL = SentenceTransformer(MODEL_NAME)

class EmbRequest(BaseModel):
    texts: list[str]

@app.post("/embed")
def embed(req: EmbRequest):
    try:
        embs = EMB_MODEL.encode(req.texts, convert_to_numpy=True)
        # convert to lists for JSON
        return {"embeddings": [e.astype("float32").tolist() for e in embs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

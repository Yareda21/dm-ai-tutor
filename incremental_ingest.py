"""
incremental_ingest.py

Idempotent ingestion:
- materials.json: list of {path, courseId, moduleId, language}
- computes stable file hash and stable chunk ids
- if file changed, marks old chunks deprecated and deletes from Pinecone
- upserts new vectors and updates Firestore manifests
"""

import os, json, time, hashlib
from pathlib import Path
import docx2txt
import pdfplumber
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import firebase_admin
from firebase_admin import credentials, firestore

# ---------- CONFIG ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "dm-digital-marketing"  # ensure correct index
FIREBASE_SERVICE_ACCOUNT = "firebase-service-account.json"
MATERIALS_JSON = "materials.json"
EMBEDDING_MODEL = "intfloat/e5-small-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 50
# ----------------------------

# Init Firebase
cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Init embedding model
embed_model = SentenceTransformer(EMBEDDING_MODEL)


# ---------- Helpers ----------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def file_hash(path: Path) -> str:
    b = path.read_bytes()
    return sha256_bytes(b)

def file_id_from_path(path: str) -> str:
    # stable ID for manifest doc: hash of absolute path
    return sha256_text(str(Path(path).resolve()))

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".docx":
        return docx2txt.process(str(path)) or ""
    elif ext == ".pdf":
        parts = []
        with pdfplumber.open(str(path)) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    parts.append(t)
        return "\n".join(parts)
    else:
        try:
            return path.read_text(encoding="utf-8")
        except:
            return path.read_text(encoding="latin-1")

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def stable_chunk_id(file_hash_str: str, idx: int) -> str:
    # deterministic per file content; when file changes hash, chunk ids change
    return sha256_text(f"{file_hash_str}:{idx}")

def mark_old_chunks_deprecated(courseId: str, path: str, new_manifest_version: int):
    # find manifests for this source path with version < new_manifest_version
    manifests_ref = db.collection("courses").document(courseId).collection("materialsManifest")
    q = manifests_ref.where("source", "==", path).stream()
    to_delete_ids = []
    for doc in q:
        m = doc.to_dict()
        if m.get("version", 0) < new_manifest_version:
            old_chunk_ids = m.get("chunkIds", [])
            # mark deprecated in materialsMeta
            for cid in old_chunk_ids:
                db.collection("courses").document(courseId).collection("materialsMeta").document(cid).update({
                    "deprecated": True,
                    "deprecatedAt": firestore.SERVER_TIMESTAMP
                })
            to_delete_ids.extend(old_chunk_ids)
            # Optionally delete the old manifest doc
            # doc.reference.update({"deprecated": True})
    # Delete vectors from Pinecone if any
    if to_delete_ids:
        print(f"Deleting {len(to_delete_ids)} old vectors from Pinecone...")
        try:
            index.delete(deleteRequest={"ids": to_delete_ids})
        except Exception as e:
            print("Warning: Pinecone delete failed:", e)

# ---------- Main ingestion ----------
def main():
    materials = json.load(open(MATERIALS_JSON, "r", encoding="utf-8"))
    for mat in materials:
        path = Path(mat["path"])
        courseId = mat.get("courseId", "digital-marketing")
        moduleId = mat.get("moduleId", "unknown")
        language = mat.get("language", "en")

        if not path.exists():
            print(f"[SKIP] {path} does not exist")
            continue

        fid = file_id_from_path(str(path))
        fhash = file_hash(path)
        manifest_ref = db.collection("courses").document(courseId).collection("materialsManifest").document(fid)
        manifest_doc = manifest_ref.get()
        existing = manifest_doc.to_dict() if manifest_doc.exists else None

        if existing and existing.get("file_hash") == fhash:
            print(f"[SKIP] No change: {path}")
            continue  # idempotent: nothing to do for unchanged file

        # New or changed file
        new_version = (existing.get("version", 0) + 1) if existing else 1
        print(f"\n[PROCESS] {path} (course={courseId}, module={moduleId}, lang={language}) version={new_version}")

        text = extract_text(path)
        chunks = chunk_text(text)
        print(f" → {len(chunks)} chunks")

        # If file changed, mark old chunks deprecated & delete
        if existing:
            mark_old_chunks_deprecated(courseId, str(path), new_version)

        # Prepare batches
        batch_vectors = []
        batch_meta_pairs = []  # list of tuples (chunkId, metadata)
        chunk_ids = []

        for i, chunk in enumerate(chunks):
            cid = stable_chunk_id(fhash, i)
            chunk_ids.append(cid)
            emb = embed_model.encode(chunk).tolist()

            batch_vectors.append({"id": cid, "values": emb})
            meta = {
                "pineconeId": cid,
                "source": str(path),
                "moduleId": moduleId,
                "language": language,
                "text": chunk[:2000],
                "file_hash": fhash,
                "version": new_version,
                "deprecated": False
            }
            batch_meta_pairs.append((cid, meta))

            # flush batches
            if len(batch_vectors) >= BATCH_SIZE or i == len(chunks)-1:
                upsert_req = {"vectors": batch_vectors}
                try:
                    index.upsert(upsertRequest=upsert_req)
                except Exception as e:
                    print("Pinecone upsert failed:", e)
                    raise

                # write metadata to Firestore
                batch = db.batch()
                for cid2, md in batch_meta_pairs:
                    doc_ref = db.collection("courses").document(courseId).collection("materialsMeta").document(cid2)
                    batch.set(doc_ref, md, merge=True)
                batch.commit()

                batch_vectors = []
                batch_meta_pairs = []
                time.sleep(0.2)  # avoid rate limiting

        # Save manifest
        manifest = {
            "file_path": str(path),
            "file_hash": fhash,
            "version": new_version,
            "chunkCount": len(chunks),
            "chunkIds": chunk_ids,
            "moduleId": moduleId,
            "language": language,
            "source": str(path),
            "updatedAt": firestore.SERVER_TIMESTAMP
        }
        manifest_ref.set(manifest, merge=True)
        print(f"[DONE] {path} → version {new_version} uploaded ({len(chunks)} chunks)")

    print("\nAll files processed.")

if __name__ == "__main__":
    main()

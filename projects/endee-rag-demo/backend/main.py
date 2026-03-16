from typing import List, Dict, Any

import uuid
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from . import endee_client


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponseItem(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    results: List[QueryResponseItem]


app = FastAPI(title="Endee RAG Demo", version="0.1.0")

# Configure file-based logging (backend.log in project root, rotated at 1 MB)
log_path = Path(__file__).resolve().parent.parent / "backend.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
root_logger = logging.getLogger()
if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == str(log_path) for h in root_logger.handlers):
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

logger = logging.getLogger("endee_rag_demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """
    Initialize embedding model and ensure Endee collection exists.
    """
    app.state.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    await endee_client.ensure_collection()


def _chunk_text(text: str, max_chars: int = 500) -> List[str]:
    """
    Naive text chunking by characters.
    """
    text = text.strip()
    chunks: List[str] = []
    for i in range(0, len(text), max_chars):
        chunk = text[i : i + max_chars].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


async def _upsert_vectors(vectors: List[List[float]], texts: List[str]) -> None:
    """
    Store vectors and their texts in Endee.
    """
    points = []
    for idx, (vec, txt) in enumerate(zip(vectors, texts)):
        point = {
            "id": str(uuid.uuid4()),
            "vector": vec,
            "meta": {"text": txt},
        }
        logger.info("Prepared point %d for upsert: id=%s, text_preview=%r", idx, point["id"], txt[:80])
        points.append(point)

    await endee_client.upsert_points(points)


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    source: str = Form("uploaded_file"),
) -> Dict[str, Any]:
    """
    Ingest a text file into Endee.
    """
    raw_bytes = await file.read()
    text = raw_bytes.decode("utf-8", errors="ignore")
    chunks = _chunk_text(text)
    logger.info("Ingesting file '%s', %d chunks", file.filename, len(chunks))

    model: SentenceTransformer = app.state.model
    embeddings = model.encode(chunks, convert_to_numpy=True).tolist()

    # Attach source info into text payload.
    decorated_chunks = [f"[{source}] {c}" for c in chunks]
    await _upsert_vectors(embeddings, decorated_chunks)

    return {"status": "ok", "chunks_ingested": len(chunks)}


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    """
    Semantic search over ingested chunks using Endee.
    """
    logger.info("Received query: %r (top_k=%d)", body.query, body.top_k)
    model: SentenceTransformer = app.state.model
    query_vec = model.encode([body.query], convert_to_numpy=True)[0].tolist()

    data = await endee_client.search(query_vec, body.top_k)
    logger.info("Raw search response from Endee: %s", data)

    # Endee's MessagePack result is expected to be a mapping like:
    # {"results": [{"id": "...", "similarity": 0.0, "meta": {...}}, ...]}
    raw_results = data.get("results", []) if isinstance(data, dict) else []
    logger.info("Decoded %d hits from Endee", len(raw_results))

    results: List[QueryResponseItem] = []
    for i, hit in enumerate(raw_results):
        meta = hit.get("meta") or {}
        if isinstance(meta, dict):
            text = str(meta.get("text", ""))
        else:
            text = str(meta)

        score = float(hit.get("similarity", 0.0))
        logger.info("Hit %d: score=%.4f text_preview=%r", i, score, text[:80])
        results.append(QueryResponseItem(text=text, score=score))

    return QueryResponse(results=results)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


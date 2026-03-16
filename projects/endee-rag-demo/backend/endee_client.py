from typing import Any, Dict, List

import os

import httpx


ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
# Use a fresh index name that clearly matches 384-dim vectors,
# so we don't conflict with any older 768-dim "documents" index.
INDEX_NAME = os.getenv("ENDEE_INDEX", "documents_384")
# MiniLM-L6-v2 outputs 384-dim vectors
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


async def ensure_collection() -> None:
    """
    Ensure a dense index exists in Endee via HTTP API.
    """
    payload = {
        "index_name": INDEX_NAME,
        "dim": EMBEDDING_DIM,
        "space_type": "cosine",
    }

    async with httpx.AsyncClient(base_url=ENDEE_URL, timeout=15.0) as client:
        # If index already exists, server will respond with an error; we ignore it.
        try:
            await client.post("/api/v1/index/create", json=payload)
        except Exception:
            pass


async def upsert_points(points: List[Dict[str, Any]]) -> None:
    """
    Upsert a batch of points into the Endee index via HTTP API.

    Each point is expected to be:
    {
        "id": "<str>",
        "vector": [float, ...],
        "meta": "..."              # stored as plain string
    }
    """
    # Endee HTTP API expects either a single object or a list.
    async with httpx.AsyncClient(base_url=ENDEE_URL, timeout=20.0) as client:
        # Transform points so that `meta` is always a string, as expected by the HTTP API.
        wire_points: List[Dict[str, Any]] = []
        for p in points:
            meta_val = p.get("meta", "")
            if isinstance(meta_val, dict):
                # If we ever pass a dict, just keep the "text" field or stringify.
                meta_val = meta_val.get("text", str(meta_val))
            else:
                meta_val = str(meta_val)

            wire_points.append(
                {
                    "id": p.get("id"),
                    "vector": p.get("vector", []),
                    "meta": meta_val,
                }
            )

        await client.post(
            f"/api/v1/index/{INDEX_NAME}/vector/insert",
            json=wire_points,
        )


async def search(vector: List[float], top_k: int) -> Dict[str, Any]:
    """
    Run a similarity search via HTTP API and normalize the response.

    Endee returns a MessagePack-encoded list of hits, where each hit is:
        [score, id, meta_bytes, filter, vector_index, sparse_ids]

    We normalize this into:
        {"results": [{"id": ..., "similarity": ..., "meta": ...}, ...]}
    """
    body = {
        "k": top_k,
        "vector": vector,
    }

    async with httpx.AsyncClient(base_url=ENDEE_URL, timeout=20.0) as client:
        resp = await client.post(
            f"/api/v1/index/{INDEX_NAME}/search",
            json=body,
        )
        resp.raise_for_status()
        # Endee returns MessagePack bytes.
        import msgpack  # local import to avoid global dependency if unused

        raw = msgpack.unpackb(resp.content, raw=False)

        results: List[Dict[str, Any]] = []
        # Expected raw structure: list of lists
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, list) or len(item) < 3:
                    continue
                score = float(item[0])
                vec_id = str(item[1])
                meta_bytes = item[2]
                try:
                    text = meta_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    text = str(meta_bytes)

                results.append(
                    {
                        "id": vec_id,
                        "similarity": score,
                        "meta": text,
                    }
                )

        return {"results": results}



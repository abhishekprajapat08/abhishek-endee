## Endee RAG Demo – Semantic Search on Your Documents

This project is a **simple RAG (Retrieval-Augmented Generation) style demo** that uses the **Endee vector database** as its vector store.

You can:

- **Ingest text files** into Endee (they are chunked and embedded using a local SentenceTransformer model).
- **Run semantic search queries** against the stored chunks via Endee.

> Note: This repo focuses on the **backend + Endee integration**. A lightweight frontend or Jupyter notebook client can be added easily on top of the existing HTTP API.

---

## 1. Project Overview & Problem Statement

**Problem**: Given a collection of unstructured text documents (notes, articles, documentation), how can we **search them semantically**, not just by keywords?

**Solution**: Use an embedding model to convert text into dense vectors, store them in **Endee** as the vector database, and then query by similarity. This is a core building block for:

- Question answering over documents (RAG)
- Semantic search
- Recommendation systems

This project implements:

- A small **FastAPI** backend with:
  - `POST /ingest` – upload a text file, chunk it, embed it with SentenceTransformers, and store vectors in an Endee index.
  - `POST /query` – send a natural language query and get back the closest chunks from Endee with similarity scores.
  - `GET /health` – simple health check.

---

## 2. System Design / Technical Approach

### 2.1 High-Level Architecture

- **Client** (curl / Postman / simple frontend):
  - Uploads text files for ingestion.
  - Sends queries to search the ingested data.

- **Backend (FastAPI, `backend/main.py`)**:
  - Uses **SentenceTransformers** (`all-MiniLM-L6-v2`) to create 384-dimensional embeddings.
  - Talks to **Endee** over HTTP to:
    - Create a dense index (if it does not exist).
    - Upsert vectors with metadata into the index.
    - Run similarity search and decode MessagePack results.

- **Endee (vector database)**:
  - Runs as a separate service (Docker).
  - Exposes a REST API on `http://localhost:8080` by default.

### 2.2 Data Flow

**Ingestion (`POST /ingest`)**:

1. User uploads a text file.
2. Backend reads content and splits it into **fixed-size character chunks** (simple character-based chunking).
3. Each chunk is embedded using SentenceTransformers into a **384-dimensional** vector.
4. For each chunk, a point (`id`, `vector`, `meta`) is upserted into an Endee index named `documents_384`.

**Query (`POST /query`)**:

1. User sends a natural-language `query` and optional `top_k`.
2. Backend encodes the query into a 384-dim vector using the same embedding model.
3. Backend calls Endee's **search** endpoint for the `documents_384` index to find the most similar vectors.
4. The raw MessagePack response is decoded into a list of hits (`id`, `similarity`, `meta`), and the backend returns the matching text chunks and their similarity scores in a JSON-friendly format.

---

## 3. How Endee Is Used

The backend expects Endee to be available via HTTP (default: `http://localhost:8080`) and uses the **Endee OSS HTTP API** to manage a dense index:

- **Create index** (on startup):

  - `POST /api/v1/index/create`
  - JSON body:

    ```json
    {
      "index_name": "documents_384",
      "dim": 384,
      "space_type": "cosine"
    }
    ```

- **Insert vectors**:

  - `POST /api/v1/index/documents_384/vector/insert`
  - JSON body (single object or list):

    ```json
    [
      {
        "id": "uuid",
        "vector": [0.1, 0.2, ...],
        "meta": "original text chunk"
      }
    ]
    ```

- **Search**:

  - `POST /api/v1/index/documents_384/search`
  - JSON body:

    ```json
    {
      "k": 5,
      "vector": [0.1, 0.2, ...]
    }
    ```

- **Response**:

  - Endee returns a **MessagePack-encoded** list of hits. The backend decodes and normalizes it into:

    ```json
    {
      "results": [
        {
          "text": "...chunk text...",
          "score": 0.48
        }
      ]
    }
    ```

---

## 4. Setup & Running the Project

### 4.1 Prerequisites

- Python 3.10+ recommended
- Docker (to run Endee)
- Git

### 4.2 Clone and Star the Official Endee Repo

- Go to the official Endee GitHub repository and **star it**.
- Optionally, in a separate folder, clone the Endee repo and use its Docker setup to run the Endee server.

### 4.3 Run Endee (Vector Database)

Below is an example using the Docker image produced from the `infra/Dockerfile` of the official Endee repo.

From the Endee repo:

```bash
docker build -t endee-oss -f infra/Dockerfile .

docker run --rm -p 8080:8080 -v endee_data:/data --name endee-oss endee-oss
```

This exposes Endee on `http://localhost:8080` with the health check at:

```bash
curl http://localhost:8080/api/v1/health
```

Make sure this returns a healthy response before starting the backend.

### 4.4 Create and Activate a Virtual Environment

```bash
cd endee-rag-demo
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 4.5 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The first run may download the SentenceTransformers model.

### 4.6 Environment Variables

You can customize the connection to Endee using environment variables:

- `ENDEE_URL` – default `http://localhost:8080`
- `ENDEE_INDEX` – default `documents_384`
- `EMBEDDING_DIM` – default `384`

On Windows (PowerShell), for example:

```powershell
$env:ENDEE_URL="http://localhost:8080"
$env:ENDEE_INDEX="documents_384"
```

---

## 5. Running the Backend

From the project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

The FastAPI docs UI will be available at:

- `http://localhost:8000/docs`

---

## 6. Example Usage

### 6.1 Ingest a Text File

Use curl or Postman to upload a `.txt` file:

```bash
curl -X POST "http://localhost:8000/ingest" ^
  -F "file=@sample.txt" ^
  -F "source=sample-doc"
```

Expected JSON response:

```json
{
  "status": "ok",
  "chunks_ingested": 3
}
```

### 6.2 Query

```bash
curl -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What is Endee used for?\", \"top_k\": 5}"
```

Example JSON response (truncated for readability):

```json
{
  "results": [
    {
      "text": "[uploaded_file] ... This shows how to connect an AI or LLM application to Endee as the vector store...",
      "score": 0.48
    },
    {
      "text": "[uploaded_file] Endee is a high-performance vector database designed for semantic search, recommendations, and retrieval-augmented generation (RAG) workflows...",
      "score": 0.41
    }
  ]
}
```

---

## 7. Extending the Project

- **LLM Answering (full RAG)**:
  - Take the top retrieved chunks from `/query` and feed them, along with the user question, into an LLM (OpenAI, local LLM, etc.) to generate final answers with citations.

- **Frontend**:
  - Build a minimal React or plain HTML/JS frontend to:
    - Upload files
    - Ask questions
    - Display ranked chunks or LLM answers

- **Index Management**:
  - Add endpoints to list or delete documents / collections.

---

## 8. Submission Notes (for Internship Evaluation)

When using this project for the Endee internship:

- Make sure your GitHub repository:
  - Is public
  - Has this README (or an expanded version)
  - Clearly mentions **Endee** as the vector database.
- Optionally:
  - Add screenshots of your `/docs` page, ingestion, and query examples.
  - Add a simple diagram (PNG or Markdown) of the architecture.

Then share the GitHub repository link as required in the application or evaluation form.


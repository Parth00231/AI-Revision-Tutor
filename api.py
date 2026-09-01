#!/usr/bin/env python3
"""
FastAPI REST API for the AI Course RAG Assistant.

Endpoints:
  GET  /          - Serves the Single Page Application UI (HTML)
  GET  /health    - Health check / backend readiness JSON
  GET  /topics    - List all indexed video sources with chunk stats
  POST /ask       - RAG Q&A — returns answer + sources
  POST /notes     - Generate structured study notes on a topic
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from groq import Groq

import embed_and_upload

# ============================================================
# STARTUP & SHARED STATE
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "openai/gpt-oss-120b"

app = FastAPI(
    title="AI Course RAG Assistant",
    description="Ask questions and generate notes from 23 AI course video transcripts.",
    version="1.1.0",
)

# Enable CORS for all origins to ensure seamless local testing & deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded singletons (initialised on first request to keep startup fast)
_qdrant_client = None
_embed_model   = None
_groq_client   = None
_topics_cache  = None


def get_clients():
    """Return (qdrant_client, embed_model, groq_client), initialising lazily."""
    global _qdrant_client, _embed_model, _groq_client

    if _qdrant_client is None:
        _qdrant_client = embed_and_upload.get_qdrant_client()

    if _embed_model is None:
        print("🧠 Loading embedding model...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model ready!")

    if _groq_client is None and GROQ_API_KEY:
        _groq_client = Groq(api_key=GROQ_API_KEY)

    return _qdrant_client, _embed_model, _groq_client


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


class NotesRequest(BaseModel):
    topic: str
    top_k: Optional[int] = 7


class NotesResponse(BaseModel):
    topic: str
    notes: str
    sources: list[dict]


class TopicsResponse(BaseModel):
    count: int
    topics: list[str]


# ============================================================
# HELPER — build context string from search results
# ============================================================

def build_context(results) -> str:
    parts = []
    for r in results:
        text   = r.payload.get("text", "")
        source = r.payload.get("source", "")
        chunk_id = r.payload.get("chunk_id", "")
        parts.append(f"Source: {source} (Chunk {chunk_id})\nContent:\n{text}")
    return "\n\n---\n\n".join(parts)


def format_sources(results) -> list[dict]:
    return [
        {
            "source":   r.payload.get("source", ""),
            "chunk_id": r.payload.get("chunk_id", ""),
            "score":    round(float(r.score), 4),
            "preview":  r.payload.get("text", "")[:300],
            "full_text": r.payload.get("text", ""),
        }
        for r in results
    ]


# ============================================================
# ROUTES
# ============================================================

@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint, returns JSON status and Qdrant readiness."""
    collection_ready = False
    points_count = 0
    try:
        client, _, _ = get_clients()
        collection_ready = client.collection_exists(embed_and_upload.COLLECTION_NAME)
        if collection_ready:
            info = client.get_collection(embed_and_upload.COLLECTION_NAME)
            points_count = info.points_count or 0
    except Exception as e:
        print(f"Health check error: {e}")

    return {
        "status": "ok",
        "collection": embed_and_upload.COLLECTION_NAME,
        "collection_ready": collection_ready,
        "points_count": points_count,
        "groq_configured": bool(GROQ_API_KEY),
    }


@app.get("/topics", response_model=TopicsResponse, tags=["Knowledge"])
def list_topics():
    """Return all unique video sources currently indexed in Qdrant."""
    global _topics_cache
    if _topics_cache:
        return TopicsResponse(count=len(_topics_cache), topics=_topics_cache)

    client, model, _ = get_clients()

    if not client.collection_exists(embed_and_upload.COLLECTION_NAME):
        raise HTTPException(
            status_code=503,
            detail="Collection not found. Run 'python main.py --embed' first.",
        )

    # Scroll through all points and collect unique sources
    sources = set()
    offset  = None
    while True:
        response = client.scroll(
            collection_name=embed_and_upload.COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=["source"],
            with_vectors=False,
        )
        points, next_offset = response
        for point in points:
            src = point.payload.get("source", "")
            if src:
                sources.add(src)
        if next_offset is None:
            break
        offset = next_offset

    def sort_key(s: str):
        # Sort naturally by number in filename (e.g., "1. intro", "2. ...", "10. ...")
        import re
        m = re.match(r"^(\d+)", s)
        return int(m.group(1)) if m else 999

    sorted_sources = sorted(list(sources), key=sort_key)
    _topics_cache = sorted_sources
    return TopicsResponse(count=len(sorted_sources), topics=sorted_sources)


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask_question(req: AskRequest):
    """RAG Q&A: retrieve relevant context from Qdrant, then answer with Groq LLM."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    client, model, groq_client = get_clients()

    if not client.collection_exists(embed_and_upload.COLLECTION_NAME):
        raise HTTPException(
            status_code=503,
            detail="Collection not found. Run 'python main.py --embed' first.",
        )

    # Vector search
    results = embed_and_upload.search(client, model, req.question, top_k=req.top_k)
    if not results:
        return AskResponse(
            question=req.question,
            answer="No relevant content found in the course video transcripts.",
            sources=[],
        )

    context = build_context(results)

    if groq_client:
        try:
            answer = embed_and_upload.ask_llm(groq_client, req.question, context)
        except Exception as e:
            answer = f"Error communicating with LLM ({e}). Here is the most relevant lecture content:\n\n" + results[0].payload.get("text", "")
    else:
        # Fallback: return raw top match
        answer = results[0].payload.get("text", "GROQ_API_KEY not configured. Displaying top transcript match.")

    return AskResponse(
        question=req.question,
        answer=answer,
        sources=format_sources(results),
    )


@app.post("/notes", response_model=NotesResponse, tags=["Notes"])
def generate_notes(req: NotesRequest):
    """Generate structured study notes on a given topic using RAG + Groq."""
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    client, model, groq_client = get_clients()

    if not client.collection_exists(embed_and_upload.COLLECTION_NAME):
        raise HTTPException(
            status_code=503,
            detail="Collection not found. Run 'python main.py --embed' first.",
        )

    # Search with query enriched for study notes
    search_query = f"explain {req.topic} concepts key ideas mechanisms examples details"
    results = embed_and_upload.search(client, model, search_query, top_k=req.top_k)

    if not results:
        return NotesResponse(
            topic=req.topic,
            notes="No relevant content found for this topic in the course material.",
            sources=[],
        )

    context = build_context(results)

    if groq_client:
        notes_prompt = f"""You are an expert AI tutor creating comprehensive study notes from course material.

Generate well-structured, detailed study notes for the topic: **{req.topic}**

Use ONLY the provided course context below. Format your notes in clean Markdown with:
- A clear `# {req.topic}` title
- `## 📌 Key Concepts` section with bullet points and clear explanations
- `## ⚙️ How It Works` section explaining the inner mechanics step-by-step
- `## 🔍 Important Details & Terminology` with formulas, rules, and best practices
- `## 💡 Examples & Applications` from the course lectures
- `## 🎯 Summary & Quick Revision Checklist` with 3-5 key takeaways

Context from course video lectures:
{context}

Rules:
1. Only use information from the context above.
2. Write in clear, concise language suited for rapid revision.
3. Use Markdown formatting (bold terms, bullet points, code blocks where applicable).
4. If information for a section is absent from context, adapt gracefully.
5. Do NOT hallucinate or add facts not supported by the lecture transcripts.
"""
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": notes_prompt}],
                temperature=0.1,
            )
            notes_text = response.choices[0].message.content.strip()
        except Exception as e:
            notes_text = f"Failed to generate AI notes ({e}). Context summary:\n\n" + context[:1200]
    else:
        notes_text = "GROQ_API_KEY not configured in environment. Please provide a key in `.env` to generate notes."

    return NotesResponse(
        topic=req.topic,
        notes=notes_text,
        sources=format_sources(results),
    )


# ============================================================
# STATIC FILE SERVING & SPA ROOT
# ============================================================

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static folder for assets (css, js, icons)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
def serve_root():
    """Serve the single page application at the root route."""
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"error": "Frontend not found. Make sure static/index.html exists."}


@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa(full_path: str):
    """Serve static files or fall back to SPA index.html for client-side routing."""
    # Check if a specific file exists inside static directory (e.g. style.css, app.js)
    requested_file = static_dir / full_path
    if requested_file.is_file():
        return FileResponse(str(requested_file))
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"error": "Frontend not found. Make sure static/index.html exists."}

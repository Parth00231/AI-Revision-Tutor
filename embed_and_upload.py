# ============================================================
# PART 1 — IMPORTS AND ENVIRONMENT
# ============================================================

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

COLLECTION_NAME = "ai_course"
EMBEDDING_SIZE = 384  # all-MiniLM-L6-v2 size


# ============================================================
# PART 2 — CONNECT TO QDRANT (WITH LOCAL FALLBACK)
# ============================================================

def get_qdrant_client():
    """Connect to Qdrant Cloud or fall back to local disk storage."""
    if QDRANT_URL and QDRANT_URL.lower() not in ("local", "./qdrant_db", ":memory:"):
        try:
            print(f"🔌 Connecting to Qdrant Cloud at {QDRANT_URL}...")
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=10,
                check_compatibility=False
            )
            # Test connectivity
            client.get_collections()
            print("✅ Connected to Qdrant Cloud!")
            return client
        except Exception as e:
            print(f"⚠️ Could not connect to Qdrant Cloud ({e}).")
            print("💡 Falling back to local Qdrant database (./qdrant_db)...")
    
    local_path = Path("qdrant_db")
    local_path.mkdir(exist_ok=True)
    client = QdrantClient(path=str(local_path))
    print(f"✅ Connected to Local Qdrant database ({local_path.resolve()})!")
    return client


# ============================================================
# PART 3 — LOAD KNOWLEDGE CHUNKS
# ============================================================

def load_documents():
    """Find and load chunk JSON files from ./chunks or ./transcripts/chunks."""
    chunks_folder = Path("chunks")
    if not chunks_folder.exists() or not list(chunks_folder.glob("*.json")):
        alt_folder = Path("transcripts/chunks")
        if alt_folder.exists() and list(alt_folder.glob("*.json")):
            chunks_folder = alt_folder

    json_files = sorted(chunks_folder.glob("*.json"))
    print(f"\n📚 Found {len(json_files)} JSON chunk files in '{chunks_folder}'")

    documents = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        source = data.get("source", json_file.name)
        for chunk in data.get("chunks", []):
            documents.append({
                "text": chunk["text"],
                "source": source,
                "chunk_id": chunk["chunk_id"]
            })

    print(f"🔢 Total chunks loaded: {len(documents)}")
    return documents


# ============================================================
# PART 4 — CREATE EMBEDDINGS & UPLOAD TO QDRANT
# ============================================================

def index_documents(client, documents, model):
    """Create collection, generate embeddings, and upload to Qdrant."""
    if not documents:
        print("⚠️ No documents to index. Please run chunk.py first.")
        return

    # Delete existing collection if present
    if client.collection_exists(COLLECTION_NAME):
        print(f"🗑️ Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    # Create new collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"✅ Created collection '{COLLECTION_NAME}' (Vector size: {EMBEDDING_SIZE}, COSINE)")

    # Create embeddings
    texts = [doc["text"] for doc in documents]
    print("\n🔄 Creating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"✅ Generated {len(embeddings)} embeddings")

    # Create Qdrant points
    points = []
    for i, doc in enumerate(documents):
        point = PointStruct(
            id=i + 1,
            vector=embeddings[i].tolist(),
            payload={
                "text": doc["text"],
                "source": doc["source"],
                "chunk_id": doc["chunk_id"]
            }
        )
        points.append(point)

    print("\n⬆️ Uploading vectors to Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )
    print(f"🎉 Uploaded {len(points)} chunks to Qdrant!")


# ============================================================
# PART 5 — SEARCH QDRANT
# ============================================================

def search(client, model, query, top_k=3):
    """Convert question to vector and search Qdrant."""
    query_vector = model.encode(query).tolist()
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return results


# ============================================================
# PART 6 — ASK THE LLM VIA GROQ
# ============================================================

def ask_llm(groq_client, question, context):
    """Send retrieved context and question to Groq LLM."""
    if not GROQ_API_KEY:
        return "GROQ_API_KEY missing in environment."

    prompt = f"""You are an AI assistant for an AI course.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Rules:
1. Do not make up information.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
"I don't know based on the provided course material."
4. Give a clear and concise answer.
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================

def main():
    # 1. Connect to Qdrant
    client = get_qdrant_client()

    # 2. Load Documents
    documents = load_documents()
    if not documents:
        print("🛑 Pipeline stopped: No document chunks available.")
        return

    # 3. Load Embedding Model
    print("\n🧠 Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model ready!")

    # 4. Index Documents in Qdrant
    index_documents(client, documents, model)

    # 5. Connect to Groq
    if not GROQ_API_KEY:
        print("⚠️ GROQ_API_KEY missing in .env. LLM responses will be disabled.")
        groq_client = None
    else:
        groq_client = Groq(api_key=GROQ_API_KEY)

    # 6. Test Search & LLM Query
    question = "What is RAG and how does it work?"
    print(f"\n❓ Question: '{question}'")

    results = search(client, model, question, top_k=5)

    print("\n🔍 Top Search Results:")
    for result in results[:3]:
        print("\n" + "=" * 60)
        print(f"Score: {result.score:.4f} | Source: {result.payload['source']} (chunk {result.payload['chunk_id']})")
        print(f"\n{result.payload['text']}")

    # Build Context
    context_parts = []
    for result in results:
        text = result.payload.get("text", "")
        source = result.payload.get("source", "")
        context_parts.append(f"Source: {source}\nContent: {text}")
    context = "\n\n---\n\n".join(context_parts)

    # Ask LLM
    if groq_client:
        print("\n" + "=" * 60)
        print("🤖 FINAL ANSWER")
        print("=" * 60)
        answer = ask_llm(groq_client, question, context)
        print(answer)

        print("\n" + "=" * 60)
        print("📚 SOURCES USED")
        print("=" * 60)
        for result in results:
            print(f"- {result.payload.get('source')} (chunk {result.payload.get('chunk_id')}) [score: {result.score:.4f}]")


if __name__ == "__main__":
    main()
# 🧠 AI Revision Tutor

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-dc2626.svg?style=for-the-badge)](https://qdrant.tech)
[![Groq](https://img.shields.io/badge/Groq-Fast_Inference-f97316.svg?style=for-the-badge)](https://groq.com)

An intelligent Retrieval-Augmented Generation (**RAG**) revision portal and study assistant built over transcripts from **23 AI Course Video Lectures** (~455 vectorized knowledge chunks).

---

## ✨ Features

- 🔍 **RAG Q&A with Citations**: Ask any question about course concepts and receive grounded, accurate answers backed by lecture transcripts.
- 📝 **Structured Study Notes Generator**: Automatically generate comprehensive revision notes with key concepts, how-it-works breakdowns, terminology, and revision checklists.
- 🎬 **Transcript Inspector Modal**: Click on any source badge to inspect the exact transcript chunk with its cosine similarity score.
- ⚡ **Qdrant Vector Database**: High-performance semantic vector search using `sentence-transformers/all-MiniLM-L6-v2` embeddings (384 dimensions).
- 🚀 **High-Speed Inference with Groq**: Powered by `openai/gpt-oss-120b` for rapid answer generation.
- 🎨 **Dark Glassmorphism Single Page Application**: Responsive modern UI featuring ambient glow orbs, lecture filter search, Markdown formatting, clipboard copy, and `.md` file download.

---

## 🏗️ Architecture & Pipeline

```
Video Transcripts (.txt)
        │
        ▼
   [ chunk.py ] ─── Recursive Character Text Splitter (500 chars / 50 overlap)
        │
        ▼
  JSON Chunks (455 total)
        │
        ▼
[ embed_and_upload.py ] ─── SentenceTransformer ("all-MiniLM-L6-v2")
        │
        ▼
   Qdrant Cloud / Local Vector DB (Cosine Similarity)
        │
   ┌────┴───────────────────────────┐
   ▼                                ▼
[ /ask Endpoint ]           [ /notes Endpoint ]
   │ (Top-5 Vector Search)         │ (Top-7 Concept Search)
   ▼                               ▼
Context + Prompt ──────────► Groq LLM (gpt-oss-120b) ──────────► Clean Revision Output
```

---

## 📁 Repository Structure

```
.
├── api.py                  # FastAPI server with /health, /topics, /ask, /notes & SPA routes
├── server.py               # Uvicorn entry point (runs on http://localhost:8001)
├── embed_and_upload.py     # Qdrant client, embedding generator, search & LLM caller
├── chunk.py                # Text splitting logic for transcript files
├── transcribe.py           # Whisper transcription script for lecture audio/video
├── main.py                 # Interactive CLI interface
├── static/
│   ├── index.html          # SPA HTML structure
│   ├── style.css           # Glassmorphism styling and responsive layout
│   └── app.js              # State management, API integration, and marked parser
├── transcripts/            # 23 original lecture transcript files (.txt)
├── chunks/                 # 23 chunked JSON datasets
├── requirements.txt        # Python package dependencies
├── pyproject.toml          # Project configuration
└── .env.example            # Environment variable template
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- A [Qdrant Cloud](https://cloud.qdrant.io) cluster (or local Qdrant instance)
- A [Groq API Key](https://console.groq.com)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/Parth00231/AI-Revision-Tutor.git
cd AI-Revision-Tutor

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```env
QDRANT_URL=https://your-qdrant-cluster-url.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=gsk_your_groq_api_key
```

### 4. Index the Lecture Knowledge Base (One-Time)

To split transcripts, generate embeddings, and upload to Qdrant:

```bash
# Chunk the transcripts
python chunk.py

# Embed and upload to Qdrant
python embed_and_upload.py
```

### 5. Launch the Application

```bash
# Start the FastAPI web server
python server.py
# or using uv:
uv run python server.py
```

Open your browser and navigate to:
👉 **`http://localhost:8001`**

---

## 💡 API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend single page application |
| `GET` | `/health` | Backend status & Qdrant collection readiness |
| `GET` | `/topics` | Returns all 23 indexed lecture titles |
| `POST` | `/ask` | Semantic search + Groq LLM answer with source citations |
| `POST` | `/notes` | Generates structured Markdown revision notes |

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

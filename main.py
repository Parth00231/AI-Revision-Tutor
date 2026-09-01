#!/usr/bin/env python3
"""
AI Course RAG Assistant - Main Entry Point

Usage:
  python main.py                  # Interactive Q&A loop
  python main.py --query "What is RAG?"
  python main.py --embed          # Embed chunks and upload to vector DB
  python main.py --chunk          # Chunk transcripts into JSON
  python main.py --transcribe     # Transcribe MP3 audio files
  python main.py --run-all        # Run complete pipeline (transcribe -> chunk -> embed)
"""

import sys
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from groq import Groq

# Import core pipeline functions
import embed_and_upload


def run_transcribe():
    """Run transcription stage."""
    print("\n==========================================")
    print("🎧 STEP 1: TRANSCRIBE AUDIO FILES")
    print("==========================================")
    os.system(f"{sys.executable} transcribe.py")


def run_chunk():
    """Run text chunking stage."""
    print("\n==========================================")
    print("📄 STEP 2: CHUNK TRANSCRIPTS")
    print("==========================================")
    os.system(f"{sys.executable} chunk.py")


def run_embed():
    """Run embedding and Qdrant upload stage."""
    print("\n==========================================")
    print("🧠 STEP 3: EMBED & INDEX DOCUMENTS")
    print("==========================================")
    embed_and_upload.main()


def query_rag(question: str):
    """Query the indexed vector DB and ask Groq LLM."""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    client = embed_and_upload.get_qdrant_client()
    if not client.collection_exists(embed_and_upload.COLLECTION_NAME):
        print(f"⚠️ Collection '{embed_and_upload.COLLECTION_NAME}' not found.")
        print("💡 Please run 'python main.py --embed' first to index your documents.")
        return

    print("🧠 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"\n❓ Question: '{question}'")
    results = embed_and_upload.search(client, model, question, top_k=5)

    if not results:
        print("⚠️ No relevant matches found in vector database.")
        return

    # Build Context
    context_parts = []
    for result in results:
        text = result.payload.get("text", "")
        source = result.payload.get("source", "")
        context_parts.append(f"Source: {source}\nContent: {text}")
    context = "\n\n---\n\n".join(context_parts)

    if not groq_api_key:
        print("\n🔍 Top Search Matches:")
        for r in results[:3]:
            print(f"- [{r.payload.get('source')}] Score: {r.score:.4f}")
            print(f"  {r.payload.get('text')[:200]}...\n")
        return

    groq_client = Groq(api_key=groq_api_key)
    answer = embed_and_upload.ask_llm(groq_client, question, context)

    print("\n" + "=" * 60)
    print("🤖 ANSWER")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("📚 SOURCES USED")
    print("=" * 60)
    for result in results:
        print(f"- {result.payload.get('source')} (chunk {result.payload.get('chunk_id')}) [score: {result.score:.4f}]")


def interactive_loop():
    """Run interactive CLI terminal chat loop."""
    print("==========================================")
    print("🤖 AI Course RAG Interactive Assistant")
    print("==========================================")
    print("Type your question below (or 'exit' / 'quit' to stop).\n")

    while True:
        try:
            user_input = input("❓ Question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Bye! 👋")
                break
            query_rag(user_input)
            print("\n" + "-" * 60 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBye! 👋")
            break


def main():
    parser = argparse.ArgumentParser(description="AI Course RAG Assistant CLI")
    parser.add_argument("--transcribe", action="store_true", help="Transcribe audio files in ./audio or user folder")
    parser.add_argument("--chunk", action="store_true", help="Chunk transcripts into ./chunks")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings and index into Qdrant")
    parser.add_argument("--run-all", action="store_true", help="Run full pipeline: transcribe -> chunk -> embed")
    parser.add_argument("--query", type=str, help="Ask a single question to the RAG system")

    args = parser.parse_args()

    if args.transcribe:
        run_transcribe()
    elif args.chunk:
        run_chunk()
    elif args.embed:
        run_embed()
    elif args.run_all:
        run_transcribe()
        run_chunk()
        run_embed()
    elif args.query:
        query_rag(args.query)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()

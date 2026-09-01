from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# FOLDERS
# =========================

# Determine transcripts directory: check local ./transcripts first
local_transcripts = Path("transcripts")
external_transcripts = Path("/Users/parth/Audio AI Course/transcripts")

if local_transcripts.exists() and list(local_transcripts.glob("*.txt")):
    transcript_folder = local_transcripts
elif external_transcripts.exists():
    transcript_folder = external_transcripts
else:
    transcript_folder = local_transcripts

# Output chunks directly to project root ./chunks directory
output_folder = Path("chunks")
output_folder.mkdir(exist_ok=True)


# =========================
# CHUNKING CONFIGURATION
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# =========================
# PROCESS ALL TRANSCRIPTS
# =========================

for txt_file in transcript_folder.glob("*.txt"):

    print(f"\n📄 Processing: {txt_file.name}")

    # Read transcript
    text = txt_file.read_text(encoding="utf-8")

    # Create chunks
    chunks = text_splitter.split_text(text)

    print(f"🔹 Created {len(chunks)} chunks")


    # =========================
    # CREATE JSON DATA
    # =========================

    data = {
        "source": txt_file.name,
        "chunks": []
    }

    for i, chunk in enumerate(chunks):

        data["chunks"].append({
            "chunk_id": i,
            "text": chunk.strip()
        })


    # =========================
    # SAVE JSON
    # =========================

    output_file = output_folder / f"{txt_file.stem}.json"

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"✅ Saved: {output_file.name}")


print("\n🎉 All transcripts converted into JSON chunks!")
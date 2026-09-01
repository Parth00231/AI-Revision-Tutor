from faster_whisper import WhisperModel
from pathlib import Path

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

# Determine audio folder: check local ./audio first, fall back to external directory if present
local_audio = Path("audio")
external_audio = Path("/Users/parth/Audio AI Course")

if local_audio.exists() and list(local_audio.glob("*.mp3")):
    audio_folder = local_audio
elif external_audio.exists():
    audio_folder = external_audio
else:
    audio_folder = local_audio

output_folder = Path("transcripts")
output_folder.mkdir(exist_ok=True)

audio_files = list(audio_folder.glob("*.mp3"))

if not audio_files:
    print(f"⚠️ No MP3 files found in '{audio_folder.resolve()}'. Please add audio files to transcribe.")
else:
    print(f"📂 Found {len(audio_files)} audio files in '{audio_folder.name}'")
    for audio_file in audio_files:

        output_file = output_folder / f"{audio_file.stem}.txt"

        # Skip already processed files
        if output_file.exists():
            print(f"⏭️ Already done: {audio_file.name}")
            continue

        print(f"\n🎧 Transcribing: {audio_file.name}")

        segments, info = model.transcribe(
            str(audio_file),
            beam_size=5
        )

        text = " ".join(segment.text for segment in segments)

        output_file.write_text(
            text.strip(),
            encoding="utf-8"
        )

        print(f"✅ Saved: {output_file.name}")

    print("\n🎉 All files processed!")
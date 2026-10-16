"""Word-level transcription with faster-whisper.

    TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PY scripts/transcribe.py voice.mp3 \
        --model small --language en --prompt "Claude Code, HyperFrames"

PY is the virtualenv's Python (.venv/bin/python, or .venv\\Scripts\\python on Windows).

Writes transcript.json: every word with its start and end time, in seconds. The
first run downloads the model from Hugging Face (small is about 460 MB); later
runs load it from the local cache and work offline.

  --language  language of the voice; leave it out to auto-detect
  --prompt    names and brand words that appear in the video
  --model     tiny, base, small (default), medium, large-v3; bigger is more
              accurate and slower on a CPU

Import into a HyperFrames project with: npx hyperframes transcribe transcript.json
Subtitles always come from these timings, never from guesses. Karaoke subtitles:
show a short phrase, light each word up at its own start time.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from faster_whisper import WhisperModel  # noqa: E402
from faster_whisper.utils import download_model  # noqa: E402
from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402

MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def model_path(name):
    try:
        return download_model(name, local_files_only=True)
    except LocalEntryNotFoundError:
        pass
    print(f"Downloading the {name} model (first run only)...", file=sys.stderr)
    try:
        return download_model(name)
    except Exception as e:
        sys.exit(f"Could not download the {name} model from Hugging Face: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio or video to word-level timestamps.")
    parser.add_argument("audio", help="audio or video file")
    parser.add_argument("--model", default="small", choices=MODELS)
    parser.add_argument("--language", help="language code of the voice, e.g. en; omit to auto-detect")
    parser.add_argument("--prompt", help="names and brand words to help recognition")
    parser.add_argument("-o", "--output", default="transcript.json", help="default: transcript.json")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        sys.exit(f"No such file: {args.audio}")

    model = WhisperModel(model_path(args.model), device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        args.audio,
        language=args.language,
        initial_prompt=args.prompt,
        word_timestamps=True,
    )

    words = []
    for seg in segments:
        print(f"[{seg.start:7.2f} -> {seg.end:7.2f}] {seg.text.strip()}", file=sys.stderr)
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append({"id": f"w{len(words)}", "text": text, "start": round(w.start, 3), "end": round(w.end, 3)})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {len(words)} words to {args.output} (language: {info.language}, {info.duration:.1f}s)")


if __name__ == "__main__":
    main()

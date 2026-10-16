"""Word-level transcription with Whisper (openai-whisper).

    TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PY scripts/transcribe.py voice.mp3 \
        --model small --language en --prompt "Claude Code, HyperFrames"

PY is the edit-tools Python: ~/.local/edit-tools/.venv/bin/python

Writes transcript.json: every word with its start and end time, in seconds. The
first run downloads the model once (small is about 460 MB, into ~/.cache/whisper);
later runs load it from there and work offline.

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
import urllib.error

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import whisper  # noqa: E402

MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def load_model(name):
    try:
        return whisper.load_model(name)
    except urllib.error.URLError as e:
        sys.exit(f"Could not download the {name} model (from openaipublic.azureedge.net): {e.reason}")


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

    audio = whisper.load_audio(args.audio)
    model = load_model(args.model)
    result = model.transcribe(
        audio,
        language=args.language,
        initial_prompt=args.prompt,
        word_timestamps=True,
        fp16=model.device.type == "cuda",
    )

    words = []
    for seg in result["segments"]:
        print(f"[{seg['start']:7.2f} -> {seg['end']:7.2f}] {seg['text'].strip()}", file=sys.stderr)
        for w in seg.get("words", []):
            text = w["word"].strip()
            if text:
                words.append({"id": f"w{len(words)}", "text": text, "start": round(w["start"], 3), "end": round(w["end"], 3)})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)
        f.write("\n")
    duration = len(audio) / whisper.audio.SAMPLE_RATE
    print(f"Wrote {len(words)} words to {args.output} (language: {result['language']}, {duration:.1f}s)")


if __name__ == "__main__":
    main()

"""Label sound effects with CLAP (zero-shot audio classification).

    TOKENIZERS_PARALLELISM=false PY scripts/listen.py label sfx/*.wav
    PY scripts/listen.py label click.wav whoosh.mp3 --top 3
    PY scripts/listen.py label sfx/*.wav --labels "click, pop, whoosh" --json

PY is the edit-tools Python: ~/.local/edit-tools/.venv/bin/python

Each file is scored against a list of sound labels (the built-in LABELS below,
or --labels) and the best matches are printed with their scores. The first run
downloads the CLAP model from Hugging Face (about 600 MB); later runs load it
from the local cache and work offline. Any format ffmpeg can read works.
"""

import argparse
import json
import os
import subprocess
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import ClapModel, ClapProcessor  # noqa: E402

MODEL = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48000
PROMPT = "This is a sound of {}."
LABELS = [
    "click", "mouse click", "keyboard typing", "pop", "whoosh", "swoosh", "swipe",
    "ding", "chime", "bell", "notification", "beep", "error buzz", "boom", "impact",
    "punch", "thud", "explosion", "glass breaking", "riser", "drum roll", "drum hit",
    "cymbal crash", "applause", "cheering", "laughter", "crowd", "footsteps", "door",
    "camera shutter", "cash register", "coins", "glitch", "static", "record scratch",
    "rewind", "wind", "rain", "thunder", "water splash", "birds", "dog bark",
    "car engine", "siren", "phone ringing", "heartbeat", "magic sparkle",
    "sci-fi laser", "speech", "music",
]


def load_clap(name):
    try:
        return ClapModel.from_pretrained(name, local_files_only=True), ClapProcessor.from_pretrained(name, local_files_only=True)
    except OSError:
        pass
    print(f"Downloading the {name} model (first run only)...", file=sys.stderr)
    try:
        return ClapModel.from_pretrained(name), ClapProcessor.from_pretrained(name)
    except OSError as e:
        sys.exit(f"Could not download {name} from Hugging Face: {str(e).splitlines()[0]}")


def load_audio(path):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise ValueError(out.stderr.decode(errors="replace").strip() or "ffmpeg could not read it")
    audio = np.frombuffer(out.stdout, np.float32).copy()
    if audio.size == 0:
        raise ValueError("no audio")
    return audio


def label(args):
    labels = [s.strip() for s in args.labels.split(",") if s.strip()] if args.labels else LABELS
    model, processor = load_clap(args.model)
    model.eval()
    # Fused models take 4 stacked views of each clip, unfused ones a single view.
    truncation = "fusion" if model.config.audio_config.enable_fusion else "rand_trunc"

    results, failed = [], False
    with torch.no_grad():
        text = processor(text=[PROMPT.format(s) for s in labels], return_tensors="pt", padding=True)
        text_emb = model.get_text_features(**text)
        for path in args.files:
            try:
                audio = load_audio(path)
            except ValueError as e:
                print(f"{path}: {e}", file=sys.stderr)
                failed = True
                continue
            np.random.seed(0)  # the feature extractor crops clips over 10 s at random
            inputs = processor(audios=audio, sampling_rate=SAMPLE_RATE, truncation=truncation, return_tensors="pt")
            audio_emb = model.get_audio_features(**inputs)
            probs = (audio_emb @ text_emb.T * model.logit_scale_a.exp()).softmax(dim=-1)[0]
            top = probs.topk(min(args.top, len(labels)))
            results.append({
                "file": path,
                "labels": [{"label": labels[i], "score": round(float(p), 3)} for p, i in zip(top.values, top.indices)],
            })

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(f"{r['file']}: " + ", ".join(f"{x['label']} {x['score']:.2f}" for x in r["labels"]))
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="Label sound effects with CLAP.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("label", help="score audio files against sound labels")
    p.add_argument("files", nargs="+", help="audio or video files")
    p.add_argument("--labels", help="comma-separated labels to choose from (default: built-in list)")
    p.add_argument("--top", type=int, default=3, help="matches to show per file (default: 3)")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    p.add_argument("--model", default=MODEL, help=f"CLAP model id or local path (default: {MODEL})")
    args = parser.parse_args()
    sys.exit(label(args))


if __name__ == "__main__":
    main()

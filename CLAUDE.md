# Video editing tools

`my-video/` is a HyperFrames project with its own CLAUDE.md. The scripts here
work on audio and 3D for it. Run the Python ones with the edit-tools interpreter:

    PY=~/.local/edit-tools/.venv/bin/python

`scripts/setup-cloud.sh` installs that interpreter's packages, Blender, espeak-ng
and HyperFrames' Chrome. It is meant for the cloud environment's setup script.

| Task | Command |
|---|---|
| Transcribe to word timings (`transcript.json`) | `TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 $PY scripts/transcribe.py voice.mp3 --model small --language en --prompt "Names, Brands"` |
| Import a transcript into a HyperFrames project | `npx hyperframes transcribe transcript.json` (in the project directory) |
| Label sound effects | `$PY scripts/listen.py label sfx/*.wav` (`--labels "a, b"`, `--top N`, `--json`) |
| Split into stems: drums, bass, other, vocals | `$PY -m demucs -o stems voice.wav` (writes `stems/htdemucs/<name>/*.wav`) |
| Render a Blender scene to PNG | `blender -b --factory-startup --python blender/donut.py -- --out donut.png` |
| Make a test voice (Linux) | `espeak-ng -w test.wav "Claude edits my videos"` |

## Rules

- Subtitles always come from transcript timings, never from guesses. Karaoke
  subtitles: show a short phrase, light each word up at its own start time.
- First runs download models: Whisper from `openaipublic.azureedge.net`, CLAP
  from `huggingface.co`, Demucs from `dl.fbaipublicfiles.com`. In the cloud,
  those hosts must be in the environment's allowed domains.
- Ubuntu's Blender has no OpenImageDenoise: don't force Cycles denoising on.
  `blender/donut.py` turns it on only when `_cycles.with_openimagedenoise` is true.

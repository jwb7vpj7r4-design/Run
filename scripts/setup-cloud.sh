#!/bin/bash
# Cloud environment setup for the video tools. Paste this into the environment's
# "Setup script" field (cloud environment menu in the session title bar > Edit).
# When it finishes in under about five minutes, the environment caches the result,
# so new sessions start with everything already installed.
#
# Every step may fail without blocking the session: the script always exits 0.

PY="$HOME/.local/edit-tools/.venv/bin/python"

# Blender (3D renders) and espeak-ng (test voices).
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq blender espeak-ng) &

# HyperFrames' headless Chrome for local renders.
npx -y hyperframes@latest browser ensure &

# Whisper, CLAP (transformers) and Demucs, then their models when the network
# allows the hosts; models that don't download here download on first use.
(
  mkdir -p "$HOME/.local/edit-tools" &&
  uv venv --allow-existing --python 3.11 "$HOME/.local/edit-tools/.venv" &&
  uv pip install --python "$PY" torch==2.2.2 torchaudio==2.2.2 "numpy<2" transformers==4.44.2 soundfile demucs==4.0.1 &&
  uv pip install --python "$PY" --only-binary llvmlite --only-binary numba openai-whisper || exit
  "$PY" -c "import whisper; whisper.load_model('small')" >/dev/null 2>&1 ||
    echo "setup: Whisper model not downloaded (needs openaipublic.azureedge.net)"
  "$PY" -c "from demucs.pretrained import get_model; get_model('htdemucs')" >/dev/null 2>&1 ||
    echo "setup: Demucs model not downloaded (needs dl.fbaipublicfiles.com)"
  "$PY" -c "from transformers import ClapModel, ClapProcessor; m = 'laion/clap-htsat-unfused'; ClapModel.from_pretrained(m); ClapProcessor.from_pretrained(m)" >/dev/null 2>&1 ||
    echo "setup: CLAP model not downloaded (needs huggingface.co)"
) &

wait
exit 0

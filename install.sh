#!/usr/bin/env bash
set -eu
cd /workspace/Colony-ONE
export UV_CACHE_DIR=/workspace/.cache/uv
if command -v uv >/dev/null 2>&1; then
    if [ ! -x .venv/bin/python ]; then
        uv venv .venv
    fi
    uv pip install --python .venv/bin/python -r requirements.txt
else
    python -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -c 'import numpy, onnx, onnxruntime; from colony_one import ColonyOneEnv; print("Colony: One dependencies ready")'

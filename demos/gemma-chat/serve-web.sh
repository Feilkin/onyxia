#!/usr/bin/env bash
# Build the wasm demo and serve it together with the model files.
#
# Usage: ./serve-web.sh [model-dir] [extra trunk args...]
#   e.g. ./serve-web.sh ../../models/gemma-3-270m-it-ONNX --release
#
# The web app fetches ./onnx/model.onnx, ./onnx/model.onnx_data,
# ./tokenizer.json and ./chat_template.jinja relative to the page, so we
# symlink the model into dist/ (symlinks, not copies — the fp32 model is ~1.1GB)
# and serve dist/ with a plain static server.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_DIR="${1:-../../models/gemma-3-270m-it-ONNX}"
MODEL_ABS="$(cd "$MODEL_DIR" && pwd)"
shift || true

trunk build "$@"

ln -sfn "$MODEL_ABS/onnx" dist/onnx
ln -sfn "$MODEL_ABS/tokenizer.json" dist/tokenizer.json
if [ -f "$MODEL_ABS/chat_template.jinja" ]; then
  ln -sfn "$MODEL_ABS/chat_template.jinja" dist/chat_template.jinja
fi

echo
echo "Serving http://localhost:8080  (model: $MODEL_ABS)"
echo "Open in a WebGPU-capable browser (Chrome/Edge)."
python3 -m http.server 8080 --directory dist

# Run all test, including those requiring a GPU
test-all:
    cargo nextest run --run-ignored=all --no-fail-fast

# Run Gemma 3 270m inference with a given prompt
prompt PROMPT:
    cargo run -p onyxia-cli -- run-model models/gemma-3-270m-it-ONNX/onnx/model.onnx --tokenizer models/gemma-3-270m-it-ONNX/ --max-seq-len 1024 --max-tokens 1024 --temperature 0.7 --prompt "{{PROMPT}}"

# Run inference with tracy
trace-prompt PROMPT:
    cargo run --release -p onyxia-cli --features tracy -- run-model models/gemma-3-270m-it-ONNX/onnx/model.onnx --tokenizer models/gemma-3-270m-it-ONNX/ --max-seq-len 1024 --max-tokens 1024 --temperature 0.7 --prompt "{{PROMPT}}"

# Prefill/decode throughput + per-kernel GPU-time breakdown (Gemma 3 270m)
bench:
    cargo run --release -p onyxia-cli -- bench models/gemma-3-270m-it-ONNX/onnx/model.onnx --prefill-len 64 --decode-tokens 32 --profile

# Kernel microbenchmarks at LLM shapes (criterion)
bench-kernels:
    cargo bench -p onyxia-backend-wgpu

# Install the onnx package (Apache-2.0) into .venv for its node test data
fetch-onnx-tests:
    uv venv .venv --allow-existing
    uv pip install --python .venv/bin/python onnx==1.22.0

# ONNX operator conformance matrix on the reference backend
conformance *ARGS:
    cargo run --release -p onyxia-conformance -- --quiet --ops {{ARGS}}

# GPU tests with every layout/dispatch fallback forced (web-like device)
test-gpu-fallbacks:
    ONYXIA_NO_F16=1 ONYXIA_NO_INT64=1 ONYXIA_NO_IMMEDIATES=1 ONYXIA_SUBMIT_CHUNK=1 cargo nextest run --run-ignored=all --no-fail-fast -p onyxia-backend-wgpu

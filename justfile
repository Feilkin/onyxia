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

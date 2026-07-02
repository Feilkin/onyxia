# Run all test, including those requiring a GPU
test-all:
    cargo nextest run --run-ignored=all --no-fail-fast

# Run Gemma 3 270m inference with a given prompt
prompt PROMPT:
    cargo run -p onyxia-cli -- run-model models/gemma-3-270m-it-ONNX/onnx/model.onnx --tokenizer models/gemma-3-270m-it-ONNX/ --max-seq-len 1024 --max-tokens 1024 --temperature 0.7 --prompt "{{PROMPT}}"

# Run inference with tracy
trace-prompt PROMPT:
    cargo run --release -p onyxia-cli --features tracy -- run-model models/gemma-3-270m-it-ONNX/onnx/model.onnx --tokenizer models/gemma-3-270m-it-ONNX/ --max-seq-len 1024 --max-tokens 1024 --temperature 0.7 --prompt "{{PROMPT}}"

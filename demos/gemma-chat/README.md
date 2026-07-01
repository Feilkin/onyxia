# gemma-chat

A minimal egui chatbot demonstrating Gemma 3 270M inference on the Onyxia
runtime. Self-contained example code (sampling, tokenizer, KV-cache session are
vendored here, not part of onyxia) that runs on **desktop** and **web (WASM)**
from the same source.

The demo loads the full-precision `onnx/model.onnx`. The community
`model_q4.onnx` 4-bit quantization badly degrades this small model — verified
against onnxruntime, fp32 stays coherent while q4 collapses into garbage.

## Desktop

```sh
cargo run --release -p gemma-chat -- ../../models/gemma-3-270m-it-ONNX
```

(the model dir defaults to `models/gemma-3-270m-it-ONNX` relative to the cwd).

## Web (WASM)

Requires the `wasm32-unknown-unknown` target and [trunk]:

```sh
rustup target add wasm32-unknown-unknown
cargo install trunk

cd demos/gemma-chat
trunk serve --release
# then open http://localhost:8080 in Chrome/Edge
```

The app fetches the model over HTTP relative to the page. `Trunk.toml` has a
`post_build` hook that symlinks the model into the served directory after each
build (a symlink, not a copy — the fp32 model is ~1.1 GB), so `trunk serve`
serves the app and the model together with hot reload. Override the model with
`MODEL_DIR=/path/to/model-dir trunk serve --release`.

### Requirements & caveats

- **WebGPU** — inference uses a WebGPU compute device (separate from egui's
  renderer). Use a browser with WebGPU enabled (recent Chrome/Edge; Firefox
  behind a flag).
- **Memory** — the fp32 model is ~1.1 GB. The browser fetches it and parses it
  in wasm linear memory (peaks around ~2×), so a 64-bit browser with headroom
  is needed. If it OOMs, a smaller/better-quantized model would be future work.
- First load is slow (large download + GPU weight upload); watch the tab's
  console (`console_log` + panic hook are wired up) for progress/errors.

## Architecture

`main.rs` runs one async inference loop, driven by a background thread
(`pollster::block_on`) on native or `wasm_bindgen_futures::spawn_local` on web.
It talks to the egui UI over channels. Model loading reads files on native and
fetches over HTTP on web (`onyxia_onnx::parse_model_from_bytes`).

[trunk]: https://trunkrs.dev

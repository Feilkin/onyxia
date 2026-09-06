# gemma-chat

An egui chatbot demonstrating Gemma 3 270M inference on the Onyxia runtime.
Self-contained example code (sampling, tokenizer, KV-cache session are vendored
here, not part of onyxia) that runs on **desktop** and **web (WASM)** from the
same source, plus **Android** as a `NativeActivity`. The UI (two themes,
staged loader, streaming, per-answer metrics) lives in `theme.rs` + `lib.rs`;
`main.rs` is the desktop/web command-line wrapper.

The demo loads the full-precision `onnx/model.onnx`. The community
`model_q4.onnx` 4-bit quantization badly degrades this small model — verified
against onnxruntime, fp32 stays coherent while q4 collapses into garbage.

## Desktop

```sh
cargo run --release -p gemma-chat -- ../../models/gemma-3-270m-it-ONNX
```

(the model dir defaults to `models/gemma-3-270m-it-ONNX` relative to the cwd;
pass several directories and the header gets a picker to switch between
them).

To preview the interface without a model, `cargo run -p gemma-chat -- --demo`
drives the UI with scripted load stages and a canned streamed answer.

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

## Android

The same library builds as a `cdylib` exposing `android_main`, packaged with
[cargo-apk] (NativeActivity, no Java or Gradle project). One-time setup, all
in user space (no root):

```sh
# JDK 17 (keytool + apksigner) and Google's command-line tools into ~/Android
mkdir -p ~/Android/jdk ~/Android/sdk/cmdline-tools
curl -L "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse" | tar -xz -C ~/Android/jdk --strip-components=1
curl -LO https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip
unzip commandlinetools-linux-13114758_latest.zip -d ~/Android/sdk/cmdline-tools
mv ~/Android/sdk/cmdline-tools/cmdline-tools ~/Android/sdk/cmdline-tools/latest
JAVA_HOME=~/Android/jdk ~/Android/sdk/cmdline-tools/latest/bin/sdkmanager --sdk_root=$HOME/Android/sdk \
    --install platform-tools "build-tools;34.0.0" "platforms;android-34" "ndk;27.2.12479018"
rustup target add aarch64-linux-android
cargo install cargo-apk
```

Then, with a phone in USB-debugging mode (Android 9+ with Vulkan 1.1; the
fp32 model parses in about 3 GB of RAM):

```sh
just android-install       # cargo apk build --release + adb install
just android-push-model    # copies models/gemma-3-270m-it-ONNX (fp32) to the app's files dir
just android-push-model gemma-3-1b-it-ONNX-GQA model_q4   # the 1B, 4-bit
just android-run           # launches the activity and tails logcat
```

Models live in the app's external files directory,
`/sdcard/Android/data/games.bilberry.onyxia.gemma/files/<model dir>`, which
`adb push` can write without any storage permission (the app has to have
created `files/` first — the recipe starts it once). Every directory there
with an `onnx/` subdir is offered by the model picker in the header
(`gemma-3-1b-it-ONNX-GQA` and `gemma-3-270m-it-ONNX` first, then the rest
alphabetically); the first one loads at start. Inside a directory
`model.onnx` is preferred over `model_q4.onnx`. Switching models tears down
the old session (its GPU memory is released once it finishes) and loads
the new one. The Rust side logs
under the `gemma-chat` logcat tag. The soft keyboard opens when the prompt
field gets focus; the header, composer and system-bar/keyboard insets come
from the activity's content rect.

Measured on a OnePlus 10T (Snapdragon 8+ Gen 1, Adreno 730, Android 15):

| model     | decode      | time to first token | resident |
|-----------|-------------|-------------------|----------|
| 270m fp32 | 11–19 tok/s | 1.5 s             | 1.8 GB   |
| 270m q4   | 22 tok/s    | 3.4 s             | 1.4 GB   |
| 1B q4     | 16 tok/s    | 3.1 s             | 0.9 GB   |

The 1B q4 answers well; the 270m q4 does not (see above), so for the small
model use fp32. Mobile Vulkan drivers cap a single storage-buffer binding
at 128 MiB, under the 671 MB fp32 embedding table (which the 270m q4 export
keeps in fp32 as the tied lm_head) and under the 1B's 151 MB 4-bit one. The
wgpu session asks the IR to split such tables into row chunks
(`onyxia_ir::split_large_tables`: chunked gathers merged by range selects,
chunked `MatMul`/`MatMulNBits` heads concatenated), so all three exports run
unchanged; `ONYXIA_MAX_BINDING=134217728` forces the same path on a desktop
GPU, where the 1B q4 output is identical to the unsplit run. The `ONYXIA_*`
fallbacks (no f16, no int64, no push constants) are exercised by the wgpu
tests, so an older driver should still run.

[cargo-apk]: https://github.com/rust-mobile/cargo-apk

## Architecture

`lib.rs` runs one async inference loop, driven by a background thread
(`pollster::block_on`) on native or `wasm_bindgen_futures::spawn_local` on web.
It talks to the egui UI over channels. Model loading reads files on native and
fetches over HTTP on web (`onyxia_onnx::parse_model_from_bytes`).

[trunk]: https://trunkrs.dev

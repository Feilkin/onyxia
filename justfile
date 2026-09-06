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

# ── Android (gemma-chat demo) ────────────────────────────────────────────────
# One-time setup, all user-space (no root): JDK 17 + Android cmdline-tools in
# ~/Android, then `sdkmanager --install platform-tools "build-tools;34.0.0"
# "platforms;android-34" "ndk;27.2.12479018"`, `rustup target add
# aarch64-linux-android`, `cargo install cargo-apk`.

android_env := "JAVA_HOME=$HOME/Android/jdk PATH=$HOME/Android/jdk/bin:$HOME/Android/sdk/platform-tools:$PATH ANDROID_HOME=$HOME/Android/sdk ANDROID_NDK_ROOT=$HOME/Android/sdk/ndk/27.2.12479018 CARGO_APK_RELEASE_KEYSTORE=$HOME/.android/debug.keystore CARGO_APK_RELEASE_KEYSTORE_PASSWORD=android"
android_pkg := "games.bilberry.onyxia.gemma"

# Build the signed release APK (target/release/apk/gemma-chat.apk)
android-apk:
    {{android_env}} cargo apk build -p gemma-chat --lib --release

# Install the APK on the connected phone (USB debugging on)
android-install: android-apk
    {{android_env}} adb install -r target/release/apk/gemma-chat.apk

# Copy the 270m model into the app's external files dir (fp32, ~1.1 GB,
# one-time). The app is started once first so Android creates `files/` owned
# by the app uid; a dir created by `adb shell mkdir` is owned by `shell` and
# unreadable from inside the app. `just android-push-model model_q4` pushes
# the 4-bit export instead (the app prefers model.onnx when both are there).
android-push-model MODEL="model": android-install
    {{android_env}} adb shell am start -n {{android_pkg}}/android.app.NativeActivity
    sleep 3
    {{android_env}} adb shell am force-stop {{android_pkg}}
    {{android_env}} adb shell mkdir -p /sdcard/Android/data/{{android_pkg}}/files/gemma-3-270m-it-ONNX/onnx
    {{android_env}} adb push models/gemma-3-270m-it-ONNX/tokenizer.json models/gemma-3-270m-it-ONNX/chat_template.jinja /sdcard/Android/data/{{android_pkg}}/files/gemma-3-270m-it-ONNX/
    {{android_env}} adb push models/gemma-3-270m-it-ONNX/onnx/{{MODEL}}.onnx models/gemma-3-270m-it-ONNX/onnx/{{MODEL}}.onnx_data /sdcard/Android/data/{{android_pkg}}/files/gemma-3-270m-it-ONNX/onnx/

# Launch the app and tail its logcat output
android-run:
    {{android_env}} adb shell am start -n {{android_pkg}}/android.app.NativeActivity
    {{android_env}} adb logcat -c
    {{android_env}} adb logcat -s gemma-chat:V RustStdoutStderr:V wgpu:V AndroidRuntime:E DEBUG:E

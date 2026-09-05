"""onnxruntime benchmark mirroring `onyxia bench`:
warmup prefill(P) + 2 decode, then measured prefill(P), then D measured
single-token decode steps. Tokenizer-free (dummy token 42). Logits are
copied to host every step (Onyxia downloads logits too). KV cache stays
on device in the 'cuda-iobinding' mode; round-trips host in 'cuda-numpy'.
"""
import sys, time, json, statistics as st
import numpy as np
import onnxruntime as ort

model, mode, P, D = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
DUMMY = 42

so = ort.SessionOptions()
so.log_severity_level = 3
if mode == "cpu":
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
elif mode in ("webgpu", "webgpu-iobinding"):
    # ORT's WebGPU EP as a runtime plugin (pip install onnxruntime-ep-webgpu).
    import onnxruntime_ep_webgpu as w
    ort.register_execution_provider_library(w.get_ep_name(), w.get_library_path())
    devs = [d for d in ort.get_ep_devices() if d.ep_name == w.get_ep_name()]
    assert devs, "no WebGPU EP device"
    so.add_provider_for_devices(devs[:1], {})
    sess = ort.InferenceSession(model, so)
else:
    providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    sess = ort.InferenceSession(model, so, providers=providers)
    assert sess.get_providers()[0] == "CUDAExecutionProvider", sess.get_providers()

inputs = sess.get_inputs()
outputs = sess.get_outputs()
input_names = {i.name for i in inputs}
kv_in = [i for i in inputs if i.name.startswith("past_key_values.")]
kv_out = [o for o in outputs if o.name.startswith("present.")]
assert len(kv_in) == len(kv_out)
head_dim = kv_in[0].shape[-1]
n_kv_heads = kv_in[0].shape[1]
dev = {"cpu": "cpu", "webgpu": "cpu", "webgpu-iobinding": "webgpu"}.get(mode, "cuda")
use_binding = mode.endswith("-iobinding")


def empty_kv():
    return {i.name: np.zeros((1, n_kv_heads, 0, head_dim), np.float32) for i in kv_in}


class State:
    def __init__(self):
        self.pos = 0
        self.kv = empty_kv()  # numpy (cpu/cuda-numpy) or OrtValue (iobinding)

    def step(self, ids):
        S = len(ids)
        feed = {
            "input_ids": np.array([ids], np.int64),
            "attention_mask": np.ones((1, self.pos + S), np.int64),
            "position_ids": np.arange(self.pos, self.pos + S, dtype=np.int64)[None, :],
        }
        feed = {k: v for k, v in feed.items() if k in input_names}
        if use_binding:
            io = sess.io_binding()
            for k, v in feed.items():
                io.bind_cpu_input(k, v)
            for i in kv_in:
                v = self.kv[i.name]
                if isinstance(v, np.ndarray):
                    io.bind_cpu_input(i.name, v)   # empty cache on step 0
                else:
                    io.bind_ortvalue_input(i.name, v)
            for o in kv_out:
                io.bind_output(o.name, dev, 0)
            io.bind_output("logits", "cpu")
            sess.run_with_iobinding(io)
            outs = io.get_outputs()                     # in *binding* order
            names = [o.name for o in kv_out] + ["logits"]
            logits = outs[names.index("logits")].numpy()
            for o in kv_out:
                self.kv["past_key_values." + o.name[len("present."):]] = outs[names.index(o.name)]
        else:
            feed.update(self.kv)
            outs = sess.run(None, feed)
            names = [o.name for o in outputs]
            logits = outs[names.index("logits")]
            for o in kv_out:
                self.kv["past_key_values." + o.name[len("present."):]] = outs[names.index(o.name)]
        self.pos += S
        return logits[0, -1]


# warmup
s = State()
s.step([DUMMY] * P)
for _ in range(2):
    s.step([DUMMY])

# measured
s = State()
t0 = time.perf_counter()
s.step([DUMMY] * P)
prefill_s = time.perf_counter() - t0
steps = []
for _ in range(D):
    t0 = time.perf_counter()
    s.step([DUMMY])
    steps.append(time.perf_counter() - t0)

mean = st.mean(steps)
print(f"onnxruntime {ort.__version__} mode={mode} providers={sess.get_providers()[0]}")
print(f"prefill: {P} tokens in {prefill_s*1e3:.1f} ms ({P/prefill_s:.1f} tok/s)")
print(f"decode:  {D} tokens, {mean*1e3:.2f} ms/tok mean (min {min(steps)*1e3:.2f}, max {max(steps)*1e3:.2f}, σ {st.pstdev(steps)*1e3:.2f}) → {1/mean:.2f} tok/s")
print(json.dumps({"mode": mode, "prefill_tok_s": P / prefill_s, "prefill_ms": prefill_s * 1e3,
                  "decode_ms_per_tok": mean * 1e3, "decode_tok_s": 1 / mean}))

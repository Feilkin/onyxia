"""onnxruntime-genai benchmark, the tuned-ORT counterpart of `onyxia bench`:
ORT's own model builder export (device-resident shared KV buffer, no
shape subgraphs, optional CUDA graphs) driven by genai's generation loop.
Tokenizer-free: P copies of a dummy token are appended (the prefill),
then D greedy decode steps are timed one by one. Prefill is the median
of 5 fresh generators. Usage: genai_bench.py <model_dir> <P> <D>
"""
import sys, time, statistics as st
import numpy as np
import onnxruntime_genai as og

path, P, D = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
DUMMY = 42
model = og.Model(path)


def make():
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=P + D + 8, do_sample=False, min_length=P + D + 4)
    return og.Generator(model, params)


def prefill(gen):
    t0 = time.perf_counter()
    gen.append_tokens(np.full((1, P), DUMMY, np.int32))
    if hasattr(gen, "get_logits"):
        gen.get_logits()  # force the forward to be observable on host, like Onyxia's logits row
    return time.perf_counter() - t0


# warmup (compiles / captures CUDA graph)
g = make(); prefill(g)
for _ in range(3):
    g.generate_next_token()
del g

prefills = []
for _ in range(5):
    g = make()
    prefills.append(prefill(g))
    del g
g = make(); prefill(g)
steps = []
for _ in range(D):
    t0 = time.perf_counter()
    g.generate_next_token()
    steps.append(time.perf_counter() - t0)
first = steps[0]
tail = steps[1:]
mean = st.mean(tail)
print(f"onnxruntime-genai {og.__version__} model={path}")
print(f"prefill: {P} tokens in {st.median(prefills)*1e3:.1f} ms median of 5 (min {min(prefills)*1e3:.1f}) ({P/st.median(prefills):.1f} tok/s); first generate_next_token after it {first*1e3:.2f} ms")
print(f"decode:  {len(tail)} tokens, {mean*1e3:.2f} ms/tok mean (min {min(tail)*1e3:.2f}, max {max(tail)*1e3:.2f}, σ {st.pstdev(tail)*1e3:.2f}) → {1/mean:.2f} tok/s")

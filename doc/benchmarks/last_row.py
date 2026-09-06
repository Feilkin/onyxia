"""Make `<model>_last.onnx` next to `<model>.onnx`: same graph, but the
`logits` output is the last position only ([1, S, V] -> [1, 1, V]), so an
onnxruntime run() copies one row to host, as Onyxia's LlmSession does.
External weights are referenced, not copied."""
import sys, onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
src = sys.argv[1]
m = onnx.load(src, load_external_data=False)
g = m.graph
out = next(o for o in g.output if o.name == "logits")
# rename the producer's output, slice it
for n in g.node:
    for i, name in enumerate(n.output):
        if name == "logits":
            n.output[i] = "logits_all"
g.initializer.extend([
    numpy_helper.from_array(np.array([-1], np.int64), "last_starts"),
    numpy_helper.from_array(np.array([2**62], np.int64), "last_ends"),
    numpy_helper.from_array(np.array([1], np.int64), "last_axes"),
])
g.node.append(helper.make_node("Slice", ["logits_all", "last_starts", "last_ends", "last_axes"], ["logits"], name="logits_last_row"))
out.type.tensor_type.shape.dim[1].ClearField("dim_param"); out.type.tensor_type.shape.dim[1].dim_value = 1
dst = src[:-5] + "_last.onnx"
onnx.save(m, dst)
print("wrote", dst)

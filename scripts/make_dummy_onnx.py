"""Dummy ONNX with the contract I/O signature (CONTRACT.md section 5).
features [1,30,13] -> flatten -> 64 relu -> engagement [1,4], states [1,4] (raw logits).
Run once, commit output: python scripts/make_dummy_onnx.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

rng = np.random.default_rng(42)
def w(name, shape, scale=0.1):
    return numpy_helper.from_array((rng.standard_normal(shape) * scale).astype(np.float32), name)

inits = [
    numpy_helper.from_array(np.array([1, 390], dtype=np.int64), "flat_shape"),
    w("W1", (390, 64)), w("b1", (64,)),
    w("W2", (64, 4)),  w("b2", (4,)),
    w("W3", (64, 4)),  w("b3", (4,)),
]
nodes = [
    helper.make_node("Reshape", ["features", "flat_shape"], ["flat"]),
    helper.make_node("Gemm", ["flat", "W1", "b1"], ["h_pre"]),
    helper.make_node("Relu", ["h_pre"], ["h"]),
    helper.make_node("Gemm", ["h", "W2", "b2"], ["engagement"]),
    helper.make_node("Gemm", ["h", "W3", "b3"], ["states"]),
]
graph = helper.make_graph(
    nodes, "dummy_engagement",
    [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 30, 13])],
    [helper.make_tensor_value_info("engagement", TensorProto.FLOAT, [1, 4]),
     helper.make_tensor_value_info("states", TensorProto.FLOAT, [1, 4])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "web/public/model/model_int8.onnx")
print("saved web/public/model/model_int8.onnx")

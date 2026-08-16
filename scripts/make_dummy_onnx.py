"""Dummy ONNX with the contract I/O signature (CONTRACT.md section 5).

features [1,30,13] -> flatten -> 64 relu -> engagement [1,4], states [1,4]
(raw logits). RANDOM WEIGHTS — this exists only as the Day-2 scaffold that let
Track B build against the contract before a real model existed (CONTRACT.md
section 1), and for latency-only benchmarking.

This is NOT how the shipped model is produced. That comes from:
    python ml/src/export.py --checkpoint artifacts/runs/<ts>/best.pt --ship

Writes to artifacts/dummy/ by default and refuses to clobber an existing file.
Earlier versions hard-coded `web/public/model/model_int8.onnx` as the
destination, so a single stray run replaced the trained 60 KB TCN with random
weights — and because the dummy honours the same contract signature, the app
kept running and simply emitted noise. Nothing in CI would have caught it.

    python scripts/make_dummy_onnx.py [--out PATH] [--force]
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "artifacts" / "dummy" / "model_int8.onnx"
SHIPPED_DIR = REPO_ROOT / "web" / "public" / "model"

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"destination .onnx (default: {DEFAULT_OUT})")
parser.add_argument("--force", action="store_true",
                    help="overwrite --out if it already exists")
args = parser.parse_args()

out_path = args.out.resolve()
if out_path.exists() and not args.force:
    raise SystemExit(f"refusing to overwrite existing {out_path} (pass --force)")
if out_path.parent == SHIPPED_DIR:
    print(f"WARNING: writing a RANDOM-WEIGHT model into the shipped model "
          f"directory ({SHIPPED_DIR}).\n"
          f"         The app will keep running and silently emit noise. "
          f"Restore with:\n"
          f"         git checkout -- web/public/model/")

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
out_path.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, str(out_path))
print(f"saved {out_path}")

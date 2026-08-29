"""ONNX export + parity check + STATIC QDQ int8 quantization (A9 / A6.5).

Order of operations (each step gates the next):
  1. model.eval()  (BatchNorm must use running stats, not batch stats)
  2. torch.onnx.export, opset 17, dynamic batch axis
  3. onnx.checker.check_model
  4. PyTorch vs onnxruntime parity on 100 random inputs, max|diff| < 1e-5
     — the build FAILS otherwise
  5. quantize_static (QDQ format), calibrated on real standardised training
     windows. NOT quantize_dynamic: dynamic quantization emits ConvInteger
     nodes that onnxruntime-web's WASM backend does not implement — found
     empirically via the A6.5 browser smoke test (see quantize()'s
     docstring); an earlier revision of this header still described the
     rejected dynamic design.
  6. size + (optional) copy to web/public/model/

Usage:
  python ml/src/export.py --untrained --out artifacts/smoke
      A6.5 smoke: export an untrained model for the browser op-support test.
  python ml/src/export.py --checkpoint artifacts/runs/<ts>/best.pt --ship
      Real export; --ship copies model_int8.onnx into web/public/model/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from model import N_FEATURES, EngagementTCN  # noqa: E402

PARITY_TOLERANCE = 1e-5
PARITY_RUNS = 100


def export_fp32(model: torch.nn.Module, out_path: Path) -> None:
    model.eval()  # CRITICAL: else BN uses batch stats and browser outputs drift
    dummy = torch.zeros(1, 30, N_FEATURES)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["features"],
        output_names=["engagement", "states"],
        dynamic_axes={"features": {0: "batch"},
                      "engagement": {0: "batch"},
                      "states": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        # torch>=2.13 defaults to the dynamo exporter, which forces opset 18
        # and emits a graph the ORT dynamic quantizer's shape inference
        # rejects (found during the A6.5 smoke). The legacy exporter honours
        # opset 17 and quantizes cleanly.
        dynamo=False,
    )


def check_model(onnx_path: Path) -> None:
    import onnx
    onnx.checker.check_model(onnx.load(str(onnx_path)))


def parity_check(model: torch.nn.Module, onnx_path: Path,
                 seed: int = 0) -> float:
    """Max abs diff between PyTorch and onnxruntime over random inputs."""
    import onnxruntime as ort
    model.eval()
    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(PARITY_RUNS):
        x = rng.standard_normal((1, 30, N_FEATURES)).astype(np.float32)
        with torch.no_grad():
            t_eng, t_states = model(torch.from_numpy(x))
        o_eng, o_states = session.run(None, {"features": x})
        worst = max(worst,
                    float(np.max(np.abs(t_eng.numpy() - o_eng))),
                    float(np.max(np.abs(t_states.numpy() - o_states))))
    return worst


def quantize(fp32_path: Path, int8_path: Path,
             calibration_npz: Path | None,
             allow_random_calibration: bool = False) -> str:
    """STATIC QDQ int8 quantization. Returns the calibration source used.

    Dynamic quantization emits ConvInteger nodes, which onnxruntime-web's
    WASM backend does not implement (found in the A6.5 browser smoke).
    Static QDQ emits QLinearConv, which it does. Calibration uses real
    standardised training windows; random normals are permitted ONLY with
    allow_random_calibration=True (untrained smoke — op support is what
    matters there), and missing calibration data is otherwise a hard error.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader, QuantFormat, QuantType, quantize_static)

    if calibration_npz is not None and calibration_npz.exists():
        data = np.load(calibration_npz)["x"]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(data), size=min(256, len(data)), replace=False)
        samples = [data[i:i + 1].astype(np.float32) for i in idx]
        calibration_source = str(calibration_npz)
    elif allow_random_calibration:
        rng = np.random.default_rng(0)
        samples = [rng.standard_normal((1, 30, N_FEATURES)).astype(np.float32)
                   for _ in range(256)]
        calibration_source = "RANDOM (standard normal) — smoke use only"
        print("WARNING: calibrating int8 quantization on RANDOM data — "
              "acceptable for the --untrained browser smoke only, never "
              "for a shipped model.")
    else:
        # Never silently ship a model calibrated on random noise: a typo'd
        # --calibration path used to fall through here without a word, and
        # the resulting badly-calibrated int8 model would pass smoke_run_int8
        # (shape check on zeros) and reach web/public/model/.
        raise SystemExit(
            f"calibration data not found: {calibration_npz} — build "
            f"artifacts/dataset/Train.npz first (python ml/src/dataset.py) "
            f"or pass --allow-random-calibration for a smoke export.")

    class WindowReader(CalibrationDataReader):
        def __init__(self):
            self._iter = iter(samples)

        def get_next(self):
            nxt = next(self._iter, None)
            return None if nxt is None else {"features": nxt}

    quantize_static(str(fp32_path), str(int8_path), WindowReader(),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True)
    return calibration_source


def smoke_run_int8(int8_path: Path) -> None:
    """The quantized graph must load and run in Python ORT at minimum."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(int8_path),
                                   providers=["CPUExecutionProvider"])
    out = session.run(None, {"features":
                             np.zeros((1, 30, N_FEATURES), np.float32)})
    assert out[0].shape == (1, 4) and out[1].shape == (1, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="trained state_dict (.pt); omit with --untrained")
    parser.add_argument("--untrained", action="store_true",
                        help="export a freshly initialised model (A6.5 smoke)")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / "export")
    parser.add_argument("--ship", action="store_true",
                        help="copy model_int8.onnx to web/public/model/")
    parser.add_argument("--calibration", type=Path,
                        default=REPO_ROOT / "artifacts" / "dataset" / "Train.npz",
                        help="npz with standardised windows for static "
                             "quantization calibration")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.untrained and args.checkpoint is None:
        parser.error("need --checkpoint or --untrained")

    torch.manual_seed(args.seed)
    model = EngagementTCN()
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint,
                                         map_location="cpu",
                                         weights_only=True))
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    fp32_path = args.out / "model_fp32.onnx"
    int8_path = args.out / "model_int8.onnx"

    export_fp32(model, fp32_path)
    check_model(fp32_path)
    print("onnx.checker: OK")

    worst = parity_check(model, fp32_path, seed=args.seed)
    print(f"parity max|diff| over {PARITY_RUNS} runs: {worst:.2e} "
          f"(tolerance {PARITY_TOLERANCE:.0e})")
    if worst >= PARITY_TOLERANCE:
        raise SystemExit("PARITY CHECK FAILED — build aborted")

    calibration_source = quantize(
        fp32_path, int8_path, args.calibration,
        allow_random_calibration=args.untrained)
    smoke_run_int8(int8_path)

    sizes = {p.name: p.stat().st_size for p in (fp32_path, int8_path)}
    for name, size in sizes.items():
        print(f"{name}: {size/1024:.1f} KB")
    with open(args.out / "export_report.json", "w") as fh:
        json.dump({"parity_max_abs_diff": worst,
                   "sizes_bytes": sizes,
                   "checkpoint": str(args.checkpoint) if args.checkpoint
                   else "untrained",
                   "calibration_source": calibration_source,
                   "opset": 17, "seed": args.seed}, fh, indent=1)

    if args.ship:
        dest = REPO_ROOT / "web" / "public" / "model" / "model_int8.onnx"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(int8_path.read_bytes())
        print(f"shipped to {dest}")


if __name__ == "__main__":
    main()

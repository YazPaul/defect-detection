"""
benchmark.py — Compare inference backends and print a latency summary table

Usage:
    python src/benchmark.py \
        --ckpt mlruns/Patchcore/MVTecAD/cable/latest/weights/lightning/model.ckpt \
        --onnx exports/patchcore_cable.onnx \
        --trt  exports/patchcore_cable.trt \
        --n 300

Any backend whose file/library is missing is skipped. Latency only — for
AUROC per backend run `inference.py --eval` with the same model file.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from backends import ONNXInspector, TRTInspector


def measure(infer_fn, n: int = 300, warmup: int = 30, image_size: int = 256) -> dict:
    dummy = np.random.rand(1, 3, image_size, image_size).astype(np.float32)
    for _ in range(warmup):
        infer_fn(dummy)
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        infer_fn(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": np.mean(latencies),
        "p50_ms":  np.percentile(latencies, 50),
        "p95_ms":  np.percentile(latencies, 95),
        "fps":     1000 / np.mean(latencies),
    }


def torch_backends(ckpt: str):
    """Yield (name, infer_fn) for PyTorch on CPU and, if available, CUDA."""
    import anomalib
    import torch
    from anomalib.models import Patchcore

    torch.serialization.add_safe_globals([anomalib.PrecisionType])
    model = Patchcore.load_from_checkpoint(ckpt, weights_only=False, map_location="cpu").eval()

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        model.to(dev)

        def infer(x, dev=dev):
            with torch.inference_mode():
                model(torch.from_numpy(x).to(dev))  # pre-processor -> model -> post-processor
            if dev == "cuda":
                torch.cuda.synchronize()

        yield f"PyTorch ({dev.upper()})", infer


def run_all(args):
    results = []

    if args.ckpt and Path(args.ckpt).exists():
        try:
            for name, fn in torch_backends(args.ckpt):
                print(f"[{name}] measuring...")
                results.append({"backend": name, "precision": "FP32",
                                **measure(fn, args.n, image_size=args.image_size)})
        except Exception as e:
            print(f"[PyTorch] Skipped: {e}")
    else:
        print(f"[PyTorch] Skipped: checkpoint not found ({args.ckpt})")

    if Path(args.onnx).exists():
        try:
            insp = ONNXInspector(args.onnx)
            name = f"ONNX Runtime ({insp.provider.replace('ExecutionProvider', '')})"
            print(f"[{name}] measuring...")
            results.append({"backend": name, "precision": "FP32",
                            **measure(insp.infer, args.n, image_size=args.image_size)})
        except Exception as e:
            print(f"[ONNX] Skipped: {e}")
    else:
        print(f"[ONNX] Skipped: file not found ({args.onnx})")

    if Path(args.trt).exists():
        try:
            insp = TRTInspector(args.trt)
            print("[TensorRT] measuring...")
            results.append({"backend": "TensorRT", "precision": args.trt_precision,
                            **measure(insp.infer, args.n, image_size=args.image_size)})
        except Exception as e:
            print(f"[TRT] Skipped: {e}")
    else:
        print(f"[TRT] Skipped: file not found ({args.trt})")

    print(f"\n-- Backend Comparison ({args.n} runs) " + "-" * 44)
    print(f"{'Backend':<35} {'Prec':<8} {'Mean ms':>8} {'P50 ms':>8} {'P95 ms':>8} {'FPS':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['backend']:<35} {r['precision']:<8} {r['mean_ms']:>8.2f} "
              f"{r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f} {r['fps']:>8.1f}")
    print("-" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="mlruns/Patchcore/MVTecAD/cable/latest/weights/lightning/model.ckpt",
                        help="anomalib Lightning checkpoint for the PyTorch rows")
    parser.add_argument("--onnx", type=str, default="exports/patchcore_cable.onnx")
    parser.add_argument("--trt",  type=str, default="exports/patchcore_cable.trt")
    parser.add_argument("--trt-precision", type=str, default="FP16",
                        help="Label for the TRT row; must match how the engine was built")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=256,
                        help="Input image size (must match the exported model, default: 256)")
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())

"""
inference.py — Real-time defect inspection CLI

Usage:
    # Single image, TensorRT engine
    python src/inference.py --image data/cable/test/bent_wire/000.png --engine exports/patchcore_cable.trt

    # Single image, ONNX (no GPU required)
    python src/inference.py --image data/cable/test/good/000.png --onnx exports/patchcore_cable.onnx

    # Benchmark mode: measure latency/throughput over N runs
    python src/inference.py --benchmark --engine exports/patchcore_cable.trt --n 500

    # Full test set evaluation: AUROC, accuracy, and a suggested threshold
    python src/inference.py --eval --onnx exports/patchcore_cable.onnx --data data/cable/test
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from backends import IMAGE_SIZE, ONNXInspector, TRTInspector, preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# 0.5 is only meaningful for exports that keep anomalib's post-processor
# (scores normalized to [0, 1]). For raw-score exports (export_from_checkpoint.py)
# calibrate with --eval and pass the suggested value via --threshold.
DEFAULT_THRESHOLD = 0.5


def save_heatmap(heatmap: np.ndarray, image_path: str, out_dir: str = "results/heatmaps"):
    """Overlay anomaly heatmap on original image and save."""
    try:
        import cv2
        from PIL import Image
    except ImportError:
        log.warning("pip install opencv-python Pillow — skipping heatmap save")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{Path(image_path).stem}_heatmap.png"

    orig = np.array(Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    hm = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * orig + 0.4 * hm_color).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)
    log.info(f"Heatmap saved → {out_path}")


def benchmark(inspector, n: int = 500, warmup: int = 20):
    """Measure latency and throughput over N timed runs (after warmup)."""
    dummy = np.random.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    for _ in range(warmup):
        inspector.infer(dummy)

    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        inspector.infer(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    print("\n── Benchmark Results ─────────────────────────")
    print(f"  Runs        : {n}")
    print(f"  Mean latency: {np.mean(latencies):.2f} ms")
    print(f"  P50 latency : {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95 latency : {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99 latency : {np.percentile(latencies, 99):.2f} ms")
    print(f"  Throughput  : {1000 / np.mean(latencies):.1f} FPS")
    print("──────────────────────────────────────────────\n")


def evaluate(inspector, data_dir: Path, threshold: float):
    """Score every image under data_dir; folder 'good' = normal, anything else = defect."""
    images = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images:
        raise FileNotFoundError(f"No images found under {data_dir}")

    scores, labels = [], []
    for img_path in images:
        score, _ = inspector.infer(preprocess(str(img_path)))
        is_gt_defect = img_path.parent.name != "good"
        scores.append(score)
        labels.append(int(is_gt_defect))
        correct = (score > threshold) == is_gt_defect
        log.debug(f"{'✓' if correct else '✗'} {img_path.parent.name}/{img_path.name} | score={score:.4f}")

    scores, labels = np.array(scores), np.array(labels)
    preds = scores > threshold
    acc = float(np.mean(preds == labels.astype(bool)))

    print("\n── Evaluation ────────────────────────────────")
    print(f"  Images      : {len(images)}  (defect: {labels.sum()}, good: {len(labels) - labels.sum()})")
    print(f"  Score range : {scores.min():.4f} – {scores.max():.4f}")
    print(f"  Threshold   : {threshold}  -> accuracy {acc:.3f}, flagged {int(preds.sum())}")
    try:
        from sklearn.metrics import precision_recall_curve, roc_auc_score
        if 0 < labels.sum() < len(labels):
            print(f"  Image AUROC : {roc_auc_score(labels, scores):.4f}")
            prec, rec, thr = precision_recall_curve(labels, scores)
            f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
            best = int(np.argmax(f1))
            print(f"  Best-F1 thr : {thr[best]:.4f}  (F1 {f1[best]:.3f}) — pass via --threshold")
            print("                (tuned on the test set; use a held-out split for production)")
    except ImportError:
        print("  pip install scikit-learn for AUROC / threshold suggestion")
    print("──────────────────────────────────────────────\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Defect inspection CLI")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--engine", type=str, help="Path to TensorRT .trt engine")
    parser.add_argument("--onnx", type=str, help="Path to ONNX model (alternative to TRT)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Anomaly score threshold (default 0.5, valid for normalized exports; "
                             "calibrate with --eval for raw-score exports)")
    parser.add_argument("--save-heatmap", action="store_true", help="Save anomaly heatmap overlay")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--n", type=int, default=500, help="Number of benchmark iterations")
    parser.add_argument("--eval", action="store_true", help="Evaluate full test directory")
    parser.add_argument("--data", type=str, help="Test data directory (for --eval)")
    return parser.parse_args()


def main():
    # Windows consoles/pipes default to cp1252, which can't print ✓ ⚠ ─
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = parse_args()

    if args.engine:
        inspector = TRTInspector(args.engine)
    elif args.onnx:
        inspector = ONNXInspector(args.onnx)
    else:
        raise SystemExit("Provide --engine (TRT) or --onnx")

    if args.benchmark:
        benchmark(inspector, n=args.n)
    elif args.image:
        image = preprocess(args.image)
        t0 = time.perf_counter()
        score, heatmap = inspector.infer(image)
        latency_ms = (time.perf_counter() - t0) * 1000

        label = "⚠  DEFECT DETECTED" if score > args.threshold else "✓  OK"
        print(f"\n{label}  |  score: {score:.4f}  |  threshold: {args.threshold}  |  latency: {latency_ms:.1f}ms\n")
        if args.save_heatmap:
            if heatmap is None:
                log.warning("Model has no anomaly_map output — no heatmap to save")
            else:
                save_heatmap(heatmap, args.image)
    elif args.eval:
        if not args.data:
            raise SystemExit("--data required for --eval mode")
        evaluate(inspector, Path(args.data), args.threshold)
    else:
        raise SystemExit("Choose a mode: --image, --benchmark or --eval")


if __name__ == "__main__":
    main()

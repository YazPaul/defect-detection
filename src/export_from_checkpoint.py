"""
export_from_checkpoint.py — Export a saved PatchCore checkpoint to ONNX

Usage:
    # Normalized scores in [0, 1] (works with the default --threshold 0.5)
    python src/export_from_checkpoint.py --category cable

    # Raw distance scores, outputs only pred_score + anomaly_map
    # (calibrate the threshold with `inference.py --eval`)
    python src/export_from_checkpoint.py --category cable --raw-scores
"""

import argparse
import shutil
from pathlib import Path

import anomalib
import torch
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Export PatchCore checkpoint to ONNX")
    parser.add_argument("--category", type=str, default="cable")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path (default: latest run for --category)")
    parser.add_argument("--raw-scores", action="store_true",
                        help="Drop the post-processor (no score normalization / thresholding)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .onnx path (default: exports/patchcore_<category>[_raw].onnx)")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else (
        PROJECT_ROOT / "mlruns" / "Patchcore" / "MVTecAD" / args.category
        / "latest" / "weights" / "lightning" / "model.ckpt"
    )
    suffix = "_raw" if args.raw_scores else ""
    output = Path(args.output) if args.output else (
        PROJECT_ROOT / "exports" / f"patchcore_{args.category}{suffix}.onnx"
    )

    torch.serialization.add_safe_globals([anomalib.PrecisionType])
    model = Patchcore.load_from_checkpoint(ckpt, weights_only=False)
    if args.raw_scores:
        model.post_processor = None

    export_root = output.parent / f"_export_{output.stem}"
    engine = Engine(default_root_dir=str(PROJECT_ROOT / "mlruns"))
    engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=str(export_root),
        input_size=(args.image_size, args.image_size),
    )

    # anomalib writes {export_root}/weights/onnx/model.onnx — move to the canonical path
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(export_root / "weights" / "onnx" / "model.onnx"), str(output))
    shutil.rmtree(export_root, ignore_errors=True)
    print(f"Exported → {output}")


if __name__ == "__main__":
    main()

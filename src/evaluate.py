"""
evaluate.py — Re-run anomalib's test loop on a saved PatchCore checkpoint

Usage:
    python src/evaluate.py --category cable
"""

import argparse
from pathlib import Path

import anomalib
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ -> project root


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PatchCore checkpoint on MVTec AD")
    parser.add_argument("--category", type=str, default="cable")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path (default: latest run for --category)")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else (
        PROJECT_ROOT / "mlruns" / "Patchcore" / "MVTecAD" / args.category
        / "latest" / "weights" / "lightning" / "model.ckpt"
    )

    torch.serialization.add_safe_globals([anomalib.PrecisionType])
    model = Patchcore.load_from_checkpoint(ckpt, weights_only=False)

    datamodule = MVTecAD(
        root=args.data_dir,
        category=args.category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,  # Windows-safe
    )

    engine = Engine(default_root_dir=str(PROJECT_ROOT / "mlruns"))
    print(engine.test(model=model, datamodule=datamodule))


if __name__ == "__main__":
    main()

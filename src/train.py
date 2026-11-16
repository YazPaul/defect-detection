"""
train.py — PatchCore training on MVTec AD + ONNX export

Usage:
    python src/train.py --category cable --export-onnx
    python src/train.py --category bottle --export-onnx --backbone wide_resnet50_2
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Resolve default paths against the project root, regardless of the working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTec AD")
    parser.add_argument("--category", type=str, default="cable",
                        help="MVTec AD category (cable, bottle, carpet, ...)")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data"),
                        help="Root directory containing MVTec AD")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export trained model to ONNX after training")
    parser.add_argument("--export-dir", type=str, default=str(PROJECT_ROOT / "exports"),
                        help="Directory for ONNX/TRT exports")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2",
                        choices=["wide_resnet50_2", "resnet18", "efficientnet_b4"],
                        help="Feature extractor backbone")
    parser.add_argument("--coreset-ratio", type=float, default=0.1,
                        help="Fraction of patches to keep in memory bank (0.01–0.5)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Input image size for training")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train(args):
    try:
        from anomalib.data import MVTecAD
        from anomalib.models import Patchcore
        from anomalib.engine import Engine
        from anomalib.deploy import ExportType

    except ImportError:
        raise ImportError(
            "anomalib not installed. Run: pip install anomalib"
        )
    try:
        import torch
        from torchvision.transforms import v2
    except ImportError:
        raise ImportError(
            "pytorch not installed. Run: pip install torchvision"
        )


    torch.manual_seed(args.seed)
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Category   : {args.category}")
    log.info(f"Backbone   : {args.backbone}")
    log.info(f"Coreset    : {args.coreset_ratio}")
    log.info(f"Image size : {args.image_size}")

    # Data
    # Image resize is now handled via transforms, not datamodule args
    augmentations = v2.Compose([
        v2.Resize((args.image_size, args.image_size)),
    ])

    datamodule = MVTecAD(
        root=args.data_dir,
        category=args.category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,  # set to 0 on Windows to avoid multiprocessing issues
        augmentations=augmentations,
    )

    # Model
    model = Patchcore(
        backbone=args.backbone,
        coreset_sampling_ratio=args.coreset_ratio,
        num_neighbors=9,
    )

    # Engine. Note: "mlruns" is only the anomalib output directory (checkpoints,
    # results); no MLflow logger is attached.
    engine = Engine(
        max_epochs=1,           # PatchCore is one-epoch: just feature extraction
        accelerator="auto",
        devices=1,
        default_root_dir=str(PROJECT_ROOT / "mlruns"),
    )

    log.info("Starting training...")
    engine.fit(model=model, datamodule=datamodule)

    # Evaluate
    log.info("Evaluating...")
    results = engine.test(model=model, datamodule=datamodule)
    auroc = results[0].get("image_AUROC", "N/A")
    log.info(f"AUROC: {auroc:.4f}" if isinstance(auroc, float) else f"AUROC: {auroc}")

    # Export
    if args.export_onnx:
        onnx_path = export_dir / f"patchcore_{args.category}.onnx"
        log.info(f"Exporting ONNX → {onnx_path}")
        import anomalib
        import shutil
        torch.serialization.add_safe_globals([anomalib.PrecisionType])
        engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=str(export_dir),
            input_size=(args.image_size, args.image_size),
        )

        # anomalib always writes to {export_root}/weights/onnx/model.onnx;
        # move it to the canonical path used by optimize_trt.py and benchmark.py
        anomalib_out = export_dir / "weights" / "onnx" / "model.onnx"
        if anomalib_out.exists():
            shutil.move(str(anomalib_out), str(onnx_path))
            log.info(f"Moved ONNX → {onnx_path}")

        log.info("Export complete.")

    return model


if __name__ == "__main__":
    args = parse_args()
    train(args)

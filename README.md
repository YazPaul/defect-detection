# Industrial Defect Detection Pipeline

> Real-time anomaly detection on MVTec AD — **PatchCore → ONNX → TensorRT**
> From training to GPU-optimized inference in a single reproducible pipeline.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red) ![anomalib](https://img.shields.io/badge/anomalib-2.x-orange) ![TensorRT](https://img.shields.io/badge/TensorRT-10.x-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Problem

Industrial inline inspection systems must detect surface defects at high throughput — often 30–200 frames/s — with minimal false positives. Traditional rule-based methods struggle with texture variability; deep learning models are accurate but often too slow for real-time deployment without optimization.

This project builds a complete, deployable defect detection pipeline:
- **Accurate**: 98.4% image AUROC on MVTec AD *cable* (ONNX export, measured)
- **Fast**: TensorRT FP16 engine for GPU inference
- **Portable**: Docker image with a clean CLI

---

## Approach

**Model: PatchCore** ([Roth et al., 2022](https://arxiv.org/abs/2106.08265))
PatchCore is a memory-bank anomaly detection method. It extracts patch-level features from a pretrained backbone (WideResNet-50), stores a coreset of normal features, and detects anomalies by measuring distance to the nearest stored patch. No anomaly labels required during training — only normal images.

**Why PatchCore over alternatives?**
- No training on defect examples (scarce in real industrial settings)
- State-of-the-art AUROC with no fine-tuning
- Deterministic, interpretable anomaly scores
- Straightforward ONNX export path

**Optimization stack:**
```
PyTorch (anomalib) → ONNX export → TensorRT FP16 engine → Python inference CLI
```

The exported ONNX graph contains anomalib's pre-processor (ImageNet normalization) and, by default, its post-processor (score normalization to [0, 1]). The inference CLI therefore feeds raw RGB in [0, 1].

---

## Results

### Cable category (MVTec AD, 150 test images: 92 defect / 58 good)

| Backend | Precision | Latency (ms/img) | Throughput (FPS) | Image AUROC |
|---|---|---|---|---|
| PyTorch CPU | FP32 | ~740 (P50) | ~1 | — |
| ONNX Runtime CPU | FP32 | ~710 (P50) | ~1 | **98.4%** |
| PyTorch GPU | FP32 | TBD | TBD | TBD |
| ONNX Runtime GPU | FP32 | TBD | TBD | TBD |
| TensorRT | FP16 | TBD | TBD | TBD |

CPU rows were measured on a laptop CPU (`benchmark.py --n 20`, so treat them as rough). At threshold 0.5 the ONNX model reaches 95.3% accuracy; the best-F1 threshold on the test set is ≈0.51 (F1 0.963).
Fill the GPU rows with `python src/benchmark.py` and `python src/inference.py --eval --engine ...` on the target GPU.

### Sample result

`python src/inference.py --onnx exports/patchcore_cable.onnx --image data/cable/test/bent_wire/000.png --save-heatmap`
→ `⚠ DEFECT DETECTED | score: 0.9350`, heatmap written to `results/heatmaps/000_heatmap.png`.

---

## Quickstart

### 1. Install
```bash
pip install -r requirements.txt
# TensorRT + pycuda are installed separately (see requirements.txt), or use Docker / Colab
```

### 2. Download MVTec AD
Download from https://www.mvtec.com/company/research/datasets/mvtec-ad and extract to `data/` (e.g. `data/cable/train/good`, `data/cable/test/...`).

### 3. Train PatchCore & export ONNX
```bash
python src/train.py --category cable --export-onnx
# Outputs: exports/patchcore_cable.onnx, checkpoint under mlruns/Patchcore/MVTecAD/cable/latest/
```
Re-export from an existing checkpoint (optionally with raw, un-normalized scores):
```bash
python src/export_from_checkpoint.py --category cable              # -> exports/patchcore_cable.onnx
python src/export_from_checkpoint.py --category cable --raw-scores # -> exports/patchcore_cable_raw.onnx
```

### 4. Build the TensorRT engine
```bash
python src/optimize_trt.py --onnx exports/patchcore_cable.onnx --fp16
# Outputs: exports/patchcore_cable.trt
```
Engines are tied to the TensorRT version **and** GPU architecture they were built on — build on the machine (or in the container) that will run them. No local TensorRT? Use `notebooks/colab_trt_export.ipynb` (engine then only runs on a T4 with the same TRT version).

### 5. Run inference
```bash
# Single image
python src/inference.py --engine exports/patchcore_cable.trt --image data/cable/test/bent_wire/000.png --save-heatmap

# Evaluate a test folder: accuracy, AUROC, suggested threshold
python src/inference.py --eval --onnx exports/patchcore_cable.onnx --data data/cable/test

# Latency benchmark (one backend)
python src/inference.py --benchmark --engine exports/patchcore_cable.trt --n 500

# Compare all backends (PyTorch CPU/GPU, ONNX Runtime, TensorRT)
python src/benchmark.py --n 300
```

**Threshold:** the default `--threshold 0.5` assumes the normalized export. For `--raw-scores` exports, run `--eval` and use the suggested threshold (ideally calibrated on a held-out split, not the test set).

### Docker
```bash
docker build -f docker/Dockerfile -t defect-inspector .

# Build the engine inside the container (matching TRT version + your GPU)
docker run --gpus all -v $(pwd)/exports:/app/exports --entrypoint python defect-inspector \
  src/optimize_trt.py --onnx /app/exports/patchcore_cable.onnx --fp16

docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/exports:/app/exports -v $(pwd)/results:/app/results \
  defect-inspector --image /app/data/cable/test/bent_wire/000.png --engine /app/exports/patchcore_cable.trt --save-heatmap
```
The image is inference-only (no anomalib/torch); models are mounted, not baked in.

---

## Project Structure

```
defect_detection/
├── src/
│   ├── train.py                  # PatchCore training + test + ONNX export
│   ├── evaluate.py               # Re-run anomalib test loop on a checkpoint
│   ├── export_from_checkpoint.py # ONNX export from a saved checkpoint
│   ├── optimize_trt.py           # TensorRT engine builder (FP16)
│   ├── backends.py               # Shared preprocessing + ONNX/TRT inference wrappers
│   ├── inference.py              # CLI: single image, eval, benchmark
│   └── benchmark.py              # Backend comparison table
├── notebooks/
│   └── colab_trt_export.ipynb    # Build + benchmark a TRT engine on a Colab T4
├── docker/
│   └── Dockerfile
├── data/                # MVTec AD (git-ignored)
├── mlruns/              # anomalib output dir: checkpoints/results (git-ignored)
├── exports/             # ONNX + TRT engines (git-ignored)
├── results/heatmaps/    # Output visualizations (git-ignored)
├── requirements.txt
└── README.md
```

Note: `mlruns/` is anomalib's output directory; no MLflow logger is configured.

---

## Discussion & Engineering Notes

**Why not a supervised model?**
In real industrial settings, defect examples are rare and imbalanced. PatchCore requires only normal training images — more realistic for production. The tradeoff is that it's less precise at localizing subtle defects compared to a fine-tuned segmentation model. For a production system, I'd combine PatchCore as a fast first-pass filter with a lighter classifier for confirmed defect regions.

**TensorRT FP16 tradeoffs**
FP16 should be close to lossless here because anomaly scoring is based on Euclidean distances in feature space, which are robust to small precision changes — verify by comparing `--eval` AUROC for the ONNX model and the FP16 engine. INT8 is not implemented yet; it would need a calibrator (or Q/DQ quantization) and would likely degrade AUROC more.

**Scaling to a real production line**
The current pipeline processes single images. For a real inline system (as in cable manufacturing), you'd add: (1) a frame buffer with async preprocessing, (2) multi-GPU load balancing, (3) a sliding threshold that adapts to production drift.

**What I'd do differently with more time**
- INT8 calibration and an accuracy/latency comparison against FP16
- A C++ inference wrapper to remove Python overhead
- Evaluate EfficientAD as a lower-latency alternative
- A simple Gradio demo for interactive threshold tuning

---

## References

- [PatchCore paper](https://arxiv.org/abs/2106.08265) — Roth et al., CVPR 2022
- [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) — Bergmann et al.
- [Anomalib](https://github.com/openvinotoolkit/anomalib) — Intel's anomaly detection library
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)

---

## License

MIT

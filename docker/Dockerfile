# Inference image: TensorRT 10.x + CUDA 12 + Python 3.10
# Build from the project root:
#   docker build -f docker/Dockerfile -t defect-inspector .
FROM nvcr.io/nvidia/tensorrt:24.08-py3

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Inference-only Python deps (training deps like anomalib/torch are not needed here).
# Version specifiers are quoted so the shell doesn't treat ">" as a redirect.
RUN pip install --no-cache-dir \
    "numpy>=1.24,<2" \
    "Pillow>=9.0" \
    "opencv-python-headless>=4.7" \
    "onnxruntime-gpu>=1.18" \
    "scikit-learn>=1.3" \
    "pycuda>=2022.1"

# App code. Models are NOT baked in (exports/ is git-ignored) — mount them at run time.
COPY src/ ./src/

ENTRYPOINT ["python", "src/inference.py"]
CMD ["--help"]

# Usage (from the project root):
#   # Build the TRT engine inside the container so its TensorRT version and GPU match:
#   docker run --gpus all -v $(pwd)/exports:/app/exports --entrypoint python defect-inspector \
#     src/optimize_trt.py --onnx /app/exports/patchcore_cable.onnx --fp16
#
#   docker run --gpus all \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/exports:/app/exports \
#     -v $(pwd)/results:/app/results \
#     defect-inspector \
#     --image /app/data/cable/test/bent_wire/000.png \
#     --engine /app/exports/patchcore_cable.trt \
#     --save-heatmap

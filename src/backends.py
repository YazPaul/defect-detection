"""
backends.py — Shared preprocessing and ONNX Runtime / TensorRT inference wrappers.

Both inspectors return (anomaly_score, heatmap_or_None) and look up outputs by
name ("pred_score", "anomaly_map") rather than by position, so they work with
either anomalib export:
  * with post-processor   -> outputs: pred_score, pred_label, anomaly_map, pred_mask
                             (scores normalized to [0, 1])
  * without post-processor -> outputs: pred_score, anomaly_map (raw distances)
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

IMAGE_SIZE = 256


def preprocess(image_path: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Load an image as float32 in [0, 1], shape (1, 3, H, W).

    ImageNet mean/std normalization is NOT applied here: anomalib embeds its
    pre-processor in the exported ONNX graph (first ops are Sub/Div by the
    ImageNet statistics), so normalizing here would apply it twice.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[np.newaxis])


def _parse_outputs(outputs: dict) -> tuple[float, np.ndarray | None]:
    """Pick score + heatmap out of a {name: array} dict."""
    score_arr = outputs.get("pred_score")
    if score_arr is None:
        score_arr = next(iter(outputs.values()))
    score = float(np.asarray(score_arr, dtype=np.float32).ravel()[0])

    heatmap = outputs.get("anomaly_map")
    if heatmap is not None:
        heatmap = np.asarray(heatmap, dtype=np.float32)
        heatmap = heatmap.reshape(heatmap.shape[-2:])  # (B,1,H,W) -> (H,W)
    return score, heatmap


class ONNXInspector:
    """Inference using ONNX Runtime (CUDA EP if available, else CPU)."""

    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("pip install onnxruntime-gpu  (or onnxruntime for CPU)")

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.provider = self.session.get_providers()[0]
        log.info(f"ONNX session loaded | provider: {self.provider}")

    def infer(self, image: np.ndarray) -> tuple[float, np.ndarray | None]:
        outs = self.session.run(None, {self.input_name: image})
        return _parse_outputs(dict(zip(self.output_names, outs)))


class TRTInspector:
    """Inference using a serialized TensorRT engine.

    Uses the name-based tensor API (set_tensor_address / execute_async_v3),
    which works on TensorRT 8.5+ and 10.x. Note that engines are tied to the
    TensorRT version and GPU architecture they were built with.
    """

    def __init__(self, trt_path: str, batch_size: int = 1):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401  (creates CUDA context)
        except ImportError:
            raise ImportError("TensorRT + pycuda required: see README")

        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        with open(trt_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize {trt_path}. Engines must be built with the same "
                "TensorRT version and GPU architecture they run on — rebuild with optimize_trt.py."
            )
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = None
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        # Resolve dynamic dims (e.g. batch = -1) before sizing buffers
        in_shape = list(self.engine.get_tensor_shape(self.input_name))
        if in_shape[0] == -1:
            in_shape[0] = batch_size
        self.context.set_input_shape(self.input_name, tuple(in_shape))

        self.buffers = {}
        for name in [self.input_name, *self.output_names]:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(device))
            self.buffers[name] = (host, device, shape)
        log.info(f"TensorRT engine loaded: {trt_path} | outputs: {self.output_names}")

    def infer(self, image: np.ndarray) -> tuple[float, np.ndarray | None]:
        cuda = self.cuda
        host, device, shape = self.buffers[self.input_name]
        if image.shape != shape:
            raise ValueError(f"Input shape {image.shape} != engine input shape {shape}")
        np.copyto(host, image.ravel().astype(host.dtype, copy=False))
        cuda.memcpy_htod_async(device, host, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for name in self.output_names:
            h, d, _ = self.buffers[name]
            cuda.memcpy_dtoh_async(h, d, self.stream)
        self.stream.synchronize()
        outs = {n: self.buffers[n][0].reshape(self.buffers[n][2]) for n in self.output_names}
        return _parse_outputs(outs)

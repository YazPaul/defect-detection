"""
optimize_trt.py — Convert ONNX model to TensorRT engine

Usage:
    python src/optimize_trt.py --onnx exports/patchcore_cable.onnx --fp16

Build the engine on the machine (TensorRT version + GPU architecture) where it
will run — serialized engines are not portable across either.
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", type=str, required=True, help="Path to input ONNX file")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision (recommended)")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 precision — not implemented yet (needs a calibrator)")
    parser.add_argument("--workspace-gb", type=float, default=4.0,
                        help="Max GPU workspace in GB for TRT optimization")
    parser.add_argument("--batch", type=int, nargs=3, default=[1, 1, 1], metavar=("MIN", "OPT", "MAX"),
                        help="Batch sizes for the optimization profile if the ONNX batch dim is dynamic")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .trt path (default: same as ONNX with .trt extension)")
    return parser.parse_args()


def build_engine(args):
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT not installed. See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/"
        )

    if args.int8:
        raise NotImplementedError(
            "INT8 needs a calibrator (IInt8EntropyCalibrator2) or a Q/DQ-quantized ONNX; "
            "neither is implemented. Use --fp16."
        )

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    output_path = Path(args.output) if args.output else onnx_path.with_suffix(".trt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_major = int(trt.__version__.split(".")[0])

    log.info(f"TensorRT    : {trt.__version__}")
    log.info(f"Input ONNX  : {onnx_path}")
    log.info(f"Output TRT  : {output_path}")
    log.info(f"FP16        : {args.fp16}")
    log.info(f"Workspace   : {args.workspace_gb} GB")

    builder = trt.Builder(TRT_LOGGER)
    # Explicit batch is the default (and the flag deprecated) from TensorRT 10
    flags = 0 if trt_major >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser_onnx = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))

    if args.fp16:
        if not getattr(builder, "platform_has_fast_fp16", True):
            log.warning("GPU does not support fast FP16 — falling back to FP32")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            log.info("FP16 mode enabled")

    # Parse ONNX (parse_from_file resolves external-data weights next to the .onnx)
    log.info("Parsing ONNX...")
    if not parser_onnx.parse_from_file(str(onnx_path)):
        for i in range(parser_onnx.num_errors):
            log.error(parser_onnx.get_error(i))
        raise RuntimeError("ONNX parsing failed")

    # anomalib exports a dynamic batch dim -> an optimization profile is mandatory
    profile = None
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        if any(d == -1 for d in shape[1:]):
            raise ValueError(f"Input {inp.name} has dynamic non-batch dims {shape}; not supported")
        if shape[0] == -1:
            profile = profile or builder.create_optimization_profile()
            mn, opt, mx = ([b, *shape[1:]] for b in args.batch)
            profile.set_shape(inp.name, mn, opt, mx)
            log.info(f"Profile     : {inp.name} min={mn} opt={opt} max={mx}")
    if profile is not None:
        config.add_optimization_profile(profile)

    log.info("Building TensorRT engine (this may take several minutes)...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")
    log.info(f"Engine built in {time.time() - t0:.1f}s")

    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = output_path.stat().st_size / (1 << 20)
    log.info(f"Engine saved → {output_path} ({size_mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    build_engine(parse_args())

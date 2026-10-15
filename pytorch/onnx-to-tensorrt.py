"""
Build a TensorRT engine (.plan) from an ONNX model.

    python onnx-to-tensorrt.py --onnx macondo-nn.onnx --output macondo-nn.engine
    python onnx-to-tensorrt.py --onnx x.onnx --output x.engine --precision fp32

TensorRT 11 only builds strongly typed networks: there are no FP16/INT8
builder flags any more, precision comes from the data types in the ONNX
graph. So `--precision fp16` converts the graph to fp16 in memory first
(weights and activations), keeping the graph inputs/outputs fp32 so the
Triton config (TYPE_FP32 board/scalars/value) is unchanged. The legacy
INT8 calibration API was removed in TensorRT 11 (and never worked well for
us), so it is gone here too.

The engine only loads on the exact TensorRT version that built it, which
must match the Triton container (see the NVIDIA frameworks support matrix).
"""

import argparse

import onnx
import tensorrt as trt


def to_fp16(model):
    """Convert weights and activations to fp16, keep graph I/O fp32."""
    from onnxconverter_common import float16

    return float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
    )


def build_engine_from_onnx(
    onnx_file_path,
    engine_file_path,
    precision="fp16",
    max_batch_size=128,
    max_workspace_size=1 << 30,
    tf32=True,
):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    model = onnx.load(onnx_file_path)
    if precision == "fp16":
        model = to_fp16(model)
        print("Converted ONNX graph to fp16 (fp32 inputs/outputs kept)")
    else:
        print("Using FP32 precision" + (" with TF32 tensor cores" if tf32 else ""))

    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model.SerializeToString()):
        for error in range(parser.num_errors):
            print(f"ONNX parsing error: {parser.get_error(error)}")
        return None

    # Optimization profile for the dynamic batch axis. Every input keeps its
    # non-batch dims; only the batch range is set.
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        dims = list(t.shape[1:])
        profile.set_shape(
            t.name,
            (1, *dims),
            (max_batch_size // 2, *dims),
            (max_batch_size, *dims),
        )
        print(f"input {t.name}: {t.dtype} batch 1..{max_batch_size} x {dims}")
    config.add_optimization_profile(profile)

    print("Building TensorRT engine, this may take a few minutes...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to create TensorRT engine")
        return None

    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_file_path}")
    return engine_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument(
        "--output",
        type=str,
        default="scrabble_model.engine",
        help="Output engine file path",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16"],
        help="fp16 converts the ONNX graph to half precision before building",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=128, help="Maximum batch size"
    )
    parser.add_argument(
        "--no-tf32", action="store_true", help="disable TF32 tensor cores for fp32"
    )
    args = parser.parse_args()

    build_engine_from_onnx(
        args.onnx,
        args.output,
        args.precision,
        args.max_batch_size,
        tf32=not args.no_tf32,
    )

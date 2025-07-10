import os
import argparse
import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

def build_engine_from_onnx(onnx_file_path, engine_file_path, precision='fp16',
                          max_batch_size=128, max_workspace_size=1<<30):
    """
    Builds a TensorRT engine from an ONNX model

    Args:
        onnx_file_path: Path to the ONNX model
        engine_file_path: Path to save the TensorRT engine
        precision: 'fp32', 'fp16', or 'int8'
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace size in bytes

    Returns:
        The path to the generated engine file
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # Create network with explicit batch flag
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    # Note: max_batch_size is handled via optimization profiles for explicit batch networks

    # Set precision flags
    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    elif precision == 'int8' and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("Using INT8 precision (calibrator would be needed for proper INT8 quantization)")
    else:
        print("Using FP32 precision")

    # Parse ONNX file
    parser = trt.OnnxParser(network, logger)
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parsing error: {parser.get_error(error)}")
            return None

    print(f"Building TensorRT engine, this may take a few minutes...")

    # Set optimization profiles for dynamic batch size
    profile = builder.create_optimization_profile()
    profile.set_shape("board", (1, C, H, W), (max_batch_size//2, C, H, W), (max_batch_size, C, H, W))
    profile.set_shape("scalars", (1, N_SCAL), (max_batch_size//2, N_SCAL), (max_batch_size, N_SCAL))
    config.add_optimization_profile(profile)

    # Build and save engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to create TensorRT engine")
        return None

    # Serialize the engine
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_file_path}")
    return engine_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ONNX model to TensorRT')
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--output', type=str, default='scrabble_model.engine', help='Output engine file path')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'],
                        help='Precision for TensorRT engine')
    parser.add_argument('--max-batch-size', type=int, default=128, help='Maximum batch size')

    args = parser.parse_args()

    # Import C, H, W, N_SCAL from training.py
    from training import C, H, W, N_SCAL

    build_engine_from_onnx(
        args.onnx,
        args.output,
        args.precision,
        args.max_batch_size
    )
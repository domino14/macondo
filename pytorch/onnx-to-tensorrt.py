import os
import argparse
import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import struct


class FileCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_file, batch_size=8, cache_file="calibration.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current_index = 0
        self.done = False

        # Load calibration data from file
        self.boards = []
        self.scalars = []

        print(f"Loading calibration data from {calibration_file}")
        with open(calibration_file, "rb") as f:
            while True:
                try:
                    # Read header - 4 bytes for the size
                    hdr = f.read(4)
                    if not hdr or len(hdr) < 4:
                        break

                    (n_bytes,) = struct.unpack("<I", hdr)

                    # Read payload
                    payload = f.read(n_bytes)
                    if len(payload) != n_bytes:
                        break

                    # Process the data
                    N_PLANE = C * H * W  # Calculate from dimensions
                    vec = np.frombuffer(payload, dtype=np.float32).copy()
                    board = vec[:N_PLANE].reshape(C, H, W).astype(np.float32)
                    scalars = vec[N_PLANE : N_PLANE + N_SCAL].astype(np.float32)

                    # Store the data
                    self.boards.append(board)
                    self.scalars.append(scalars)

                    if len(self.boards) % 100 == 0:
                        print(f"Loaded {len(self.boards)} calibration samples")

                except (IOError, struct.error) as e:
                    print(f"Error reading calibration data: {e}")
                    break

        print(f"Loaded {len(self.boards)} calibration samples")

        if len(self.boards) == 0:
            raise ValueError("No calibration data loaded")

        # Convert to numpy arrays
        self.boards = np.array(self.boards, dtype=np.float32)
        self.scalars = np.array(self.scalars, dtype=np.float32)

        # Allocate device memory for batches
        self.board_device = cuda.mem_alloc(
            self.batch_size * C * H * W * 4
        )  # 4 bytes per float
        self.scalar_device = cuda.mem_alloc(self.batch_size * N_SCAL * 4)

        # Device input pointers
        self.device_inputs = [int(self.board_device), int(self.scalar_device)]

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.boards) or self.done:
            return None

        end_idx = min(self.current_index + self.batch_size, len(self.boards))
        if end_idx - self.current_index < self.batch_size:
            # Pad the last batch if needed
            board_batch = np.zeros((self.batch_size, C, H, W), dtype=np.float32)
            scalar_batch = np.zeros((self.batch_size, N_SCAL), dtype=np.float32)

            actual_size = end_idx - self.current_index
            board_batch[:actual_size] = self.boards[self.current_index : end_idx]
            scalar_batch[:actual_size] = self.scalars[self.current_index : end_idx]
        else:
            board_batch = self.boards[self.current_index : end_idx]
            scalar_batch = self.scalars[self.current_index : end_idx]

        # Copy to device
        cuda.memcpy_htod(self.board_device, np.ascontiguousarray(board_batch))
        cuda.memcpy_htod(self.scalar_device, np.ascontiguousarray(scalar_batch))

        self.current_index += self.batch_size
        if self.current_index >= len(self.boards):
            self.done = True

        return self.device_inputs

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine_from_onnx(
    onnx_file_path,
    engine_file_path,
    precision="fp16",
    max_batch_size=128,
    max_workspace_size=1 << 30,
    calibration_file=None,
    calibration_cache="calibration.cache",
):
    """
    Builds a TensorRT engine from an ONNX model

    Args:
        onnx_file_path: Path to the ONNX model
        engine_file_path: Path to save the TensorRT engine
        precision: 'fp32', 'fp16', or 'int8'
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace size in bytes
        calibration_file: File containing calibration data for INT8
        calibration_cache: Path to save/load calibration cache

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

    # Parse ONNX file
    parser = trt.OnnxParser(network, logger)
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parsing error: {parser.get_error(error)}")
            return None

    # Set precision flags
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)

        if calibration_file:
            print(f"Using INT8 precision with calibrator from {calibration_file}")
            # Create calibrator with data from file
            calibrator = FileCalibrator(
                calibration_file=calibration_file,
                batch_size=8,  # Small batch size for calibration
                cache_file=calibration_cache,
            )
            config.int8_calibrator = calibrator
            print("Calibrator created successfully")
        else:
            print("Using INT8 precision with static ranges")
            # Set dynamic ranges for all tensors
            for i in range(network.num_inputs):
                tensor = network.get_input(i)
                if tensor.name:
                    print(f"Setting range for input: {tensor.name}")
                    tensor.dynamic_range = (0.0, 1.0)  # normalized inputs

            for i in range(network.num_outputs):
                tensor = network.get_output(i)
                if tensor.name:
                    print(f"Setting range for output: {tensor.name}")
                    tensor.dynamic_range = (-1.0, 1.0)  # typical output range

            for i in range(network.num_layers):
                layer = network.get_layer(i)
                for j in range(layer.num_outputs):
                    tensor = layer.get_output(j)
                    if tensor.name:
                        if "board" in tensor.name or "scalars" in tensor.name:
                            tensor.dynamic_range = (0.0, 1.0)  # normalized inputs
                        else:
                            tensor.dynamic_range = (
                                -6.0,
                                6.0,
                            )  # Conservative default range for activations
    else:
        print("Using FP32 precision")

    print(f"Building TensorRT engine, this may take a few minutes...")

    # Set optimization profiles for dynamic batch size
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "board", (1, C, H, W), (max_batch_size // 2, C, H, W), (max_batch_size, C, H, W)
    )
    profile.set_shape(
        "scalars", (1, N_SCAL), (max_batch_size // 2, N_SCAL), (max_batch_size, N_SCAL)
    )
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
        choices=["fp32", "fp16", "int8"],
        help="Precision for TensorRT engine",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=128, help="Maximum batch size"
    )
    parser.add_argument(
        "--calib-file",
        type=str,
        default=None,
        help="Path to calibration data file for INT8 precision",
    )
    parser.add_argument(
        "--calib-cache",
        type=str,
        default="calibration.cache",
        help="Path to calibration cache file",
    )

    args = parser.parse_args()

    # Import C, H, W, N_SCAL from training.py
    from training import C, H, W, N_SCAL

    build_engine_from_onnx(
        args.onnx,
        args.output,
        args.precision,
        args.max_batch_size,
        calibration_file=args.calib_file,
        calibration_cache=args.calib_cache,
    )

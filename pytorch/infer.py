import struct
import numpy as np
import torch
import os


def run_inference(vector_path="/tmp/test-vec-infer.bin", model_path="best.pt"):
    """
    Run inference on a binary vector file using the trained model.

    Args:
        vector_path: Path to the binary vector file
        model_path: Path to the saved model checkpoint

    Returns:
        The predicted value (unscaled)
    """
    # Constants from your training code
    C = 83  # planes
    H = W = 15
    N_PLANE = C * H * W  # 18,675
    N_SCAL = 58

    # Check if files exist
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize the model architecture
    from training import ScrabbleValueNet  # Import your model class

    model = ScrabbleValueNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Read the binary vector file
    with open(vector_path, "rb") as f:
        # Read the length prefix (uint32 little-endian)
        len_bytes = f.read(4)
        n_bytes = struct.unpack("<I", len_bytes)[0]

        # Read the payload
        payload = f.read(n_bytes)
        if len(payload) != n_bytes:
            raise ValueError(f"Expected {n_bytes} bytes, got {len(payload)}")

        # Convert to float32 array
        vec = np.frombuffer(payload, dtype=np.float32)

        # Split into board and scalar components
        board = torch.from_numpy(vec[:N_PLANE]).view(1, C, H, W)  # Add batch dimension
        scalars = torch.from_numpy(vec[N_PLANE : N_PLANE + N_SCAL]).view(
            1, -1
        )  # Add batch dimension

    # Move to device
    board = board.to(device)
    scalars = scalars.to(device)

    # Run inference
    with torch.no_grad():
        with torch.amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu"
        ):
            prediction = model(board, scalars).item()

    # Denormalize the prediction (since model outputs are scaled to [-1, 1])
    # The scaling used in your Go code was: evals[i] *= 300.0
    unscaled_prediction = prediction * 300.0

    print(f"Raw model output: {prediction:.6f}")
    print(f"Unscaled prediction: {unscaled_prediction:.2f}")

    return unscaled_prediction


if __name__ == "__main__":
    score = run_inference()
    print(f"Predicted score: {score}")

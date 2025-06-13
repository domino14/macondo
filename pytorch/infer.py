import struct
import numpy as np
import torch
import os
import glob
import csv
import argparse


def run_inference_batch(
    vector_paths, model_path="best.pt", output_csv=None, batch_size=32
):
    """
    Run inference on multiple binary vector files using the trained model.

    Args:
        vector_paths: List of paths to binary vector files or a directory containing vector files
        model_path: Path to the saved model checkpoint
        output_csv: Optional path to save results to CSV
        batch_size: Number of vectors to process at once for GPU efficiency

    Returns:
        Dictionary mapping file paths to predictions
    """
    # Constants from your training code
    C = 83  # planes
    H = W = 15
    N_PLANE = C * H * W  # 18,675
    N_SCAL = 58

    # Expand directory if provided
    if isinstance(vector_paths, str) and os.path.isdir(vector_paths):
        vector_paths = glob.glob(os.path.join(vector_paths, "*.bin"))
        print(f"Found {len(vector_paths)} vector files in directory")

    # Ensure we have a list of paths
    if isinstance(vector_paths, str):
        vector_paths = [vector_paths]

    # Check if model exists
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

    results = {}

    # Process vectors in batches
    for i in range(0, len(vector_paths), batch_size):
        batch_paths = vector_paths[i : i + batch_size]
        batch_boards = []
        batch_scalars = []
        valid_paths = []

        # Load all vectors in current batch
        for path in batch_paths:
            try:
                # Read the binary vector file
                with open(path, "rb") as f:
                    # Read the length prefix (uint32 little-endian)
                    len_bytes = f.read(4)
                    n_bytes = struct.unpack("<I", len_bytes)[0]

                    # Read the payload
                    payload = f.read(n_bytes)
                    if len(payload) != n_bytes:
                        print(
                            f"Warning: Expected {n_bytes} bytes, got {len(payload)} in {path}"
                        )
                        continue

                    # Convert to float32 array
                    vec = np.frombuffer(payload, dtype=np.float32)

                    # Split into board and scalar components
                    board = torch.from_numpy(vec[:N_PLANE]).view(1, C, H, W)
                    scalars = torch.from_numpy(vec[N_PLANE : N_PLANE + N_SCAL]).view(
                        1, -1
                    )

                    batch_boards.append(board)
                    batch_scalars.append(scalars)
                    valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        if not valid_paths:
            continue

        # Concatenate tensors from batch
        batch_boards = torch.cat(batch_boards).to(device)
        batch_scalars = torch.cat(batch_scalars).to(device)

        # Run inference on batch
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu"
            ):
                predictions = model(batch_boards, batch_scalars).cpu().numpy()

        # Store results
        for path, prediction in zip(valid_paths, predictions):
            results[path] = prediction

    # Print summary
    print(f"Processed {len(results)}/{len(vector_paths)} vector files successfully")

    # Save to CSV if requested
    if output_csv:
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["File", "Raw Prediction"])
            for path, prediction in results.items():
                writer.writerow([path, prediction])
        print(f"Results saved to {output_csv}")

    return results


def run_inference(vector_path="/tmp/test-vec-infer.bin", model_path="best.pt"):
    """
    Run inference on a single binary vector file using the trained model.
    """
    results = run_inference_batch([vector_path], model_path)
    prediction = results[vector_path]

    print(f"prediction: {prediction:.2f}")

    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on Scrabble position vectors"
    )
    parser.add_argument(
        "--input",
        default="/tmp/test-vec-infer.bin",
        help="Input vector file, directory of vector files, or comma-separated list of files",
    )
    parser.add_argument("--model", default="best.pt", help="Path to model checkpoint")
    parser.add_argument("--output", help="CSV file to save results")
    parser.add_argument(
        "--batch", type=int, default=32, help="Batch size for processing"
    )
    args = parser.parse_args()

    # Handle comma-separated list of files
    if "," in args.input:
        input_paths = args.input.split(",")
    else:
        input_paths = args.input

    # Run batch inference
    results = run_inference_batch(input_paths, args.model, args.output, args.batch)

    # Print first few results if not saving to CSV
    if not args.output:
        print("\nResults (first 5):")
        for i, (path, prediction) in enumerate(results.items()):
            if i >= 5:
                break
            print(f"{os.path.basename(path)}: {prediction:.2f}")

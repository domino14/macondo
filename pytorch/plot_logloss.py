#!/usr/bin/env python3
"""
Plot training and validation loss from the loss_log.csv file,
with support for ignoring extra headers or comment lines.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
import os
import csv


def plot_loss(
    csv_path="loss_log.csv",
    output_path=None,
    dark_mode=False,
    skip_rows=0,
    comment_char=None,
):
    """
    Plot training and validation loss from a CSV file.

    Args:
        csv_path: Path to the CSV file with loss data
        output_path: Where to save the plot (if None, just display)
        dark_mode: Whether to use a dark theme for the plot
        skip_rows: Number of rows to skip at the beginning of the file
        comment_char: Character that marks comment lines to ignore
    """
    # Set style
    if dark_mode:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found")
        return

    # First, peek at the file to handle potential issues
    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Print first few lines for debugging
    print(f"First 5 lines of {csv_path}:")
    for i, line in enumerate(lines[:5]):
        print(f"Line {i}: {line.strip()}")

    # Load data with options to handle headers
    df = pd.read_csv(csv_path, skiprows=skip_rows, comment=comment_char)

    # Make sure required columns exist
    required_columns = ["step", "train_loss", "val_loss"]
    if not all(col in df.columns for col in required_columns):
        print(f"Warning: Missing required columns. Found: {df.columns.tolist()}")

        # Try to infer column names if not found
        if len(df.columns) >= 3 and all(
            col not in df.columns for col in required_columns
        ):
            print("Renaming columns to required names")
            df.columns = required_columns[: len(df.columns)]

    # Convert string values to float if needed
    for col in ["train_loss", "val_loss"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype(float)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data
    ax.plot(df["step"], df["train_loss"], "b-", label="Training Loss", linewidth=2)
    ax.plot(df["step"], df["val_loss"], "r-", label="Validation Loss", linewidth=2)

    # Add moving average for smoother visualization
    window = 3  # Window size for moving average
    if len(df) > window:
        train_smooth = df["train_loss"].rolling(window=window).mean()
        val_smooth = df["val_loss"].rolling(window=window).mean()
        ax.plot(df["step"], train_smooth, "b--", alpha=0.7, linewidth=1)
        ax.plot(df["step"], val_smooth, "r--", alpha=0.7, linewidth=1)

    # Add min point markers
    min_val_idx = df["val_loss"].idxmin()
    min_val = df.loc[min_val_idx]
    ax.scatter(
        min_val["step"],
        min_val["val_loss"],
        color="lime",
        s=100,
        zorder=5,
        label=f"Best val: {min_val['val_loss']:.6f} (step {int(min_val['step'])})",
    )

    # Customize plot
    ax.set_title("Training and Validation Loss", fontsize=16)
    ax.set_xlabel("Steps", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis to start from 0 or slightly below min value
    min_y = min(df["train_loss"].min(), df["val_loss"].min())
    ax.set_ylim(max(0, min_y * 0.95), None)

    # Add legend
    ax.legend(loc="upper right", fontsize=12)

    # Add data points info in text box
    num_points = len(df)
    last_val = df["val_loss"].iloc[-1]
    textstr = f"Data points: {num_points}\nLast val loss: {last_val:.6f}"
    props = dict(boxstyle="round", alpha=0.5)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation loss")
    parser.add_argument("--csv", default="loss_log.csv", help="Path to the CSV file")
    parser.add_argument(
        "--output", help="Output path for the plot (if not provided, will display)"
    )
    parser.add_argument("--dark", action="store_true", help="Use dark mode")
    parser.add_argument(
        "--skip", type=int, default=0, help="Number of rows to skip at the beginning"
    )
    parser.add_argument(
        "--comment", help="Character that marks comment lines to ignore"
    )
    args = parser.parse_args()

    plot_loss(args.csv, args.output, args.dark, args.skip, args.comment)

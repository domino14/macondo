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
    components_only=False,
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
    base_required_columns = ["step", "train_loss", "val_loss"]
    additional_columns = ["value_loss", "points_loss", "bingo_loss", "opp_score_loss"]
    required_columns = base_required_columns + additional_columns

    # Check if we have the basic columns at minimum
    if not all(col in df.columns for col in base_required_columns):
        print(f"Warning: Missing required base columns. Found: {df.columns.tolist()}")

        # Try to infer column names if not found
        if len(df.columns) >= 3 and all(
            col not in df.columns for col in base_required_columns
        ):
            print("Renaming columns to required names")
            df.columns = required_columns[: len(df.columns)]

    # Check for additional columns (component losses)
    has_component_losses = all(col in df.columns for col in additional_columns)
    if not has_component_losses:
        print("Note: Component loss columns not found. Will only plot overall loss.")

    # Convert string values to float if needed
    for col in df.columns:
        if col != "step" and df[col].dtype == "object":
            df[col] = df[col].astype(float)

    # Check for component losses
    additional_columns = ["value_loss", "points_loss", "bingo_loss", "opp_score_loss"]
    has_component_losses = all(col in df.columns for col in additional_columns)

    if components_only and has_component_losses:
        # Create figure with only component loss plots (2x2 grid)
        component_names = ["value_loss", "points_loss", "bingo_loss", "opp_score_loss"]
        component_titles = [
            "Value Loss",
            "Points Loss",
            "Bingo Probability Loss",
            "Opponent Score Loss",
        ]
        component_colors = ["blue", "green", "purple", "orange"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, (col, title, color) in enumerate(
            zip(component_names, component_titles, component_colors)
        ):
            ax = axes[i]
            ax.plot(df["step"], df[col], color=color, linewidth=2, label=title)

            # Add moving average
            window = 3
            if len(df) > window:
                smooth = df[col].rolling(window=window).mean()
                ax.plot(df["step"], smooth, "--", color=color, alpha=0.7, linewidth=1)

            # Mark minimum
            min_idx = df[col].idxmin()
            min_val = df.loc[min_idx]
            ax.scatter(
                min_val["step"],
                min_val[col],
                color="lime",
                s=80,
                zorder=5,
                label=f"Min: {min_val[col]:.4f} (step {int(min_val['step'])})",
            )

            # Customize subplot
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Steps", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(loc="upper right")

            # Set reasonable y limits using percentiles to handle outliers
            min_y = df[col].min()
            max_y = np.percentile(df[col], 95)  # Use 95th percentile instead of max
            ax.set_ylim(max(0, min_y * 0.9), max_y * 1.1)  # Add 10% padding to the top

        main_ax = None  # No main axis in components-only mode

    else:
        # Original layout with main plot + component plots if available
        if has_component_losses and not components_only:
            fig, axes = plt.subplots(3, 2, figsize=(15, 15))  # Changed from 2x2 to 3x2
            axes = axes.flatten()
            main_ax = axes[0]  # First subplot for overall loss
        else:
            fig, main_ax = plt.subplots(figsize=(10, 6))
            axes = [main_ax]

    # Plot overall loss in the main subplot if it exists
    if main_ax is not None:
        main_ax.plot(
            df["step"], df["train_loss"], "b-", label="Training Loss", linewidth=2
        )
        main_ax.plot(
            df["step"], df["val_loss"], "r-", label="Validation Loss", linewidth=2
        )

        # Add moving average for smoother visualization
        window = 3  # Window size for moving average
        if len(df) > window:
            train_smooth = df["train_loss"].rolling(window=window).mean()
            val_smooth = df["val_loss"].rolling(window=window).mean()
            main_ax.plot(df["step"], train_smooth, "b--", alpha=0.7, linewidth=1)
            main_ax.plot(df["step"], val_smooth, "r--", alpha=0.7, linewidth=1)

        # Add min point markers
        min_val_idx = df["val_loss"].idxmin()
        min_val = df.loc[min_val_idx]
        main_ax.scatter(
            min_val["step"],
            min_val["val_loss"],
            color="lime",
            s=100,
            zorder=5,
            label=f"Best val: {min_val['val_loss']:.6f} (step {int(min_val['step'])})",
        )

        # Customize main plot
        main_ax.set_title("Overall Training and Validation Loss", fontsize=16)
        main_ax.set_xlabel("Steps", fontsize=14)
        main_ax.set_ylabel("Loss", fontsize=14)
        main_ax.grid(True, linestyle="--", alpha=0.7)

        # Set y-axis to start from 0 or slightly below min value
        min_y = min(df["train_loss"].min(), df["val_loss"].min())
        main_ax.set_ylim(max(0, min_y * 0.95), None)

        # Add legend
        main_ax.legend(loc="upper right", fontsize=12)

        # Add data points info in text box
        num_points = len(df)
        last_val = df["val_loss"].iloc[-1]
        textstr = f"Data points: {num_points}\nLast val loss: {last_val:.6f}"
        props = dict(boxstyle="round", alpha=0.5)
        main_ax.text(
            0.05,
            0.95,
            textstr,
            transform=main_ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

    # Plot component losses if available and we're in the mixed mode (not components_only)
    if has_component_losses and not components_only and main_ax is not None:
        component_plots = [
            {
                "ax_idx": 1,
                "column": "value_loss",
                "title": "Value Loss",
                "color": "blue",
            },
            {
                "ax_idx": 2,
                "column": "points_loss",
                "title": "Points Loss",
                "color": "green",
            },
            {
                "ax_idx": 3,
                "column": "bingo_loss",
                "title": "Bingo Probability Loss",
                "color": "purple",
            },
            {
                "ax_idx": 4,  # Changed from 3 to 4
                "column": "opp_score_loss",
                "title": "Opponent Score Loss",
                "color": "orange",
            },
        ]

        for plot in component_plots:
            ax = axes[plot["ax_idx"]]
            col = plot["column"]

            ax.plot(df["step"], df[col], color=plot["color"], linewidth=2)

            # Add moving average
            if len(df) > window:
                smooth = df[col].rolling(window=window).mean()
                ax.plot(
                    df["step"],
                    smooth,
                    "--",
                    color=plot["color"],
                    alpha=0.7,
                    linewidth=1,
                )

            # Mark minimum
            min_idx = df[col].idxmin()
            min_val = df.loc[min_idx]
            ax.scatter(
                min_val["step"],
                min_val[col],
                color="lime",
                s=80,
                zorder=5,
                label=f"Min: {min_val[col]:.4f}",
            )

            # Customize subplot
            ax.set_title(plot["title"], fontsize=14)
            ax.set_xlabel("Steps", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(loc="upper right")

            # Set reasonable y limits using percentiles to handle outliers
            min_y = df[col].min()
            max_y = np.percentile(df[col], 95)  # Use 95th percentile instead of max
            ax.set_ylim(max(0, min_y * 0.9), max_y * 1.1)  # Add 10% padding to the top

    # Add a title for the whole figure if we have multiple plots
    if has_component_losses:
        fig.suptitle("Scrabble Neural Network Training Progress", fontsize=20, y=0.98)

    plt.tight_layout()

    # Adjust layout if we have a figure title
    if has_component_losses:
        plt.subplots_adjust(
            top=0.92, hspace=0.3, wspace=0.3
        )  # Added spacing adjustments

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
    parser.add_argument(
        "--components-only",
        action="store_true",
        help="Show only component losses, not the overall loss",
    )
    args = parser.parse_args()

    plot_loss(
        args.csv, args.output, args.dark, args.skip, args.comment, args.components_only
    )

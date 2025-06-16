import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

import string
import os

from training import (
    C,
    H,
    W,
    N_PLANE,
    N_SCAL,
)  # Import constants from your training module


def visualize_vector(vector_path="/tmp/test-vec-234.bin", show_all_planes=True):
    """
    Visualize a binary vector file used for Scrabble ML training.

    Args:
        vector_path: Path to the binary vector file
        show_all_planes: If True, show all 83 planes; otherwise show a summary
    """

    # Check if file exists
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")

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

    # Extract components
    board_data = vec[:N_PLANE].reshape(C, H, W)
    scalar_data = vec[N_PLANE : N_PLANE + N_SCAL]
    target = vec[-1] if len(vec) > N_PLANE + N_SCAL else None

    # Set up the figure
    if show_all_planes:
        fig = plt.figure(figsize=(20, 30))
        gs = GridSpec(15, 7, figure=fig)
    else:
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(3, 3, figure=fig)

    # Define plane groups with their indices
    plane_groups = {
        "Letters (A-Z)": slice(0, 26),
        "Blanks": slice(26, 27),
        "Horizontal Cross-Checks": slice(27, 53),
        "Vertical Cross-Checks": slice(53, 79),
        "Bonus Squares": slice(79, 83),
        "Opp Last Play": slice(83, 84),
    }

    bonus_labels = ["2L", "3L", "2W", "3W"]

    # Create a custom colormap for the board
    cmap = LinearSegmentedColormap.from_list(
        "LightBlues",
        [(1, 1, 1, 1), (0.8, 0.9, 1, 1)],  # From white to very light blue
        N=100,
    )

    # Visualize the planes
    if show_all_planes:
        # Show all 84 planes individually
        for i in range(C):
            ax = fig.add_subplot(gs[i // 7, i % 7])
            im = ax.imshow(board_data[i], cmap=cmap, vmin=0, vmax=1)

            # Add appropriate title
            if i < 26:
                ax.set_title(f"Letter {string.ascii_uppercase[i]}")
            elif i == 26:
                ax.set_title("Blanks")
            elif i < 53:
                ax.set_title(f"H-CC {string.ascii_uppercase[i-27]}")
            elif i < 79:
                ax.set_title(f"V-CC {string.ascii_uppercase[i-53]}")
            elif i < 83:
                ax.set_title(f"Bonus {bonus_labels[i-79]}")
            else:
                ax.set_title("Opponent's Last Play")
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        # Show summary visualizations
        # 1. Letters on board (combine all letter planes)
        ax1 = fig.add_subplot(gs[0, 0])
        letter_sum = np.zeros((H, W))
        letter_board = np.zeros((H, W), dtype=object)

        # Fill letter board with actual letters
        for i in range(26):
            letter = string.ascii_uppercase[i]
            for y in range(H):
                for x in range(W):
                    if board_data[i, y, x] > 0:
                        letter_board[y, x] = letter
                        letter_sum[y, x] = 1

        # Add blanks
        for y in range(H):
            for x in range(W):
                if board_data[26, y, x] > 0:
                    letter_board[y, x] = (
                        letter_board[y, x].lower() if letter_board[y, x] else "?"
                    )

        # Draw the board
        im = ax1.imshow(letter_sum, cmap=cmap, vmin=0, vmax=1)

        # Add letters as text
        for y in range(H):
            for x in range(W):
                if letter_board[y, x]:
                    ax1.text(
                        x,
                        y,
                        letter_board[y, x],
                        ha="center",
                        va="center",
                        color="red" if board_data[26, y, x] > 0 else "black",
                        fontweight="bold",
                    )

        ax1.set_title("Letters on Board")
        ax1.set_xticks(range(W))
        ax1.set_yticks(range(H))

        # 2. Bonus squares
        ax2 = fig.add_subplot(gs[0, 1])
        bonus_board = np.zeros((H, W))
        for i in range(4):
            bonus_idx = i + 79
            for y in range(H):
                for x in range(W):
                    if board_data[bonus_idx, y, x] > 0:
                        bonus_board[y, x] = i + 1

        # Create a custom colormap for bonus squares
        bonus_cmap = mcolors.ListedColormap(
            ["white", "lightblue", "blue", "pink", "red"]
        )
        im2 = ax2.imshow(bonus_board, cmap=bonus_cmap, vmin=0, vmax=4)

        # Add bonus labels
        for y in range(H):
            for x in range(W):
                if bonus_board[y, x] > 0:
                    ax2.text(
                        x,
                        y,
                        bonus_labels[int(bonus_board[y, x]) - 1],
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )

        ax2.set_title("Uncovered Bonus Squares")
        ax2.set_xticks(range(W))
        ax2.set_yticks(range(H))

        # 3. Cross-check visualization (combined)
        ax3 = fig.add_subplot(gs[0, 2])
        h_cc_sum = np.sum(board_data[27:53], axis=0)
        v_cc_sum = np.sum(board_data[53:79], axis=0)
        cc_board = h_cc_sum + v_cc_sum

        im3 = ax3.imshow(cc_board, cmap="hot", vmin=0)
        ax3.set_title("Cross-Check Density")
        ax3.set_xticks(range(W))
        ax3.set_yticks(range(H))
        plt.colorbar(im3, ax=ax3)

        # 4. Rack visualization
        ax4 = fig.add_subplot(gs[1, 0])
        rack_data = scalar_data[:27]
        ax4.bar(["?"] + list(string.ascii_uppercase), rack_data)
        ax4.set_title("Rack (count / 7.0)")
        ax4.set_ylim(0, 1.0)

        # 5. Unseen tiles visualization
        ax5 = fig.add_subplot(gs[1, 1])
        unseen_data = scalar_data[27:54]
        ax5.bar(["?"] + list(string.ascii_uppercase), unseen_data)
        ax5.set_title("Unseen Tiles (count / bagcount)")
        ax5.set_ylim(0, 1.0)

        # 6. Other scalar features
        ax6 = fig.add_subplot(gs[1, 2])
        other_scalars = scalar_data[54:]
        expected_labels = [
            "Last Move Score",
            "Last Move Leave",
            "Tiles Remaining",
            "Spread",
        ]

        # Use only as many labels as we have data for
        labels = expected_labels[: len(other_scalars)]

        # Show what we have
        if len(other_scalars) > 0:
            ax6.bar(labels, other_scalars)
            ax6.set_title(f"Other Features ({len(other_scalars)} values)")

            # Show exact values as text
            for i, v in enumerate(other_scalars):
                ax6.text(i, v + 0.02, f"{v:.3f}", ha="center")
        else:
            ax6.text(
                0.5,
                0.5,
                "No additional features available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax6.axis("off")

        # Show exact values as text
        for i, v in enumerate(other_scalars):
            ax6.text(i, v + 0.02, f"{v:.3f}", ha="center")

        # 7. Display target value if available
        if target is not None:
            ax7 = fig.add_subplot(gs[2, 0:])
            ax7.text(
                0.5,
                0.5,
                f"Target value (1 = win, -1 = loss, 0 = draw): {target:.6f}",
                ha="center",
                va="center",
                fontsize=16,
            )
            ax7.axis("off")

    # Add a button to view raw data
    btn_frame = plt.gcf().add_axes(
        [0.85, 0.01, 0.14, 0.04]
    )  # [left, bottom, width, height]
    raw_data_btn = plt.Button(btn_frame, "View Raw Data")
    raw_data_btn.on_clicked(lambda _: show_raw_vector_data(vec, N_PLANE, H, W))

    plt.tight_layout()
    plt.show()

    return {"board_data": board_data, "scalar_data": scalar_data, "target": target}


def show_raw_vector_data(vec, N_PLANE, H, W):
    """
    Display raw vector data in an interactive hex viewer window.

    Args:
        vec: The raw vector data (numpy array)
        N_PLANE: Total size of all planes
        H, W: Board dimensions
    """
    import tkinter as tk
    from tkinter import ttk

    # Create the window
    root = tk.Toplevel()
    root.title("Vector Raw Data Viewer")
    root.geometry("800x600")

    # Create main frame
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1, padx=10, pady=10)

    # Create controls frame
    controls_frame = ttk.Frame(main_frame)
    controls_frame.pack(fill=tk.X, pady=(0, 10))

    # Jump to index entry and button
    ttk.Label(controls_frame, text="Jump to index:").pack(side=tk.LEFT, padx=(0, 5))
    jump_entry = ttk.Entry(controls_frame, width=10)
    jump_entry.pack(side=tk.LEFT, padx=(0, 5))

    # Jump to plane entry and button
    ttk.Label(controls_frame, text="Jump to plane:").pack(side=tk.LEFT, padx=(10, 5))
    plane_entry = ttk.Entry(controls_frame, width=5)
    plane_entry.pack(side=tk.LEFT, padx=(0, 5))

    # Create text widget with scrollbar
    text_frame = ttk.Frame(main_frame)
    text_frame.pack(fill=tk.BOTH, expand=1)

    scrollbar = ttk.Scrollbar(text_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    text = tk.Text(text_frame, wrap=tk.NONE, font=("Courier", 10))
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Horizontal scrollbar
    h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=text.xview)
    h_scrollbar.pack(fill=tk.X)

    # Configure scrollbars
    text.config(yscrollcommand=scrollbar.set, xscrollcommand=h_scrollbar.set)
    scrollbar.config(command=text.yview)

    # Info display
    info_var = tk.StringVar()
    info_var.set(
        f"Vector length: {len(vec)}, Board planes: {N_PLANE//(H*W)}, Scalars: {len(vec)-N_PLANE}"
    )
    info_label = ttk.Label(main_frame, textvariable=info_var)
    info_label.pack(pady=(5, 0))

    # Value display when hovering
    hover_var = tk.StringVar()
    hover_label = ttk.Label(main_frame, textvariable=hover_var)
    hover_label.pack(pady=(5, 0))

    # Functions
    def populate_text():
        text.delete(1.0, tk.END)

        # Board planes (visualization)
        plane_size = H * W
        num_planes = N_PLANE // plane_size

        # Add board planes
        for p in range(num_planes):
            text.insert(tk.END, f"\n=== PLANE {p} ===\n")
            start_idx = p * plane_size

            # Add column headers
            text.insert(tk.END, "    ")
            for col in range(W):
                text.insert(tk.END, f"{col:3d} ")
            text.insert(tk.END, "\n")

            # Add rows with data
            for row in range(H):
                text.insert(tk.END, f"{row:2d}: ")
                for col in range(W):
                    idx = start_idx + row * W + col
                    val = vec[idx]
                    # Format: use hex for non-zero values, spaces for zeros
                    if val == 0:
                        text.insert(tk.END, "  . ")
                    else:
                        text.insert(tk.END, f"{val:3.1f} ")
                        tag_name = f"nonzero_{idx}"
                        text.tag_add(tag_name, f"insert-4c", "insert")
                        text.tag_config(tag_name, background="#e0f0ff")
                text.insert(tk.END, "\n")

            text.insert(tk.END, "\n")

        # Add scalar values
        text.insert(tk.END, "\n=== SCALAR VALUES ===\n")
        for i in range(N_PLANE, len(vec)):
            text.insert(tk.END, f"[{i:6d}] {vec[i]:10.6f}")
            if (i - N_PLANE) % 5 == 4:  # Group by 5 values per line
                text.insert(tk.END, "\n")
            else:
                text.insert(tk.END, "  |  ")

        # Final target value if present
        if len(vec) > N_PLANE + 58:  # If there's a target value
            text.insert(
                tk.END,
                f"\n\n=== TARGET ===\n[{len(vec)-1}] {vec[-1]:10.6f} (unscaled: {vec[-1]*300.0:10.6f})\n",
            )

    def jump_to_index():
        try:
            idx = int(jump_entry.get())
            if 0 <= idx < len(vec):
                # Calculate line position based on index
                if idx < N_PLANE:
                    # Board plane
                    plane = idx // (H * W)
                    within_plane = idx % (H * W)
                    row = within_plane // W

                    # Rough estimate of line number
                    # Header + previous planes + current plane header + current row
                    line = 1 + plane * (H + 3) + 2 + row

                    text.see(f"{line}.0")
                    info_var.set(
                        f"Index {idx}: Value = {vec[idx]}, Plane {plane}, Row {row}"
                    )
                else:
                    # Scalar section
                    # Jump to the scalar section
                    text.search(
                        "=== SCALAR VALUES ===", "1.0", stopindex=tk.END, forwards=True
                    )
                    info_var.set(f"Index {idx}: Value = {vec[idx]} (scalar)")
            else:
                info_var.set(f"Index out of range (0-{len(vec)-1})")
        except ValueError:
            info_var.set("Please enter a valid integer")

    def jump_to_plane():
        try:
            plane = int(plane_entry.get())
            if 0 <= plane < N_PLANE // (H * W):
                # Calculate the position
                line = 1 + plane * (H + 3)
                text.see(f"{line}.0")
                info_var.set(f"Jumped to plane {plane}")
            else:
                info_var.set(f"Plane out of range (0-{N_PLANE//(H*W)-1})")
        except ValueError:
            info_var.set("Please enter a valid integer")

    # Bind actions to buttons
    jump_button = ttk.Button(
        controls_frame, text="Jump to Index", command=jump_to_index
    )
    jump_button.pack(side=tk.LEFT)

    plane_button = ttk.Button(
        controls_frame, text="Jump to Plane", command=jump_to_plane
    )
    plane_button.pack(side=tk.LEFT, padx=(5, 0))

    # Add export button for saving the raw data
    def export_data():
        from tkinter import filedialog

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Raw Vector Data",
        )
        if file_path:
            with open(file_path, "w") as f:
                f.write(text.get(1.0, tk.END))
            info_var.set(f"Data exported to {file_path}")

    export_button = ttk.Button(controls_frame, text="Export Data", command=export_data)
    export_button.pack(side=tk.RIGHT)

    # Mouse hover to show value
    def on_motion(event):
        try:
            index = text.index(f"@{event.x},{event.y}")
            line, col = map(int, index.split("."))
            # This is a very rough approximation and will need refinement
            hover_var.set(f"Position: Line {line}, Column {col}")
        except Exception as e:
            hover_var.set(str(e))

    text.bind("<Motion>", on_motion)

    # Populate the text
    populate_text()

    # Focus the window
    root.lift()
    root.focus_force()

    return root


if __name__ == "__main__":
    # Example usage
    result = visualize_vector(show_all_planes=False)

    # Detailed exploration - print specific plane sums to check for anomalies
    board_data = result["board_data"]
    print(f"Sum of all letter planes: {np.sum(board_data[0:26])}")
    print(f"Sum of blank plane: {np.sum(board_data[26])}")
    print(f"Sum of H cross-checks: {np.sum(board_data[27:53])}")
    print(f"Sum of V cross-checks: {np.sum(board_data[53:79])}")
    print(f"Sum of bonus planes: {np.sum(board_data[79:83])}")

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


def visualize_vector(vector_path="/tmp/test-vec-9.bin", show_all_planes=True):
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

        # 3. Cross-check visualization (horizontal and vertical side by side in gs[0,2])
        gs_cc = GridSpec(
            1,
            2,
            width_ratios=[1, 1],
            left=gs[0, 2].get_position(fig).xmin,
            right=gs[0, 2].get_position(fig).xmax,
            bottom=gs[0, 2].get_position(fig).ymin,
            top=gs[0, 2].get_position(fig).ymax,
            figure=fig,
        )

        ax3 = fig.add_subplot(gs_cc[0])
        h_cc_sum = np.sum(board_data[27:53], axis=0)
        im3 = ax3.imshow(h_cc_sum, cmap="coolwarm", vmin=0)
        ax3.set_title("Horizontal Cross-Check Density")
        ax3.set_xticks(range(W))
        ax3.set_yticks(range(H))
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = fig.add_subplot(gs_cc[1])
        v_cc_sum = np.sum(board_data[53:79], axis=0)
        im4 = ax4.imshow(v_cc_sum, cmap="coolwarm", vmin=0)
        ax4.set_title("Vertical Cross-Check Density")
        ax4.set_xticks(range(W))
        ax4.set_yticks(range(H))
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # 4. Last opponent move and our last move visualization (move to gs[1,0])
        ax5 = fig.add_subplot(gs[1, 0])
        last_opp_move_plane = board_data[83]
        our_last_move_plane = board_data[84]
        im5 = ax5.imshow(last_opp_move_plane, cmap="Oranges", vmin=0, vmax=1)
        im5_our = ax5.imshow(
            our_last_move_plane, cmap="Blues", vmin=0, vmax=1, alpha=0.5
        )
        ax5.set_title("Last Moves: Opponent (orange), Ours (blue)")
        ax5.set_xticks(range(W))
        ax5.set_yticks(range(H))
        for y in range(H):
            for x in range(W):
                if last_opp_move_plane[y, x] > 0:
                    ax5.text(
                        x,
                        y,
                        "O",
                        ha="center",
                        va="center",
                        color="darkorange",
                        fontweight="bold",
                    )
                if our_last_move_plane[y, x] > 0:
                    ax5.text(
                        x,
                        y,
                        "U",
                        ha="center",
                        va="center",
                        color="blue",
                        fontweight="bold",
                    )

        # 5. Rack visualization (move to gs[1,1])
        ax6 = fig.add_subplot(gs[1, 1])
        rack_data = scalar_data[:27]
        ax6.bar(["?"] + list(string.ascii_uppercase), rack_data)
        ax6.set_title("Rack (count / 7.0)")
        ax6.set_ylim(0, 1.0)

        # 6. Unseen tiles visualization (move to gs[1,2])
        ax7 = fig.add_subplot(gs[1, 2])
        unseen_data = scalar_data[27:54]
        ax7.bar(["?"] + list(string.ascii_uppercase), unseen_data)
        ax7.set_title("Unseen Tiles (count / bagcount)")
        ax7.set_ylim(0, 1.0)

        # 7. Last move was exchange scalar features (move to gs[2,0])
        ax8 = fig.add_subplot(gs[2, 0])
        last_was_exchange = scalar_data[54:62]
        exchange_labels = ["Was Exchange"] + [f"{i} Exch" for i in range(1, 8)]
        ax8.bar(exchange_labels[: len(last_was_exchange)], last_was_exchange)
        ax8.set_title("Last Opp Move Was Exchange (scalar features)")
        ax8.set_ylim(0, 1.1)
        for i, v in enumerate(last_was_exchange):
            if v > 0:
                ax8.text(
                    i, v + 0.02, f"{v:.2f}", ha="center", color="red", fontweight="bold"
                )

        # 8. Other scalar features (move to gs[2,1])
        ax9 = fig.add_subplot(gs[2, 1])
        other_scalars = scalar_data[62:]
        expected_labels = [
            "Last Move Score",
            "Last Move Leave",
            "Tiles Remaining",
            "Spread",
        ]
        labels = expected_labels[: len(other_scalars)]

        if len(other_scalars) > 0:
            ax9.bar(labels, other_scalars)
            ax9.set_title(f"Other Features ({len(other_scalars)} values)")
            for i, v in enumerate(other_scalars):
                ax9.text(i, v + 0.02, f"{v:.3f}", ha="center")
        else:
            ax9.text(
                0.5,
                0.5,
                "No additional features available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax9.axis("off")

        # 9. Display target value if available (move to gs[2,2])
        if target is not None:
            ax10 = fig.add_subplot(gs[2, 2])
            ax10.text(
                0.5,
                0.5,
                f"Target value (1 = win, -1 = loss, 0 = draw)): {target:.1f}",
                ha="center",
                va="center",
                fontsize=16,
            )
            ax10.axis("off")

        # Adjust layout for new plots
        plt.tight_layout()

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

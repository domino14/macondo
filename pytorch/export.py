# export.py
import torch
from training import ScrabbleValueNet, C, H, W, N_SCAL

# Load the trained model
ckpt = torch.load("best-bak.pt", map_location="cpu")
net = ScrabbleValueNet()
net.load_state_dict(ckpt["model"])
net.eval()

# Create dummy inputs that match the model's input shapes
dummy_board = torch.randn(1, C, H, W)
dummy_scalars = torch.randn(1, N_SCAL)

# Export the model to ONNX
torch.onnx.export(
    net,
    (dummy_board, dummy_scalars),
    "macondo-nn.onnx",  # output file name
    input_names=["board", "scalars"],
    output_names=["win_pred"],
    dynamic_axes={
        "board": {0: "batch_size"},
        "scalars": {0: "batch_size"},
        "win_pred": {0: "batch_size"},
    },
    opset_version=12,
)

print("Model exported to macondo-nn.onnx")

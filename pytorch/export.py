# export.py
import torch
from training import ScrabbleValueNet, C, H, W, N_SCAL

# Load the trained model
ckpt = torch.load("best-bak.pt", map_location="cpu")
net = ScrabbleValueNet(ch=96, blocks=10)
net.load_state_dict(ckpt["model"])
net.eval()

# Create dummy inputs that match the model's input shapes
dummy_board = torch.randn(1, C, H, W)
dummy_scalars = torch.randn(1, N_SCAL)


# Wrap the model to return a tuple instead of a dictionary for ONNX export
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, board, scalars):
        outputs = self.model(board, scalars)
        # Return individual tensors in a definite order
        return (
            outputs["value"],
            # outputs["total_game_points"],
            # outputs["opp_bingo_prob"],
            # outputs["opp_score"],
        )


wrapped_model = ModelWrapper(net)

# Export the model to ONNX
torch.onnx.export(
    wrapped_model,
    (dummy_board, dummy_scalars),
    "macondo-nn.onnx",  # output file name
    input_names=["board", "scalars"],
    output_names=["value"],  # , "total_game_points", "opp_bingo_prob", "opp_score"],
    dynamic_axes={
        "board": {0: "batch_size"},
        "scalars": {0: "batch_size"},
        "value": {0: "batch_size"},
        # "total_game_points": {0: "batch_size"},
        # "opp_bingo_prob": {0: "batch_size"},
        # "opp_score": {0: "batch_size"},
    },
    opset_version=12,
)

print(
    "Model exported to macondo-nn.onnx with 1 output head: value"
    # ", total_game_points, opp_bingo_prob, opp_score"
)

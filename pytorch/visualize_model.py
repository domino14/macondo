from training import ScrabbleValueNet, C, H, W, N_SCAL
import torch

from torchviz import make_dot

# Instantiate your model
model = ScrabbleValueNet()

# Create dummy input matching your model's expected input shapes
dummy_board = torch.randn(1, C, H, W)  # batch size 1
dummy_scalars = torch.randn(1, N_SCAL)  # batch size 1

# Forward pass to get output
output = model(dummy_board, dummy_scalars)

# Visualize the computation graph
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = "png"
dot.render("scrabble_model_graph", view=True)  # This will save and open the PNG

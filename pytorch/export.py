# export.py - export a trained checkpoint (CNN or transformer) to ONNX.
#
#   python export.py --ckpt best-tf.pt --out macondo-nn-tf.onnx
#
# The architecture is rebuilt from the checkpoint's "arch"/"hparams" keys;
# checkpoints saved before those keys existed are assumed to be the
# 96-channel, 10-block CNN.
import argparse
from collections import Counter

import onnx
import torch

from training import build_model, load_state_dict_compat, C, H, W, N_SCAL

# Heads exported, in output order. `value` is what the bot ranks on; `spread`
# is the predicted spread change, for the simmer's equity stat. The other
# heads are training-only regularizers.
EXPORTED_HEADS = ["value", "spread"]


# Wrap the model to return a tuple instead of a dictionary for ONNX export
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, board, scalars):
        outputs = self.model(board, scalars)
        # Return individual tensors in a definite order
        return tuple(outputs[h] for h in EXPORTED_HEADS)


def load_net(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt.get("arch", "cnn")
    hparams = ckpt.get("hparams", {"ch": 96, "blocks": 10})
    net = build_model(arch, hparams)
    missing = load_state_dict_compat(net, ckpt["model"])
    if missing:
        print(f"note: heads not in checkpoint (random init): {sorted(missing)}")
    net.eval()
    if hasattr(net, "set_export_mode"):
        net.set_export_mode(True)
    return net, arch, hparams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="best-bak.pt")
    p.add_argument("--out", default="macondo-nn.onnx")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    net, arch, hparams = load_net(args.ckpt)
    print(f"loaded {args.ckpt}: arch={arch} hparams={hparams}")

    # Create dummy inputs that match the model's input shapes
    dummy_board = torch.randn(1, C, H, W)
    dummy_scalars = torch.randn(1, N_SCAL)

    wrapped_model = ModelWrapper(net)

    # Export the model to ONNX. dynamo=False pins the TorchScript exporter;
    # newer torch versions flip the default and emit a different graph.
    torch.onnx.export(
        wrapped_model,
        (dummy_board, dummy_scalars),
        args.out,
        input_names=["board", "scalars"],
        output_names=EXPORTED_HEADS,
        dynamic_axes={
            "board": {0: "batch_size"},
            "scalars": {0: "batch_size"},
            **{h: {0: "batch_size"} for h in EXPORTED_HEADS},
        },
        opset_version=args.opset,
        dynamo=False,
    )

    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    print(f"Model exported to {args.out} with outputs: {EXPORTED_HEADS}")
    print("opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
    print("ops:", dict(Counter(n.op_type for n in m.graph.node)))


if __name__ == "__main__":
    main()

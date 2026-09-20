"""
Render a model's computation graph with torchviz.

    python visualize_model.py                    # 96ch/10-block CNN, random init
    python visualize_model.py --ckpt best-tf.pt  # whatever architecture the checkpoint holds
    python visualize_model.py --arch transformer --layers 2   # small graph for reading
"""

import argparse

import torch
from torchviz import make_dot

from training import build_model, load_state_dict_compat, parse_args, C, H, W, N_SCAL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="load arch/hparams (and weights) from a checkpoint")
    p.add_argument("--out", default="scrabble_model_graph", help="output name (.png)")
    p.add_argument("--no-view", action="store_true", help="don't open the image")
    args, rest = p.parse_known_args()

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        arch = ckpt.get("arch", "cnn")
        hparams = ckpt.get("hparams", {"ch": 96, "blocks": 10})
        model = build_model(arch, hparams)
        load_state_dict_compat(model, ckpt["model"])
    else:
        # Any training.py model flags (--arch, --ch, --layers, ...) pass through.
        targs = parse_args(rest)
        arch, hparams = targs.arch, targs.hparams
        model = build_model(arch, hparams)
    print(f"arch={arch} hparams={hparams}")

    dummy_board = torch.randn(1, C, H, W)
    dummy_scalars = torch.randn(1, N_SCAL)
    output = model(dummy_board, dummy_scalars)["value"]

    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = "png"
    dot.render(args.out, view=not args.no_view)
    print(f"wrote {args.out}.png")


if __name__ == "__main__":
    main()

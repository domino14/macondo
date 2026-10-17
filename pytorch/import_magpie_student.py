"""Import a MAGPIE distilled student (manifest.json + weights.f32) as one of
our transformer checkpoints, so export.py / copy_model.sh can serve it.

    python import_magpie_student.py MODEL_DIR OUT.pt

MODEL_DIR is a MAGPIE models/<name> folder: manifest.json lists every tensor
(name, shape, float offset) in one float32 little-endian blob, weights.f32,
named as in our ScrabbleTransformerNet. Heads the student does not have
(value, opp_bingo, opp_score, the spatial heads) are zeroed: the checkpoint
is wdl-primary, so the served value is P(win) - P(loss) from heads.wdl, and
the zeroed heads export as constants. The rack-inference head (rack_head.*)
needs side inputs we do not build and is dropped.

Then checks parity/: our fp32 forward on its 64 frames against the
onnxruntime outputs it ships (value, spread; within ~1e-5).
"""
import json
import os
import sys

import numpy as np
import torch

from training import build_model


def main():
    src, out = sys.argv[1], sys.argv[2]
    man = json.load(open(os.path.join(src, "manifest.json")))
    blob = np.fromfile(os.path.join(src, "weights.f32"), dtype="<f4")
    assert blob.size == man["total_floats"], (blob.size, man["total_floats"])
    hp = man["hparams"]
    hparams = {"d_model": hp["d_model"], "layers": hp["layers"], "heads": hp["heads"],
               "ff_mult": hp["ff_mult"], "dropout": 0.0}
    net = build_model("transformer", hparams)
    sd = net.state_dict()
    got = set()
    for t in man["tensors"]:
        if t["name"].startswith("rack_head."):
            continue
        a = blob[t["offset_floats"]:t["offset_floats"] + t["count"]].reshape(t["shape"])
        if t["name"] not in sd or tuple(sd[t["name"]].shape) != a.shape:
            sys.exit(f"{t['name']} {a.shape}: no such tensor in our net "
                     f"({tuple(sd[t['name']].shape) if t['name'] in sd else 'missing'})")
        sd[t["name"]] = torch.from_numpy(a.copy())
        got.add(t["name"])
    zeroed = sorted(k for k in sd if k not in got)
    for k in zeroed:
        sd[k] = torch.zeros_like(sd[k])
    print(f"loaded {len(got)} tensors; zeroed {zeroed}")
    net.load_state_dict(sd)
    net.eval()

    par = os.path.join(src, "parity")
    if os.path.isdir(par):
        fx = json.load(open(os.path.join(par, "fixture.json")))
        ld = lambda k: torch.from_numpy(np.fromfile(os.path.join(par, fx[k]["file"]), dtype="<f4")
                                        .reshape(fx[k]["shape"]).copy())
        with torch.no_grad():
            o = net(ld("board"), ld("scalars"))
        p = torch.softmax(o["wdl"], dim=1)
        dv = (p[:, 2] - p[:, 0] - ld("value")).abs().max().item()
        ds = (o["spread"] - ld("spread")).abs().max().item()
        print(f"parity on {fx['frames']} frames: max |value diff| {dv:.2e}, max |spread diff| {ds:.2e}")
        if max(dv, ds) > 1e-4:
            sys.exit("parity FAILED")

    torch.save({"step": man.get("distill", {}).get("step", 0), "model": net.state_dict(),
                "arch": "transformer", "hparams": hparams, "primary": "wdl",
                "source": f"MAGPIE {src}", "distill": man.get("distill")}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

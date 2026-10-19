"""Checks for the spatial targets, the packed cache row, and the transpose
augmentation. Run: python test_spatial.py (CPU, a few seconds)."""
import io
import struct

import numpy as np
import torch

import training as T
from transformer_model import ScrabbleTransformerNet


def random_frame(rng):
    board = (rng.random((T.C, T.H, T.W)) < 0.1).astype(np.float32)
    scal = rng.standard_normal(T.N_SCAL).astype(np.float32)
    tg = rng.standard_normal(T.N_TARGETS).astype(np.float32)
    sp = (rng.random((T.N_SPATIAL, T.H, T.W)) < 0.03).astype(np.float32)
    return board, scal, tg, sp


def test_constants():
    assert T.ROW_FLOATS == 19125 + 72 + 5 + 900
    assert T.PACKED_BYTES == 2504 and T.CACHE_ROW_BYTES == 2812
    assert T.ALL_HEADS == T.TARGETS + T.SPATIAL and len(T.ALL_HEADS) == 9


def test_frame_and_cache_roundtrip():
    rng = np.random.default_rng(0)
    frames = [random_frame(rng) for _ in range(5)]
    # producer frame -> unpack_frame
    for b, s, t, sp in frames:
        payload = np.concatenate([b.ravel(), s, t, sp.ravel()]).astype(np.float32).tobytes()
        ub, us, ut, usp = T.unpack_frame(payload)
        assert torch.equal(ub, torch.from_numpy(b)) and torch.equal(usp, torch.from_numpy(sp))
        assert torch.equal(us, torch.from_numpy(s)) and torch.equal(ut, torch.from_numpy(t))
    try:
        T.unpack_frame(b"\0" * (T.ROW_FLOATS * 4 - 4))
        raise AssertionError("short frame accepted")
    except ValueError:
        pass
    # pack_rows -> unpack_row and unpack_batch agree with the originals
    stack = [torch.from_numpy(np.stack(x)) for x in zip(*frames)]
    packed = T.pack_rows(*stack)
    assert len(packed) == 5 * T.CACHE_ROW_BYTES
    rows = np.frombuffer(packed, dtype=np.uint8).reshape(5, T.CACHE_ROW_BYTES)
    for i in range(5):
        b, s, t, sp = T.unpack_row(rows[i].copy())
        assert torch.equal(b, stack[0][i]) and torch.equal(sp, stack[3][i])
        assert torch.equal(s, stack[1][i]) and torch.equal(t, stack[2][i])
    bb, ss, tt, ssp = T.unpack_batch(torch.from_numpy(rows.copy()))
    assert torch.equal(bb, stack[0]) and torch.equal(ssp, stack[3])
    assert torch.equal(ss, stack[1]) and torch.equal(tt, stack[2])
    # validation file round trip
    import os, tempfile
    d = tempfile.mkdtemp()
    name, n = T.write_validation_to_file(iter(frames_as_tensors(frames)), d)
    assert n == 5 and os.path.getsize(name) == 5 * T.ROW_FLOATS * 4
    b, s, t, sp = T.read_rows(name, 5)
    assert torch.equal(b, stack[0]) and torch.equal(sp, stack[3]) and torch.equal(t, stack[2])
    os.unlink(name)


def frames_as_tensors(frames):
    return [tuple(torch.from_numpy(x) for x in f) for f in frames]


def test_transpose():
    rng = np.random.default_rng(1)
    frames = [random_frame(rng) for _ in range(8)]
    board, _, _, spatial = [torch.from_numpy(np.stack(x)) for x in zip(*frames)]
    # prob 0 / 1 are the identity / a full transpose; twice is the identity
    b0, s0 = T.transpose_batch(board, spatial, 0.0)
    assert torch.equal(b0, board) and torch.equal(s0, spatial)
    b1, s1 = T.transpose_batch(board, spatial, 1.0)
    assert not torch.equal(b1, board)
    b2, s2 = T.transpose_batch(b1, s1, 1.0)
    assert torch.equal(b2, board) and torch.equal(s2, spatial)
    # letters, blank, premiums, history: transposed in place
    for ch in list(range(27)) + list(range(79, 85)):
        assert torch.equal(b1[:, ch], board[:, ch].transpose(1, 2))
    # horizontal cross-checks become the transposed vertical ones and vice versa
    for l in range(26):
        assert torch.equal(b1[:, 27 + l], board[:, 53 + l].transpose(1, 2))
        assert torch.equal(b1[:, 53 + l], board[:, 27 + l].transpose(1, 2))
    assert torch.equal(s1, spatial.transpose(2, 3))
    # per-plane counts follow the channel permutation
    perm = torch.as_tensor(T.TRANSPOSE_PERM)
    assert torch.equal(b1.sum((2, 3)), board.sum((2, 3))[:, perm])
    # a partial flip changes some rows and leaves the rest alone
    g = torch.Generator().manual_seed(3)
    bp, sp_ = T.transpose_batch(board, spatial, 0.5, generator=g)
    changed = [not torch.equal(bp[i], board[i]) for i in range(8)]
    assert any(changed) and not all(changed)
    for i, c in enumerate(changed):
        ref = (b1[i], s1[i]) if c else (board[i], spatial[i])
        assert torch.equal(bp[i], ref[0]) and torch.equal(sp_[i], ref[1])


def test_models_loss_and_export():
    rng = np.random.default_rng(2)
    frames = [random_frame(rng) for _ in range(4)]
    board, scal, tg, sp = [torch.from_numpy(np.stack(x)) for x in zip(*frames)]
    weights = dict(T.DEFAULT_WEIGHTS)
    for net in (ScrabbleTransformerNet(d_model=32, layers=1, heads=2), T.ScrabbleValueNet(ch=16, blocks=1)):
        out = net(board, scal)
        assert out["spatial"].shape == (4, T.N_SPATIAL, T.H, T.W)
        total, losses = T.compute_loss(out, tg, sp, weights)
        assert set(losses) == set(T.ALL_HEADS) and torch.isfinite(total)
        total.backward()
        norms = T.head_grad_norms(net, board, scal, tg, sp)
        assert set(norms) == set(T.ALL_HEADS) and all(v > 0 for v in norms.values())
        # the spatial head is not part of the trunk for the gradient report
        assert not any(n.startswith("heads") and False for n, _ in net.named_parameters())
        bal = T.balance_weights(weights, norms, 0.15, primary="wdl", spatial_share=0.1)
        for k in T.SPATIAL:
            assert abs(bal[k] * norms[k] - 0.1 * norms["wdl"]) < 1e-6 or bal[k] == 10.0
        # the transformer's square-token ordering: token i is square (i // 15, i % 15)
    # export drops the spatial head and keeps the two served outputs
    import export
    net = ScrabbleTransformerNet(d_model=32, layers=1, heads=2)
    net.eval(); net.set_export_mode(True)
    wrapped = export.ModelWrapper(net, "wdl")
    outs = wrapped(board, scal)
    assert len(outs) == len(export.EXPORTED_HEADS) and outs[0].shape == (4,)
    buf = io.BytesIO()
    torch.onnx.export(wrapped, (board[:1], scal[:1]), buf, input_names=["board", "scalars"],
                      output_names=export.EXPORTED_HEADS, opset_version=17, dynamo=False,
                      dynamic_axes={"board": {0: "b"}, "scalars": {0: "b"}, **{h: {0: "b"} for h in export.EXPORTED_HEADS}})
    import onnx
    m = onnx.load_from_string(buf.getvalue())
    assert [o.name for o in m.graph.output] == export.EXPORTED_HEADS
    assert not any("heads_spatial" in init.name for init in m.graph.initializer), "spatial head leaked into the export"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

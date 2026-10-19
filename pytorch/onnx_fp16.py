"""Convert an ONNX value net to fp16 for TensorRT.

TensorRT 11 builds only strongly typed networks: precision comes from the
types in the ONNX graph, not from builder flags. So engines are built from
an fp16 copy of the graph, with the inputs and outputs kept as they were
(fp32, or the uint8 board), so the Triton config is unchanged.

    python onnx_fp16.py model.onnx model.fp16.onnx

Used by onnx-to-tensorrt.py and build-engines.sh (inside the Triton
container, next to trtexec). Needs only onnx and onnxconverter-common.
"""

import sys

import onnx


def to_fp16(model):
    """Convert weights and activations to fp16, keep graph I/O as is."""
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    # An integer input (the uint8 board, export.py --board-uint8) reaches
    # the graph through a Cast to float that the converter leaves at fp32;
    # cast it straight to fp16 instead.
    int_inputs = {i.name for i in model.graph.input
                  if i.type.tensor_type.elem_type != onnx.TensorProto.FLOAT}
    for node in model.graph.node:
        if node.op_type == "Cast" and node.input[0] in int_inputs:
            for a in node.attribute:
                if a.name == "to" and a.i == onnx.TensorProto.FLOAT:
                    a.i = onnx.TensorProto.FLOAT16
    return model


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    onnx.save(to_fp16(onnx.load(src)), dst)

"""Time a TensorRT engine (.plan) alone, without Triton, at several batch
sizes, plain enqueue and as a CUDA graph.

    python bench-engine.py PLAN [BATCH ...]
"""
import sys
import time

import tensorrt as trt
import torch

DT = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
      trt.DataType.UINT8: torch.uint8, trt.DataType.INT32: torch.int32}


def main():
    plan = sys.argv[1]
    batches = [int(b) for b in sys.argv[2:]] or [1, 8, 16, 32, 50, 64, 128]
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    eng = rt.deserialize_cuda_engine(open(plan, "rb").read())
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    stream = torch.cuda.Stream()
    for b in batches:
        bufs = {}
        for n in names:
            shape = [b] + list(eng.get_tensor_shape(n))[1:]
            dt = DT[eng.get_tensor_dtype(n)]
            if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                ctx.set_input_shape(n, shape)
                t = (torch.rand(shape, device="cuda") < 0.08).to(dt) if dt == torch.uint8 else torch.rand(shape, device="cuda", dtype=dt)
            else:
                t = torch.empty(shape, device="cuda", dtype=dt)
            bufs[n] = t
            ctx.set_tensor_address(n, t.data_ptr())
        with torch.cuda.stream(stream):
            for _ in range(20):
                ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            iters = max(200, 20000 // b)
            t0 = time.perf_counter()
            for _ in range(iters):
                ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            plain = (time.perf_counter() - t0) / iters
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                ctx.execute_async_v3(stream.cuda_stream)
            for _ in range(10):
                g.replay()
            stream.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                g.replay()
            stream.synchronize()
            graph = (time.perf_counter() - t0) / iters
        print(f"batch {b:4d}: plain {plain*1e3:6.3f} ms ({b/plain:8.0f} pos/s)   graph {graph*1e3:6.3f} ms ({b/graph:8.0f} pos/s)", flush=True)


if __name__ == "__main__":
    main()

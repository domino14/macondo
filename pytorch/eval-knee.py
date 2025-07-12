import torch, time, numpy as np
from training import ScrabbleValueNet

device = "cuda"
net = ScrabbleValueNet(ch=96, blocks=10).to(device).eval()

for N in [8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280]:
    board = torch.randn(N, 85, 15, 15, device=device, dtype=torch.float32)
    scalar = torch.randn(N, 72, device=device, dtype=torch.float32)
    # warm-up
    for _ in range(5):
        net(board, scalar)
        torch.cuda.synchronize()
    t0 = time.time()
    net(board, scalar)
    torch.cuda.synchronize()
    dt = (time.time() - t0) * 1e3
    print(f"batch {N:>3}: {dt:>5.1f} ms  ({N/dt*1000:>6.0f} eval/s)")

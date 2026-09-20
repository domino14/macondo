"""
Transformer value net for Macondo.

Consumes exactly the same inputs as the CNN in training.py:
    board   : (B, 85, 15, 15)  float32 planes
    scalars : (B, 72)          float32 global features
and returns the same head dict as the CNN (see training.Heads), so the Go
producer, the binary frame format and the Triton client are shared.

Tokenization happens inside the model (254 tokens):
    [CLS] + 225 board squares + 27 tile types (rack count, unseen prob)
    + 1 game-state token (all 72 scalars).

Every reshape uses -1 for the batch dimension and literal constants
otherwise, so ONNX export emits constant Reshape nodes with a single
dynamic batch axis.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

C, H, W = 85, 15, 15
N_SQUARES = H * W  # 225
N_SCAL = 72
N_TILE_TYPES = 27  # blank + A..Z, matches rack[0:27] / unseen[27:54]


class Block(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, d, heads, ff_mult, dropout, n_layers):
        super().__init__()
        assert d % heads == 0, "d_model must be divisible by heads"
        self.d = d
        self.heads = heads
        self.hd = d // heads
        self.scale = 1.0 / math.sqrt(self.hd)
        self.export_mode = False

        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff_mult * d)
        self.fc2 = nn.Linear(ff_mult * d, d)
        self.drop = nn.Dropout(dropout)
        self.dropout_p = dropout

        # GPT-2 style init; residual projections scaled by depth.
        for lin in (self.qkv, self.fc1):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)
        for lin in (self.proj, self.fc2):
            nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.zeros_(lin.bias)

    def attention(self, x, T):
        qkv = self.qkv(x).view(-1, T, 3, self.heads, self.hd)
        q, k, v = qkv.unbind(2)  # each (B, T, H, hd)
        q = q.transpose(1, 2)  # (B, H, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.export_mode:
            # Manual path: MatMul -> Softmax -> MatMul. TensorRT fuses this
            # pattern into its MHA kernel; no dependence on the SDPA symbolic.
            att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            att = torch.softmax(att, dim=-1)
            out = torch.matmul(att, v)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p if self.training else 0.0
            )
        out = out.transpose(1, 2).reshape(-1, T, self.d)
        return self.proj(out)

    def forward(self, x, T):
        x = x + self.drop(self.attention(self.ln1(x), T))
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x


class ScrabbleTransformerNet(nn.Module):
    N_TOKENS = 1 + N_SQUARES + N_TILE_TYPES + 1  # 254

    def __init__(self, d_model=192, layers=8, heads=6, ff_mult=4, dropout=0.0):
        super().__init__()
        d = d_model
        self.d = d

        self.square_proj = nn.Linear(C, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, N_SQUARES, d))
        self.tile_proj = nn.Linear(2, d)
        self.tile_emb = nn.Parameter(torch.zeros(1, N_TILE_TYPES, d))
        self.game_proj = nn.Linear(N_SCAL, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.tile_emb, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

        self.blocks = nn.ModuleList(
            [Block(d, heads, ff_mult, dropout, layers) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 128)
        from training import Heads  # lazy: training imports this module

        self.heads = Heads(128)

    def set_export_mode(self, flag=True):
        for b in self.blocks:
            b.export_mode = flag

    def tokens(self, board, scalars):
        # 225 square tokens
        sq = board.reshape(-1, C, N_SQUARES).transpose(1, 2)  # (B, 225, 85)
        sq = self.square_proj(sq) + self.pos_emb
        # 27 tile-type tokens: (rack count, unseen draw probability)
        tiles = torch.stack(
            (scalars[:, :N_TILE_TYPES], scalars[:, N_TILE_TYPES : 2 * N_TILE_TYPES]),
            dim=-1,
        )  # (B, 27, 2)
        tiles = self.tile_proj(tiles) + self.tile_emb
        # 1 game-state token from all scalars
        game = self.game_proj(scalars).unsqueeze(1)  # (B, 1, d)
        # CLS via broadcast add (no expand on a dynamic batch dim)
        cls = self.cls + torch.zeros_like(game)
        return torch.cat([cls, sq, tiles, game], dim=1)  # (B, 254, d)

    def forward(self, board, scalars):
        x = self.tokens(board, scalars)
        T = self.N_TOKENS
        for blk in self.blocks:
            x = blk(x, T)
        cls = self.ln_f(x[:, 0])
        h = F.relu(self.fc1(cls))
        return self.heads(h)


if __name__ == "__main__":
    net = ScrabbleTransformerNet()
    n = sum(p.numel() for p in net.parameters())
    print(f"params: {n:,}")
    b, s = torch.randn(4, C, H, W), torch.randn(4, N_SCAL)
    net.eval()
    with torch.no_grad():
        y1 = net(b, s)["value"]
        net.set_export_mode(True)
        y2 = net(b, s)["value"]
    print("sdpa-vs-manual max diff:", (y1 - y2).abs().max().item())

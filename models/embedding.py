'''
里面包含：

时间步编码（timestep embedding）

feature projection

把 motion 的 78 维 → 128

把 lyrics 的 768 维 → 128
（随便，你设 128/256 都 OK）
'''

# models/embedding.py
import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    def forward(self, t):
        return self.time_mlp(t)

class CondProjection(nn.Module):
    def __init__(self, motion_dim=78*3, text_dim=768, out_dim=128):
        super().__init__()
        self.motion_proj = nn.Linear(motion_dim, out_dim)
        self.text_proj = nn.Linear(text_dim, out_dim)

    def forward(self, motion, lyrics):
        # 输入 (B, T, D)
        motion_f = self.motion_proj(motion)
        text_f = self.text_proj(lyrics)
        return motion_f, text_f

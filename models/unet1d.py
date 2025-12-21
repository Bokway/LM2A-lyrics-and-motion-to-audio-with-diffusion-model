import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention import CrossAttentionFusion
from .embedding import TimestepEmbedding

"""
一个初始版的 1D U-Net，用于扩散模型的噪声预测，支持 motion + lyrics cross-attention 条件输入，尚未进行优化，但是功能完整，不过效果以及效率还有提升空间。

This is a basic 1D U-Net for noise prediction in diffusion models, supporting motion + lyrics cross-attention conditional inputs. 
While it is functionally complete, there is still room for improvement in terms of performance and efficiency.
"""

class ResBlock1D(nn.Module):
    def __init__(self, channels, time_emb_dim, cond_dim=128, mel_dim=80, num_heads=4):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

        # timestep embedding → 加到通道维度上
        self.time_proj = nn.Linear(time_emb_dim, channels)

        # cross attention 模块（作用在 sequence 维度）
        self.cross_attn = CrossAttentionFusion(
            mel_dim=channels,  # 因为我们把 mel hidden 投成了 channels
            cond_dim=cond_dim,
            num_heads=num_heads
        )

        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

        self.act = nn.SiLU()

    def forward(self, x, t_emb, motion_f, text_f):
        """
        x: (B, C, T)
        motion_f, text_f: (B, T, cond_dim)
        """

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # 加 timestep embedding
        t_out = self.time_proj(t_emb)  # (B, C)
        h = h + t_out[:, :, None]      # broadcast → (B, C, T)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        # Cross Attention 要求输入 (B, T, C)
        h_seq = h.permute(0, 2, 1)
        h_seq = self.cross_attn(h_seq, motion_f, text_f)
        h = h_seq.permute(0, 2, 1)

        return x + h     # residual
        


class UNet1D(nn.Module):
    def __init__(self,
                 in_dim=80,
                 base_dim=128,
                 dim_mults=(1, 2, 4),
                 cond_dim=128,
                 time_emb_dim=256):

        super().__init__()

        # time embedding
        self.time_embedding = TimestepEmbedding(time_emb_dim)

        # 初始 projection：mel(80) → base_dim
        self.input_proj = nn.Conv1d(in_dim, base_dim, kernel_size=1)

        # Down blocks (store skip channel sizes to build symmetric up blocks)
        dims = [base_dim * m for m in dim_mults]
        self.downs = nn.ModuleList()
        prev_dim = base_dim
        skip_channels = []

        for dim in dims:
            self.downs.append(
                nn.ModuleList([
                    ResBlock1D(prev_dim, time_emb_dim, cond_dim=cond_dim, mel_dim=in_dim),
                    nn.Conv1d(prev_dim, dim, kernel_size=4, stride=2, padding=1)  # downsample
                ])
            )
            skip_channels.append(prev_dim)
            prev_dim = dim

        # Middle block
        self.mid = ResBlock1D(prev_dim, time_emb_dim, cond_dim=cond_dim, mel_dim=in_dim)

        # Up blocks (build symmetric to downs): upconv then resblock
        self.ups = nn.ModuleList()
        for dim, skip_ch in zip(reversed(dims), reversed(skip_channels)):
            # upconv: from current prev_dim -> dim
            self.ups.append(
                nn.ModuleList([
                    nn.ConvTranspose1d(prev_dim, dim, kernel_size=4, stride=2, padding=1),
                    ResBlock1D(dim + skip_ch, time_emb_dim, cond_dim=cond_dim, mel_dim=in_dim)
                ])
            )
            # after the resblock, output channels will be (dim + skip_ch)
            prev_dim = dim + skip_ch

        # Output projection
        self.out_proj = nn.Conv1d(prev_dim, in_dim, kernel_size=1)

    def forward(self, x, t, motion_f, text_f):
        """
        x: (B, C=80, T)
        motion_f: (B, T, 128)
        text_f:   (B, T, 128)
        """

        t_emb = self.time_embedding(t)  # (B, time_emb_dim)

        # (B, 80, T) → (B, base_dim, T)
        h = self.input_proj(x)

        skips = []
        # Down
        for resblock, down in self.downs:
            h = resblock(h, t_emb, motion_f, text_f)
            skips.append(h)
            h = down(h)

        # Middle
        h = self.mid(h, t_emb, motion_f, text_f)

        # Up
        for up, resblock in self.ups:
            # corresponding skip (reversed order)
            skip = skips.pop()

            # upsample, then match time dimension with skip and concat
            h = up(h)
            if h.size(2) != skip.size(2):
                diff = skip.size(2) - h.size(2)
                if diff > 0:
                    h = F.pad(h, (0, diff))
                else:
                    h = h[:, :, : skip.size(2)]

            h = torch.cat([h, skip], dim=1)
            h = resblock(h, t_emb, motion_f, text_f)

        return self.out_proj(h)

'''现在拥有：

一个可以被 diffusion 调用的 1D U-Net

输入：noisy mel (B, 80, T)

输出：predicted noise (B, 80, T)

使用 motion+lyrics cross-attention

完整上下采样结构

timestep embedding、ResBlock、skip connections

完全就是 "lyrics+motion → audio" 的可训练基础骨干。'''
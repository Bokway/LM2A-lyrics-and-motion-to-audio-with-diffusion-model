# models/cross_attention.py

import torch
import torch.nn as nn

# cross_attention is used to fuse mel hidden states with motion and lyrics features


class CrossAttentionFusion(nn.Module):
    """
    对 mel hidden states（或任意 seq）做 cross-attention：
        Q = mel_hidden
        K/V = motion_f, lyrics_f
    最后 concat(attn_motion, attn_lyrics) -> linear -> fused
    """
    def __init__(self, mel_dim=80, cond_dim=128, num_heads=4):
        super().__init__()

        self.attn_motion = nn.MultiheadAttention(
            embed_dim=mel_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.attn_text = nn.MultiheadAttention(
            embed_dim=mel_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 将 motion 与 lyrics 的输出拼接后融合
        self.fuse_proj = nn.Linear(mel_dim * 2, mel_dim)

        # 将 cond_dim 映射到 mel_dim，使得 KV dim 与 Q 一致
        self.motion_kv_proj = nn.Linear(cond_dim, mel_dim)
        self.text_kv_proj   = nn.Linear(cond_dim, mel_dim)

    def forward(self, mel_hidden, motion_f, text_f):
        """
        mel_hidden: (B, T, mel_dim)   例如 mel embedding 后的结果
        motion_f:   (B, T, cond_dim)
        text_f:     (B, T, cond_dim)
        """

        # 先把 motion/text 投到与 mel_dim 相同的特征维度
        motion_kv = self.motion_kv_proj(motion_f)  # (B, T, mel_dim)
        text_kv   = self.text_kv_proj(text_f)      # (B, T, mel_dim)

        # Motion Attention
        attn_motion, _ = self.attn_motion(
            query=mel_hidden,
            key=motion_kv,
            value=motion_kv
        )

        # Lyrics Attention
        attn_text, _ = self.attn_text(
            query=mel_hidden,
            key=text_kv,
            value=text_kv
        )

        # concat + projection
        fused = torch.cat([attn_motion, attn_text], dim=-1)
        fused = self.fuse_proj(fused)

        return fused


'''
mel_hidden（查询 Q）
它来自 U-Net 的当前层：
mel_hidden: (B, T, mel_dim=80)


motion_f（cond）
来自 embedding.py 的投影：
motion_f: (B, T, 128)


text_f（cond）
同样：
lyrics_f: (B, T, 128)


然后：
motion 和 lyrics 映射 → mel_dim（80）
Q（mel）分别与 motion/lyrics 做 attention
得到两个 attention 输出，concat → (B, T, 160)

线性融合 → (B, T, 80)
直接可以加回 U-Net
'''
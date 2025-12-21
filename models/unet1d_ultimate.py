"""
UNet1D_ultimate: 优化后的1D UNet架构，适配音频扩散模型（Mel谱生成）
核心改进：
1. FiLM时间步调制（Scale+Shift）增强时间步建模
2. 平滑上采样（插值+卷积）减少棋盘格伪影
3. 稀疏化Cross-Attention，平衡计算量和建模能力
4. 鲁棒的残差连接和归一化策略
5. 防止过拟合的Dropout层


UNet1D_ultimate: An optimized 1D UNet architecture tailored for audio diffusion models (Mel-spectrogram generation).
Key improvements:
1. FiLM-based timestep modulation (Scale+Shift) for enhanced temporal modeling.
2. Smooth upsampling (interpolation + convolution) to reduce checkerboard artifacts.
3. Sparse Cross-Attention to balance computational load and modeling capacity.
4. Robust residual connections and normalization strategies.
5. Dropout layers to prevent overfitting.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union
from .cross_attention import CrossAttentionFusion



def default_num_groups(channels: int) -> int:
    """
    自动选择合理的GroupNorm分组数（优先8/4/2/1）
    Args:
        channels: 输入通道数
    Returns:
        适配的分组数
    """
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class FiLMMOD(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) 调制模块
    从时间步嵌入生成scale和shift，对特征进行缩放+偏移
    """
    def __init__(self, time_emb_dim: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

    def forward(self, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_emb: 时间步嵌入 (B, time_emb_dim)
        Returns:
            scale: 缩放系数 (B, out_channels, 1)
            shift: 偏移系数 (B, out_channels, 1)
        """
        stats = self.net(t_emb)  # (B, 2*out_channels)
        scale, shift = stats.chunk(2, dim=1)
        return scale.unsqueeze(-1), shift.unsqueeze(-1)


class ResBlock1D(nn.Module):
    """
    1D残差块，集成FiLM时间调制和可选的Cross-Attention
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        cond_dim: int = 128,
        use_attn: bool = False,
        num_heads: int = 4
    ):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # 归一化层（自动适配分组数）
        self.gn1 = nn.GroupNorm(default_num_groups(in_channels), in_channels)
        self.gn2 = nn.GroupNorm(default_num_groups(out_channels), out_channels)

        # 激活函数
        self.act = nn.SiLU()

        # FiLM时间调制
        self.film = FiLMMOD(time_emb_dim, out_channels)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(p=0.1)

        # 可选Cross-Attention（条件特征融合）
        self.use_attn = use_attn
        if use_attn:
            # 延迟导入避免循环依赖
            self.cross_attn = CrossAttentionFusion(
                mel_dim=out_channels,
                cond_dim=cond_dim,
                num_heads=num_heads
            )

        # 残差连接：通道不匹配时用1x1卷积对齐
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
        motion_f: Optional[torch.Tensor] = None,
        text_f: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, in_channels, T)
            t_emb: 时间步嵌入 (B, time_emb_dim)
            motion_f: 运动特征 (B, T, cond_dim)
            text_f: 文本特征 (B, T, cond_dim)
        Returns:
            残差输出 (B, out_channels, T)
        """
        # 第一卷积分支
        h = self.gn1(x)
        h = self.act(h)
        h = self.conv1(h)

        # FiLM时间调制
        if t_emb is not None:
            scale, shift = self.film(t_emb)  # (B, out_channels, 1)
            h = h * (1 + scale) + shift

        # 第二卷积分支
        h = self.gn2(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.dropout(h)

        # Cross-Attention（条件特征融合）
        if self.use_attn and (motion_f is not None) and (text_f is not None):
            # Attention需要 (B, T, C) 格式
            h_seq = h.permute(0, 2, 1)
            h_seq = self.cross_attn(h_seq, motion_f, text_f)
            h = h_seq.permute(0, 2, 1)

        # 残差连接
        return self.skip(x) + h


class MidBlock(nn.Module):
    """
    UNet中间块：堆叠多个残差块，增强深层特征建模
    """
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        cond_dim: int = 128,
        num_blocks: int = 3,
        attn_every: int = 1,
        num_heads: int = 4
    ):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            use_attn = (i % attn_every == 0)
            blocks.append(ResBlock1D(
                in_channels=channels,
                out_channels=channels,
                time_emb_dim=time_emb_dim,
                cond_dim=cond_dim,
                use_attn=use_attn,
                num_heads=num_heads
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        motion_f: Optional[torch.Tensor] = None,
        text_f: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, channels, T)
            t_emb: 时间步嵌入 (B, time_emb_dim)
            motion_f: 运动特征 (B, T, cond_dim)
            text_f: 文本特征 (B, T, cond_dim)
        Returns:
            输出特征 (B, channels, T)
        """
        for blk in self.blocks:
            x = blk(x, t_emb, motion_f, text_f)
        return x


class UpSampleConv(nn.Module):
    """
    上采样模块：插值 + 卷积（减少棋盘格伪影）
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, in_channels, T)
        Returns:
            上采样特征 (B, out_channels, 2*T)
        """
        # 线性插值放大2倍
        x = F.interpolate(
            x,
            scale_factor=2,
            mode='linear',
            align_corners=True
        )
        # 卷积平滑
        x = self.conv(x)
        return x


class DownSampleConv(nn.Module):
    """
    下采样模块：步长卷积（高效且保留特征）
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, in_channels, T)
        Returns:
            下采样特征 (B, out_channels, T//2)
        """
        return self.conv(x)


class UNet1D_ultimate(nn.Module):
    """
    优化后的1D UNet架构，适配音频扩散模型（Mel谱生成）
    """
    def __init__(
        self,
        in_dim: int = 80,
        base_dim: int = 128,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        cond_dim: int = 128,
        time_emb_dim: int = 256,
        num_res_blocks: int = 2,
        mid_blocks: int = 3,
        attn_heads: int = 4
    ):
        super().__init__()

        # 时间步嵌入模块
        from .embedding import TimestepEmbedding
        self.time_embedding = TimestepEmbedding(time_emb_dim)

        # 输入投影：将Mel谱维度(80)映射到基础维度
        self.in_proj = nn.Conv1d(in_dim, base_dim, kernel_size=1)

        # 计算各阶段通道数
        dims = [base_dim * m for m in dim_mults]

        # 下采样阶段
        self.downs = nn.ModuleList()
        prev_dim = base_dim
        for i, dim in enumerate(dims):
            blocks = nn.ModuleList()
            for b in range(num_res_blocks):
                # 仅在每个stage的最后一个残差块启用Attention
                use_attn = (b == num_res_blocks - 1)
                blocks.append(ResBlock1D(
                    in_channels=prev_dim,
                    out_channels=dim,
                    time_emb_dim=time_emb_dim,
                    cond_dim=cond_dim,
                    use_attn=use_attn,
                    num_heads=attn_heads
                ))
                prev_dim = dim
            # 下采样层
            downsample = DownSampleConv(dim, dim, kernel_size=4, stride=2, padding=1)
            self.downs.append(nn.ModuleDict({
                'blocks': blocks,
                'down': downsample
            }))

        # 中间块（深层特征建模）
        self.mid = MidBlock(
            channels=prev_dim,
            time_emb_dim=time_emb_dim,
            cond_dim=cond_dim,
            num_blocks=mid_blocks,
            attn_every=1,
            num_heads=attn_heads
        )

        # 上采样阶段
        self.ups = nn.ModuleList()
        for dim in reversed(dims):
            # 上采样层
            upsample = UpSampleConv(prev_dim, dim)
            # 残差块列表
            blocks = nn.ModuleList()
            for b in range(num_res_blocks):
                # 仅在每个stage的第一个残差块启用Attention（拼接后）
                use_attn = (b == 0)
                # 第一个残差块输入通道=dim*2（上采样输出+skip连接）
                in_ch = dim * 2 if b == 0 else dim
                blocks.append(ResBlock1D(
                    in_channels=in_ch,
                    out_channels=dim,
                    time_emb_dim=time_emb_dim,
                    cond_dim=cond_dim,
                    use_attn=use_attn,
                    num_heads=attn_heads
                ))
            self.ups.append(nn.ModuleDict({
                'up': upsample,
                'blocks': blocks
            }))
            prev_dim = dim

        # 输出投影：映射回Mel谱维度(80)
        self.out_proj = nn.Sequential(
            nn.GroupNorm(default_num_groups(prev_dim), prev_dim),
            nn.SiLU(),
            nn.Conv1d(prev_dim, in_dim, kernel_size=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        motion_f: Optional[torch.Tensor] = None,
        text_f: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        Args:
            x: Mel谱输入 (B, in_dim, T)
            t: 时间步 (B,)
            motion_f: 运动特征 (B, T, cond_dim)
            text_f: 文本特征 (B, T, cond_dim)
        Returns:
            噪声预测输出 (B, in_dim, T)
        """
        # 时间步嵌入
        t_emb = self.time_embedding(t)  # (B, time_emb_dim)

        # 输入投影
        h = self.in_proj(x)
        skips = []

        # 下采样
        for stage in self.downs:
            for blk in stage['blocks']:
                h = blk(h, t_emb, motion_f, text_f)
            skips.append(h)
            h = stage['down'](h)

        # 中间块
        h = self.mid(h, t_emb, motion_f, text_f)

        # 上采样
        for stage in self.ups:
            # 上采样
            h = stage['up'](h)
            # 取出对应的skip连接特征
            skip = skips.pop()
            
            # 时间维度对齐（插值/下采样可能导致长度不一致）
            if h.size(2) != skip.size(2):
                diff = skip.size(2) - h.size(2)
                if diff > 0:
                    # 补零
                    h = F.pad(h, (0, diff))
                else:
                    # 截断
                    h = h[:, :, :skip.size(2)]
            
            # 拼接skip特征
            h = torch.cat([h, skip], dim=1)
            
            # 残差块处理
            for blk in stage['blocks']:
                h = blk(h, t_emb, motion_f, text_f)

        # 输出投影
        return self.out_proj(h)


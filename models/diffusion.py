# models/diffusion.py
import torch
import torch.nn as nn

class GaussianDiffusion:
    def __init__(self, model, timesteps=1000, device="cuda",dataset_mean=0.0,dataset_std=1.0):
        self.model = model
        self.T = timesteps
        self.device = device
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        # β schedule（线性）
        betas = torch.linspace(1e-4, 0.02, timesteps).to(device)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    # ----------- Forward process: q(x_t | x_0) -----------
    def q_sample(self, x0, t, noise=None):
        """
        x0: (B, C=80, T)
        t:  (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.alpha_bars[t].sqrt()             # (B,)
        sqrt_mab = (1 - self.alpha_bars[t]).sqrt()      # (B,)

        # reshape for broadcasting
        while sqrt_ab.dim() < x0.dim():
            sqrt_ab = sqrt_ab[..., None]
            sqrt_mab = sqrt_mab[..., None]

        return sqrt_ab * x0 + sqrt_mab * noise

    # ----------- Loss (predict noise) -----------
    def loss(self, x0, motion_f, text_f):
        """
        x0: (B, 80, T)
        motion_f: (B, T, 128)
        text_f:   (B, T, 128)
        """
        B = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        
        x0_normlized = (x0 - self.dataset_mean) / self.dataset_std

        x_t = self.q_sample(x0_normlized, t, noise)

        pred_noise = self.model(x_t, t, motion_f, text_f)

        return torch.mean((noise - pred_noise)**2)

    # ----------- Sampling (reverse) -----------
    @torch.no_grad()
    def p_sample(self, x_t, t, motion_f, text_f):
        """
        采样一步 x_{t-1}
        """
        # Normalize t to a tensor of shape (B,) so model and indexing behave consistently
        if not isinstance(t, torch.Tensor):
            t = torch.full((x_t.size(0),), int(t), device=x_t.device, dtype=torch.long)
        elif t.dim() == 0:
            t = t.view(1).expand(x_t.size(0)).to(x_t.device)
        betas_t = self.betas[t]
        alphas_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]

        # reshape
        while betas_t.dim() < x_t.dim():
            betas_t = betas_t[..., None]
            alphas_t = alphas_t[..., None]
            alpha_bar_t = alpha_bar_t[..., None]

        eps = self.model(x_t, t, motion_f, text_f)

        # 去噪
        x0_pred = (x_t - eps * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()

        # `t` is a tensor of shape (B,) when sampling a batch. Do a per-sample mask
        # so that when t==0 we don't add noise, otherwise add gaussian noise.
        # Create mask shaped (B, 1, 1, ... ) to broadcast to x_t
        if isinstance(t, torch.Tensor):
            mask = (t > 0).to(x_t.device)
            # expand mask to match x_t dims (B, C, T...)
            expand_shape = [mask.size(0)] + [1] * (x_t.dim() - 1)
            mask = mask.view(*expand_shape)
            noise = torch.randn_like(x_t) * mask
        else:
            # scalar int
            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

        coef1 = 1 / alphas_t.sqrt()
        coef2 = betas_t / (1 - alpha_bar_t).sqrt()

        x_prev = coef1 * (x_t - coef2 * eps) + betas_t.sqrt() * noise
        return x_prev

    @torch.no_grad()
    def sample(self, shape, motion_f, text_f):
        """
        从纯噪声生成 mel
        shape: (B, 80, T)
        """
        x = torch.randn(shape, device=self.device)

        B = shape[0]

        for t in reversed(range(self.T)):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t_batch, motion_f, text_f)

        return x
    
    
    
    # 新增DDIM采样核心函数。可以选择替换DDPM的p_sample进行采样，但尚未进行相应的调试和测试
    def ddim_sample(self, x, t, t_prev, eps, eta=0.0):
        # 处理t_prev=-1的边界情况（第一步）
        if t_prev < 0:
            alpha_t_prev = torch.tensor(1.0, device=x.device)
            alpha_bar_t_prev = torch.tensor(1.0, device=x.device)
        else:
            alpha_t_prev = self.alphas[t_prev]
            alpha_bar_t_prev = self.alpha_bars[t_prev]
        
        # 获取当前t的alpha（确保t是有效索引）
        if t < 0:
            alpha_t = torch.tensor(1.0, device=x.device)
            alpha_bar_t = torch.tensor(1.0, device=x.device)
        else:
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

        # 重塑维度适配mel形状
        alpha_t = alpha_t[..., None, None]
        alpha_t_prev = alpha_t_prev[..., None, None]
        alpha_bar_t = alpha_bar_t[..., None, None]
        alpha_bar_t_prev = alpha_bar_t_prev[..., None, None]

        # 预测x0并约束范围
        x0_pred = (x - eps * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -2.0, 2.0)  # 放宽约束，避免过度裁剪

        # DDIM核心计算（增加数值稳定性）
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
        )
        # 避免sigma_t为nan（当t_prev=-1时）
        sigma_t = torch.nan_to_num(sigma_t, nan=0.0, posinf=0.0, neginf=0.0)
        
        noise = torch.randn_like(x) if t_prev > 0 else torch.zeros_like(x)

        x_prev = (
            torch.sqrt(alpha_bar_t_prev) * x0_pred +
            torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps +
            sigma_t * noise
        )
        return x_prev, x0_pred

'''
高斯扩散
包含了前向加噪（q_sample）、损失计算（loss） 和反向采样（p_sample 和 sample） 这三个核心部分
理论上可以升级为latent diffusion等更复杂的扩散模型
但是目前DDPM的效果已经不错，暂时不考虑更复杂的模型
'''


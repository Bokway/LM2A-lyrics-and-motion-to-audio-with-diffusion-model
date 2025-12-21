"""
sampling script to generate mel spectrogram from motion and lyrics using a trained diffusion model.

"""


import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from datasetcode.dataset import MelDataset, match_len
from models.embedding import CondProjection
#from models.unet1d import UNet1D
from models.unet1d_ultimate import UNet1D_ultimate
from models.diffusion import GaussianDiffusion


def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    return ck


def build_models(cond_dim=128, base_dim=256, dim_mults=(1,2,4), time_emb_dim=256, device='cpu'):
    #unet = UNet1D(in_dim=80, base_dim=base_dim, dim_mults=tuple(dim_mults), cond_dim=cond_dim, time_emb_dim=time_emb_dim).to(device)
    unet = UNet1D_ultimate(
    in_dim=80,
    base_dim=base_dim,
    dim_mults=tuple(dim_mults),
    cond_dim=cond_dim,
    time_emb_dim=time_emb_dim,
    num_res_blocks=2,    # 和训练时的参数一致
    mid_blocks=3,        # 和训练时的参数一致
    attn_heads=4         # 和训练时的参数一致
).to(device)
    
    cond_proj = CondProjection(motion_dim=78*3, text_dim=768, out_dim=cond_dim).to(device)
    return unet, cond_proj


def sample_from_npz(npz_path, ckpt_path, out_dir, device='cpu', timesteps=1000, guidance_weight=1.0):
    os.makedirs(out_dir, exist_ok=True)

    # fallback dataset stats (will be overridden by ckpt if present)
    # not exactly the training dataset stats, but close enough
    dataset_mean = -4.63706636428833
    dataset_std = 1.8648223876953125

    # load data
    data = np.load(npz_path, allow_pickle=True)
    mel = data['mel']
    realmel = mel  # (80, T)
    motion = data['motion']
    lyrics = data['lyrics']
    sr = int(data.get('sr', 22050))
    hop = int(data.get('hop_length', 256))
    

    # determine target time length from mel (mel may be (80, T) or (T, 80))
    if mel.ndim == 2:
        if mel.shape[0] == 80:
            T = mel.shape[1]
        elif mel.shape[1] == 80:
            T = mel.shape[0]
            mel = mel.T
        else:
            # assume mel is (n_mels, T)
            T = mel.shape[1]
    else:
        raise RuntimeError('unexpected mel shape: ' + str(mel.shape))

    # build models and load ckpt
    device = torch.device(device)
    unet, cond_proj = build_models(device=device)

    ck = load_checkpoint(ckpt_path, device=device)
    # prefer EMA weights if available in checkpoint
    use_ema = False
    if 'ema_unet' in ck or 'ema_cond_proj' in ck:
        use_ema = True
    if use_ema:
        print('found EMA weights in ckpt; loading EMA for sampling')
        if 'ema_unet' in ck:
            try:
                unet.load_state_dict(ck['ema_unet'], strict=False)
            except Exception:
                print('failed loading ema_unet, falling back to normal unet')
        else:
            unet.load_state_dict(ck.get('unet', {}), strict=False)

        if 'ema_cond_proj' in ck:
            try:
                cond_proj.load_state_dict(ck['ema_cond_proj'], strict=False)
            except Exception:
                print('failed loading ema_cond_proj, falling back to normal cond_proj')
        else:
            cond_proj.load_state_dict(ck.get('cond_proj', {}), strict=False)
    else:
        # load state dicts if present
        unet.load_state_dict(ck.get('unet', {}), strict=False)
        cond_proj.load_state_dict(ck.get('cond_proj', {}), strict=False)

    # prefer dataset mean/std saved inside checkpoint when available
    if 'dataset_mean' in ck and 'dataset_std' in ck:
        try:
            dataset_mean = float(ck['dataset_mean'])
            dataset_std = float(ck['dataset_std'])
            print(f'using dataset mean/std from ckpt: {dataset_mean} {dataset_std}')
        except Exception:
            print('found dataset_mean/std in ckpt but failed to parse; using fallback constants')
    else:
        print(f'using fallback dataset mean/std: {dataset_mean} {dataset_std}')

    unet.eval(); cond_proj.eval()

    # prefer timesteps from ckpt if present, otherwise use default 1000; allow diffusion.T override
    ck_timesteps = ck.get('timesteps', None)
    default_timesteps = timesteps
    timesteps = int(ck_timesteps) if ck_timesteps is not None else default_timesteps
    diffusion = GaussianDiffusion(unet, timesteps=timesteps, device=device, dataset_mean=dataset_mean, dataset_std=dataset_std)

    # resample motion/lyrics to mel length (use interp to avoid repeating tail)
    motion_rs = match_len(motion, T, mode='interp')
    lyrics_rs = match_len(lyrics, T, mode='interp')

    # prepare condition tensors
    motion_t = torch.from_numpy(motion_rs[None].astype(np.float32)).to(device)  # (1, T, D)
    lyrics_t = torch.from_numpy(lyrics_rs[None].astype(np.float32)).to(device)

    with torch.no_grad():
        motion_f, text_f = cond_proj(motion_t, lyrics_t)  # (1, T, cond_dim)

        # sampling loop with optional classifier-free guidance
        B = 1
        x = torch.randn((B, 80, T), device=device) *1.0 # 原1.0→1.05，轻微增强初始噪声
        timesteps = diffusion.T

        # guidance weight: try to read from ckpt or default to 1.0 (no guidance)
        guidance_weight = float(ck.get('guidance_weight', guidance_weight))
        # debug: print stats at intervals
        report_interval = max(1, timesteps // 10)

        for ti, t in enumerate(reversed(range(timesteps))):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            with torch.no_grad():
                if guidance_weight <= 1.0:
                    # 无CFG/权重≤1.0时，直接预测
                    eps = unet(x, t_batch, motion_f, text_f)
                else:
                    # 优化1：更鲁棒的零条件（用mask而非全零，避免数值异常）
                    B, T_cond, D_cond = motion_f.shape
                    # 生成batch维度的零条件mask (B,1,1)，只屏蔽batch维度的条件
                    uncond_mask = torch.zeros((B, 1, 1), device=device, dtype=torch.float32)
                    motion_f_uncond = motion_f * uncond_mask
                    text_f_uncond = text_f * uncond_mask

                    # 优化2：拼接条件/无条件输入，一次前向推理（提升效率+避免两次推理的随机误差）
                    motion_f_cat = torch.cat([motion_f_uncond, motion_f], dim=0)  # (2B, T, D)
                    text_f_cat = torch.cat([text_f_uncond, text_f], dim=0)
                    x_cat = torch.cat([x, x], dim=0)  # (2B, 80, T)
                    t_batch_cat = torch.cat([t_batch, t_batch], dim=0)  # (2B,)

                    # 一次前向推理，同时得到无条件/有条件的eps
                    eps_cat = unet(x_cat, t_batch_cat, motion_f_cat, text_f_cat)
                    eps_uncond, eps_cond = torch.chunk(eps_cat, 2, dim=0)  # 拆分回(B,80,T)

                    # 优化3：加入clip防止数值爆炸（避免极端值导致的杂音）
                    eps_diff = torch.clamp(eps_cond - eps_uncond, -5.0, 5.0)
                    eps = eps_uncond + guidance_weight * eps_diff

                    # 最终clip eps，进一步保证数值稳定
                    eps = torch.clamp(eps, -10.0, 10.0)

            # print diffusion coefficients for debugging
            bt = diffusion.betas[t].item()
            at = diffusion.alphas[t].item()
            abar = diffusion.alpha_bars[t].item()
            coef1 = 1.0 / (at ** 0.5)
            coef2 = bt / ((1.0 - abar) ** 0.5)
            if (ti % report_interval) == 0 or t == 0:
                print(f"[coeff] t={t:4d} beta={bt:.6e} alpha={at:.6e} alpha_bar={abar:.6e} coef1={coef1:.6e} coef2={coef2:.6e} sqrt_beta={bt**0.5:.6e}")

            # compute x_{t-1} manually using same math as diffusion.p_sample
            betas_t = diffusion.betas[t]
            alphas_t = diffusion.alphas[t]
            alpha_bar_t = diffusion.alpha_bars[t]

            # reshape for broadcasting
            bt_r = betas_t
            at_r = alphas_t
            abar_r = alpha_bar_t
            while bt_r.dim() < x.dim():
                bt_r = bt_r[..., None]
                at_r = at_r[..., None]
                abar_r = abar_r[..., None]

            # predicted x0
            x0_pred = (x - eps * (1 - abar_r).sqrt()) / abar_r.sqrt()

            # noise mask for t>0
            mask = (t > 0)
            noise = torch.randn_like(x) if mask else torch.zeros_like(x)

            coef1 = 1.0 / at_r.sqrt()
            coef2 = bt_r / (1.0 - abar_r).sqrt()
            x_prev = coef1 * (x - coef2 * eps) + bt_r.sqrt() * noise

            x = x_prev
            
            # 在x = x_prev后添加
            if ti % 10 == 0 and t > 0:
                x = x + torch.randn_like(x) * 1e-3  # 极轻微噪声，不破坏结构但打破过拟合

            if (ti % report_interval) == 0 or t == 0:
                xt = x.detach().cpu()
                eps_np = eps.detach().cpu()
                if torch.isfinite(xt).all() and torch.isfinite(eps_np).all():
                    print(f"[sampling] step t={t:4d}  x min={xt.min().item():.6f} max={xt.max().item():.6f} mean={xt.mean().item():.6f} std={xt.std().item():.6f} | eps min={eps_np.min().item():.6f} max={eps_np.max().item():.6f} mean={eps_np.mean().item():.6f} std={eps_np.std().item():.6f}")
                else:
                    print(f"[sampling] step t={t:4d} contains non-finite values; stopping early")
                    break

        out = x  # (1,80,T)

    out = out.cpu().numpy().squeeze(0)  # (80, T)
    
    # de-normalize
    out = out * dataset_std + dataset_mean

    ''' 
    # 可选后处理代码（直接加在反归一化后）
    #### 1. 裁剪极端值（匹配真实mel的数值范围）
    real_min = realmel.min()
    real_max = realmel.max()
    out = np.clip(out, real_min - 0.1, real_max + 0.1)  # 少量余量，避免过度裁剪

    #### 2. 时间轴轻微平滑（降低高频噪声，不破坏结构）
    from scipy.ndimage import gaussian_filter1d
    out = gaussian_filter1d(out, sigma=0.5, axis=1)  # 只平滑时间轴（axis=1），sigma=0.5足够

    #### 3. 匹配真实mel的均值/std（进一步对齐分布）
    out = (out - out.mean()) * (realmel.std() / out.std()) + realmel.mean()'''  
    

    

    # save mel npz
    base = os.path.splitext(os.path.basename(npz_path))[0]
    out_npz = os.path.join(out_dir, base + '_gen.npz')
    # convert tensors to numpy for saving
    motion_f_np = motion_f.cpu().numpy() if isinstance(motion_f, torch.Tensor) else np.asarray(motion_f)
    text_f_np = text_f.cpu().numpy() if isinstance(text_f, torch.Tensor) else np.asarray(text_f)
    np.savez_compressed(out_npz, mel=out, motion=motion_rs, lyrics=lyrics_rs, motion_proj=motion_f_np, lyrics_proj=text_f_np, sr=sr, hop_length=hop)
    print('wrote', out_npz)

    # save a quick PNG of mel (transpose for display)
    png = os.path.join(out_dir, base + '_gen.png')
    plt.figure(figsize=(8,4))
    plt.imshow(out, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Generated mel')
    plt.savefig(png)
    plt.close()
    print('wrote', png)

    # save a quick PNG of real mel
    png_real =  os.path.join(out_dir, base + '_real.png')
    plt.figure(figsize=(8,4))
    plt.imshow(realmel, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Real mel')
    plt.savefig(png_real)
    plt.close()
    print('wrote', png_real)

    return out_npz


def parse_args():
    p = argparse.ArgumentParser()
    #p.add_argument('--npz',default='/mnt/mydev2/Bob/LM2ANew/testnpz/sample_00000020.npz', required=False, help='single input npz path (overrides --index)')
    p.add_argument('--npz',default='/mnt/mydev2/Bob/LM2ANew/npz_split/test/sample_00000129.npz', required=False, help='single input npz path (overrides --index)')
    p.add_argument('--index', type=int, default=0, help='index into npz dir')
    #p.add_argument('--npz_dir', default='/mnt/mydev2/Bob/LM2ANew/testnpz')
    p.add_argument('--npz_dir', default='/mnt/mydev2/Bob/LM2ANew/npz_split/test')
    p.add_argument('--ckpt', default='/mnt/mydev2/Bob/LM2ANew/checkpoints_adan/ckpt_step_10000.pt')
    p.add_argument('--out_dir', default='/mnt/mydev2/Bob/LM2ANew/testnpzwav')
    p.add_argument('--device', default='cuda:1')
    
    p.add_argument('--guidance', type=float, default=1.0, help='Classifier-free guidance weight. Default: 1.0 (no guidance)')
    p.add_argument('--steps', type=int, default=1000, help='Number of sampling steps. Default: 1000')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.npz:
        npz_path = args.npz
    else:
        files = [f for f in os.listdir(args.npz_dir) if f.endswith('.npz')]
        files = sorted(files)
        if len(files) == 0:
            raise RuntimeError('no npz in ' + args.npz_dir)
        idx = args.index % len(files)
        npz_path = os.path.join(args.npz_dir, files[idx])

    print('sampling', npz_path, '->', args.out_dir)
    sample_from_npz(npz_path, args.ckpt, args.out_dir, device=args.device, timesteps=args.steps, guidance_weight=args.guidance)



'''
before you run this script,
1.make sure you have installed the required packages
2.modify the default paths in parse_args() or provide them via command line arguments
3.make sure the parameters in parse_args() is what you want

'''
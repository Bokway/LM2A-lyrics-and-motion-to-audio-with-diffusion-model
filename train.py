#!/usr/bin/env python3
"""
Training script that uses Adan optimizer.
Supports optional AMP and grad clipping.
"""
import os
import time
import argparse
import torch
import csv

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy

from datasetcode.dataset import MelDataset
from models.embedding import CondProjection, TimestepEmbedding
#from models.unet1d import UNet1D
from models.unet1d_ultimate import UNet1D_ultimate
from models.diffusion import GaussianDiffusion
from models.adan import Adan


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def compute_dataset_stats(npz_dir, cap_files=None):
    import numpy as _np
    files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    if cap_files is not None:
        files = files[:cap_files]
    all_flat = []
    for f in files:
        p = os.path.join(npz_dir, f)
        try:
            d = _np.load(p, allow_pickle=True)
            mel = d['mel']
            if getattr(mel, 'ndim', 0) == 3:
                mel = _np.squeeze(mel, axis=0)
            if mel.shape[0] != 80 and mel.shape[1] == 80:
                mel = mel.T
            all_flat.append(mel.flatten())
        except Exception:
            continue
    if len(all_flat) == 0:
        raise RuntimeError('no mel data found in ' + str(npz_dir))
    arr = _np.concatenate(all_flat)
    return float(arr.mean()), float(arr.std())


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    ds = MelDataset(args.npz_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=MelDataset.collate_fn, num_workers=8, pin_memory=True)

    val_loader = None
    if getattr(args, 'val_npz_dir', None):
        if os.path.exists(args.val_npz_dir):
            val_ds = MelDataset(args.val_npz_dir)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=MelDataset.collate_fn, num_workers=4, pin_memory=True)

    writer = SummaryWriter(log_dir=args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, 'train_log.csv')
    csv_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(['epoch', 'step', 'train_loss', 'val_loss', 'time_seconds'])

    cond_proj = CondProjection(motion_dim=78*3, text_dim=768, out_dim=args.cond_dim).to(device)
    #unet = UNet1D(in_dim=80, base_dim=args.base_dim, dim_mults=tuple(map(int, args.dim_mults.split(','))), cond_dim=args.cond_dim, time_emb_dim=args.time_emb_dim).to(device)
    unet = UNet1D_ultimate(
        in_dim=80,
        base_dim=args.base_dim,
        dim_mults=tuple(map(int, args.dim_mults.split(','))),
        cond_dim=args.cond_dim,
        time_emb_dim=args.time_emb_dim,
        num_res_blocks=2,    # 可根据需求调整（建议先2）
        mid_blocks=3,        # 中间块数量（建议3）
        attn_heads=8         # 注意力头数（显存够可改8）
    ).to(device)
    
    diffusion = GaussianDiffusion(unet, timesteps=args.timesteps, device=device, dataset_mean=args.dataset_mean, dataset_std=args.dataset_std)

    if getattr(args, 'dataset_mean', None) is None or getattr(args, 'dataset_std', None) is None:
        try:
            print('computing dataset mean/std from', args.npz_dir)
            mean, std = compute_dataset_stats(args.npz_dir)
            diffusion.dataset_mean = mean
            diffusion.dataset_std = std
        except Exception as e:
            print('warning: failed to compute dataset stats, using defaults. error:', e)

    # Adan optimizer
    optim = Adan(list(unet.parameters()) + list(cond_proj.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # 空值处理（关闭学习率衰减）
    lr_decay_steps = []
    lr_decay_factors = []
    if args.lr_decay_steps.strip() and args.lr_decay_factors.strip():
        lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))
        lr_decay_factors = list(map(float, args.lr_decay_factors.split(',')))
        assert len(lr_decay_steps) == len(lr_decay_factors), "衰减步数和倍数数量必须一致"
        lr_decay_steps, lr_decay_factors = zip(*sorted(zip(lr_decay_steps, lr_decay_factors)))
    current_lr = args.lr  # 记录当前学习率
    decay_index = 0  # 跟踪当前该用第几个衰减参数

    scaler = None
    use_amp = bool(args.amp)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # EMA (exponential moving average) models for more stable evaluation/sampling
    ema_unet = copy.deepcopy(unet)
    ema_cond_proj = copy.deepcopy(cond_proj)
    # ensure EMA params not trainable
    for p in ema_unet.parameters():
        p.requires_grad = False
    for p in ema_cond_proj.parameters():
        p.requires_grad = False
    ema_decay = float(getattr(args, 'ema_decay', 0.9998))

    step = 0
    start_epoch = 0
    # 强制从0训练（忽略ckpt）
    if args.ckpt is not None:
        print('警告：已设置从0训练，忽略传入的ckpt参数！')
        args.ckpt = None

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        for batch in loader:
            unet.train()
            optim.zero_grad()

            mel = batch['mel'].permute(0, 2, 1).to(device)
            motion = batch['motion'].to(device)
            lyrics = batch['lyrics'].to(device)

            motion_f, text_f = cond_proj(motion, lyrics)

            # classifier-free guidance training: randomly drop condition with prob cond_drop_prob
            cond_drop_prob = float(getattr(args, 'cond_drop_prob', 0.2))  # 提升到0.2
            if cond_drop_prob > 0.0:
                B = motion_f.shape[0]
                device_local = motion_f.device
                drop_mask = (torch.rand(B, device=device_local) < cond_drop_prob).view(B, 1, 1)
                motion_f_train = motion_f * (~drop_mask).float()
                text_f_train = text_f * (~drop_mask).float()
            else:
                motion_f_train = motion_f
                text_f_train = text_f

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = diffusion.loss(mel, motion_f_train, text_f_train)
                scaler.scale(loss).backward()
                if args.grad_clip is not None:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(cond_proj.parameters()), args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss = diffusion.loss(mel, motion_f_train, text_f_train)
                loss.backward()
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(cond_proj.parameters()), args.grad_clip)
                optim.step()

            # update EMA after optimizer step
            with torch.no_grad():
                for ema_p, p in zip(ema_unet.parameters(), unet.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data * (1.0 - ema_decay))
                for ema_p, p in zip(ema_cond_proj.parameters(), cond_proj.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data * (1.0 - ema_decay))

            if step % args.log_interval == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.6f} lr {current_lr:.6f}")
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/lr', current_lr, step)  # 记录学习率
                try:
                    csv_writer.writerow([epoch, step, float(loss.item()), None, ''])
                    csv_file.flush()
                except Exception:
                    pass

            if step % args.save_interval == 0 and step > 0:
                ckpt_path = os.path.join(args.save_dir, f"ckpt_step_{step}.pt")
                save_state_dict = {
                    'step': step,
                    'epoch': epoch,
                    'unet': unet.state_dict(),
                    'cond_proj': cond_proj.state_dict(),
                    'optim': optim.state_dict(),
                    'dataset_mean': getattr(diffusion, 'dataset_mean', None),
                    'dataset_std': getattr(diffusion, 'dataset_std', None),
                    'ema_unet': ema_unet.state_dict(),
                    'ema_cond_proj': ema_cond_proj.state_dict()
                }
                save_checkpoint(save_state_dict, ckpt_path)
                print('已保存 checkpoint 到:', ckpt_path)
                
            # 空列表时不会执行衰减（关闭衰减）
            if decay_index < len(lr_decay_steps) and step >= lr_decay_steps[decay_index]:
                new_lr = current_lr * lr_decay_factors[decay_index]
                # 更新优化器的学习率
                for param_group in optim.param_groups:
                    param_group['lr'] = new_lr
                print(f"学习率衰减：{current_lr:.6f} → {new_lr:.6f}（step: {step}）")
                current_lr = new_lr
                decay_index += 1

            step += 1

        epoch_time = time.time() - t0
        val_loss_avg = None
        if val_loader is not None and (epoch + 1) % args.validate_every_epochs == 0:
            unet.eval()
            val_losses = []
            with torch.no_grad():
                for i, vbatch in enumerate(val_loader):
                    if args.val_cap_batches is not None and i >= args.val_cap_batches:
                        break
                    vmel = vbatch['mel'].permute(0, 2, 1).to(device)
                    vmotion = vbatch['motion'].to(device)
                    vlyrics = vbatch['lyrics'].to(device)
                    vmotion_f, vtext_f = cond_proj(vmotion, vlyrics)
                    vloss = diffusion.loss(vmel, vmotion_f, vtext_f)
                    val_losses.append(float(vloss.item()))
            if len(val_losses) > 0:
                val_loss_avg = float(sum(val_losses) / len(val_losses))
                print(f"epoch {epoch} 验证平均 loss: {val_loss_avg:.6f} (基于 {len(val_losses)} 批次)")
                writer.add_scalar('val/loss', val_loss_avg, step)
                writer.flush()

        print(f"epoch {epoch} 完成, 耗时: {epoch_time:.1f}s")
        try:
            last_train_loss = float(loss.item())
        except Exception:
            last_train_loss = None
        csv_writer.writerow([epoch, step, last_train_loss, val_loss_avg, round(epoch_time, 2)])
        csv_file.flush()

    final_ckpt_path = os.path.join(args.save_dir, 'ckpt_final_adan_1000epoch.pt')
    save_state_dict = {
        'step': step,
        'epoch': epoch,
        'unet': unet.state_dict(),
        'cond_proj': cond_proj.state_dict(),
        'optim': optim.state_dict(),
        'ema_unet': ema_unet.state_dict(),
        'ema_cond_proj': ema_cond_proj.state_dict(),
        'dataset_mean': getattr(diffusion, 'dataset_mean', None),
        'dataset_std': getattr(diffusion, 'dataset_std', None)
    }
    save_checkpoint(save_state_dict, final_ckpt_path)
    print('训练完成, 最终 checkpoint 已保存到:', final_ckpt_path)
    try:
        csv_file.close()
    except Exception:
        pass
    try:
        writer.close()
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz_dir', default='/lm2d/npz_split/train') # your train npz dir
    p.add_argument('--val_npz_dir', default='/lm2d/npz_split/val') # your val npz dir
    p.add_argument('--batch_size', type=int, default=16) # batch size
    p.add_argument('--lr', type=float, default=2e-4) # learning rate for Adan
    p.add_argument('--weight_decay', type=float, default=0.0001) # weight decay for Adan
    p.add_argument('--epochs', type=int, default=500) # total epochs
    p.add_argument('--device', default='cuda') # your device
    p.add_argument('--save_dir', default='/lm2d/checkpoints') # your save dir
    p.add_argument('--ckpt', default=None)
    p.add_argument('--save_interval', type=int, default=1000) 
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--cond_dim', type=int, default=128)
    p.add_argument('--base_dim', type=int, default=256)
    p.add_argument('--dim_mults', default='1,2,4')
    p.add_argument('--time_emb_dim', type=int, default=256)
    p.add_argument('--dataset_mean', type=float, default=None)
    p.add_argument('--dataset_std', type=float, default=None)
    p.add_argument('--validate_every_epochs', type=float, default=0.5)
    p.add_argument('--val_cap_batches', type=int, default=20, help='验证时最多处理的批次')
    p.add_argument('--amp', default=True, action='store_true', help='Use mixed precision')
    p.add_argument('--grad_clip', type=float, default=1.0, help='Max norm for gradient clipping')
    
    # 关闭学习率衰减（空字符串）
    p.add_argument('--lr_decay_steps', type=str, default='', 
                   help='触发学习率衰减的步数，用逗号分隔（留空关闭衰减）')
    p.add_argument('--lr_decay_factors', type=str, default='', 
                   help='每次衰减的倍数，与steps一一对应（留空关闭衰减）')
    # EMA衰减参数
    p.add_argument('--ema_decay', type=float, default=0.999, help='EMA衰减系数（默认0.9998）')
    
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('训练配置:', args)
    train(args)



"""
before you run this script,
1.make sure you have installed the required packages
2.modify the default paths in parse_args() or provide them via command line arguments
3.make sure the parameters in parse_args() is what you want

"""

#!/usr/bin/env python3
"""
模型评估脚本：量化+可视化评估生成Mel谱与真实Mel谱的差异
由于音频生成任务的特殊性，本脚本的评估指标并不完全反映主观听感，应当仅供参考。

Model evaluation script: Quantitative + visual evaluation of the difference between generated Mel spectrograms and real Mel spectrograms.
Due to the particularity of audio generation tasks, the evaluation metrics in
this script do not fully reflect subjective listening experience and should be used for reference only.

"""
import os
import argparse
import numpy as np
import torch
import random  # 随机模块
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import skimage  

# 直接导入采样脚本（sample.py在同级目录）
import sample  # 关键：复用采样逻辑


def compute_metrics(real_mel, gen_mel):
    """
    计算核心评估指标（逻辑不变）
    Args:
        real_mel: 真实Mel谱 (80, T)
        gen_mel: 生成Mel谱 (80, T)
    Returns:
        metrics: 字典，包含所有量化指标
    """
    # 确保维度一致（截断/补零）
    min_T = min(real_mel.shape[1], gen_mel.shape[1])
    real_mel = real_mel[:, :min_T]
    gen_mel = gen_mel[:, :min_T]

    # 1. MSE（均方误差）
    mse = np.mean((real_mel - gen_mel) ** 2)

    # 2. SSIM（结构相似性，需归一化到0-1）
    global_min = real_mel.min()
    global_max = real_mel.max()
    if global_max - global_min < 1e-6:
        global_min = min(real_mel.min(), gen_mel.min())
        global_max = max(real_mel.max(), gen_mel.max())
    real_norm = (real_mel - global_min) / (global_max - global_min + 1e-8)
    gen_norm = (gen_mel - global_min) / (global_max - global_min + 1e-8)
    real_norm = np.clip(real_norm, 0.0, 1.0)
    gen_norm = np.clip(gen_norm, 0.0, 1.0)
    skimage_version = tuple(map(int, skimage.__version__.split('.')[:2]))
    if skimage_version >= (0, 19):
        # 新版本（skimage≥0.19）：显式指定通道轴为0
        ssim_score = ssim(
            real_norm, 
            gen_norm,
            data_range=1.0,          # 归一化后数值范围0-1
            channel_axis=0,          # 80个频带是通道轴（axis=0）
            win_size=7,              # 局部窗口大小（奇数，≤80）
            sigma=1.5,               # 高斯核标准差
            use_sample_covariance=False,
            gaussian_weights=True    # 用高斯权重更贴合Mel谱的连续特征
        )
    else:
        # 旧版本兼容（skimage<0.19）：用multichannel=True
        ssim_score = ssim(
            real_norm, 
            gen_norm,
            data_range=1.0,
            multichannel=True,       # 多通道（80个频带）
            win_size=7,
            sigma=1.5,
            use_sample_covariance=False,
            gaussian_weights=True
        )
    ssim_score = np.clip(ssim_score, 0.0, 1.0)  # 确保分数在0-1之间


    # 3. 帧级余弦相似度（平均）
    cos_sim = []
    for t in range(min_T):
        real_frame = real_mel[:, t].reshape(1, -1)
        gen_frame = gen_mel[:, t].reshape(1, -1)
        sim = cosine_similarity(real_frame, gen_frame)[0][0]
        cos_sim.append(sim)
    avg_cos_sim = np.mean(cos_sim)

    # 4. 统计分布误差
    real_mean = np.mean(real_mel)
    gen_mean = np.mean(gen_mel)
    mean_error = abs(real_mean - gen_mean)
    
    real_std = np.std(real_mel)
    gen_std = np.std(gen_mel)
    std_error = abs(real_std - gen_std)

    # 5. 信噪比（SNR）：适配[0,1]尺度Mel谱
    # SNR公式：10*log10(信号方差 / 噪声方差) = 10*log10(真实谱方差 / MSE)
    real_var = np.var(real_mel)  # 真实Mel谱的方差（信号功率）
    if real_var < 1e-8:  # 防护：真实谱无波动（全0）
        snr = 0.0
    else:
        snr = 10 * np.log10(real_var / (mse + 1e-8))  # MSE是噪声功率

    return {
        'mse': round(mse, 6),
        'ssim': round(ssim_score, 6),
        'avg_cos_sim': round(avg_cos_sim, 6),
        'mean_error': round(mean_error, 6),
        'std_error': round(std_error, 6),
        'snr': round(snr, 6)
    }


def visualize_metrics(metrics, save_path):
    """可视化指标对比（修复颜色溢出问题）"""
    plt.figure(figsize=(10, 6))
    keys = list(metrics.keys())
    values = list(metrics.values())
    
    # 调整颜色（越优的指标颜色越绿）
    colors = []
    for k, v in zip(keys, values):
        if k == 'mse' or k == 'mean_error' or k == 'std_error':
            # 越小越好：0→绿色，越大→红色
            norm_v = min(max(v / 2.0, 0.0), 1.0)  # 归一化到0-1（分母放大到2避免溢出）
            colors.append((norm_v, 1 - norm_v, 0))
        else:
            # 越大越好：1→绿色，越小→红色
            norm_v = min(max(v, 0.0), 1.0)  # 强制裁剪到0-1
            colors.append((1 - norm_v, norm_v, 0))
    
    plt.bar(keys, values, color=colors)
    plt.title('Mel Spectrogram Generation Metrics')  
    plt.ylabel('Value')
    plt.grid(axis='y', alpha=0.3)
    
    # 标注数值
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, str(v), ha='center')
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_mel_pair(real_mel, gen_mel, save_path):
    """可视化真实/生成Mel谱对比（逻辑不变）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 真实Mel谱
    im1 = ax1.imshow(real_mel, aspect='auto', origin='lower')
    ax1.set_title('Real Mel Spectrogram')
    fig.colorbar(im1, ax=ax1)
    
    # 生成Mel谱
    im2 = ax2.imshow(gen_mel, aspect='auto', origin='lower')
    ax2.set_title('Generated Mel Spectrogram')
    fig.colorbar(im2, ax=ax2)
    
    plt.xlabel('Time Frames')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def assess_single_sample(npz_path, ckpt_path, out_dir, device='cuda:1'):
    """评估单个样本（保留临时目录，批量评估后统一删除）"""
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(npz_path))[0]

    # 1. 加载真实数据
    data = np.load(npz_path, allow_pickle=True)
    real_mel = data['mel']
    # 调整Mel谱维度（确保是80, T）
    if real_mel.shape[1] == 80 and real_mel.shape[0] != 80:
        real_mel = real_mel.T

    # 2. 调用sample.py生成Mel谱（复用采样逻辑）
    # 临时修改sample的输出目录为当前样本的临时目录
    temp_out_dir = os.path.join(out_dir, f"temp_{base_name}")
    os.makedirs(temp_out_dir, exist_ok=True)
    
    # 构造sample的参数（和手动运行时一致）
    sample_args = argparse.Namespace(
        npz=npz_path,
        index=0,
        npz_dir="",
        ckpt=ckpt_path,
        out_dir=temp_out_dir,
        device=device,
        guidance=1.0,
        steps=1000
    )
    
    # 调用sample的采样函数（复用逻辑）
    gen_npz_path = sample.sample_from_npz(
        npz_path=sample_args.npz,
        ckpt_path=sample_args.ckpt,
        out_dir=sample_args.out_dir,
        device=sample_args.device,
        timesteps=sample_args.steps,
        guidance_weight=sample_args.guidance
    )

    # 3. 加载生成的Mel谱
    gen_data = np.load(gen_npz_path, allow_pickle=True)
    gen_mel = gen_data['mel']
    # 调整生成Mel谱维度（和真实Mel对齐）
    if gen_mel.shape[1] == 80 and gen_mel.shape[0] != 80:
        gen_mel = gen_mel.T

    # 4. 计算指标
    metrics = compute_metrics(real_mel, gen_mel)
    print(f"【{base_name}】评估指标：")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 5. 保存结果
    # 保存指标到txt
    metrics_path = os.path.join(out_dir, f"{base_name}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"样本名称: {base_name}\n")
        f.write("="*50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # 可视化Mel谱对比
    mel_vis_path = os.path.join(out_dir, f"{base_name}_mel_pair.png")
    visualize_mel_pair(real_mel, gen_mel, mel_vis_path)

    # 可视化指标
    metric_vis_path = os.path.join(out_dir, f"{base_name}_metrics.png")
    visualize_metrics(metrics, metric_vis_path)

    # 复制生成的Mel谱到当前目录（方便查看）
    gen_mel_dst = os.path.join(out_dir, f"{base_name}_gen_mel.npz")
    import shutil
    shutil.copy(gen_npz_path, gen_mel_dst)

    # ========== 注释/删除删除临时目录的代码 ==========
    # 不再单个删除，批量评估完成后统一删
    # shutil.rmtree(temp_out_dir)

    return metrics, temp_out_dir  # 返回临时目录路径


def assess_batch(npz_dir, ckpt_path, out_dir, device='cuda:1', max_samples=None, random_sample=True, random_seed=42):
    """批量评估多个样本（支持随机选取样本）"""
    all_metrics = []
    temp_dirs = []  # 记录所有临时目录
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    
    # ========== 随机选取样本逻辑 ==========
    if random_sample and len(npz_files) > 0:
        # 设置随机种子（保证结果可复现）
        random.seed(random_seed)
        np.random.seed(random_seed)
        # 随机打乱样本列表
        random.shuffle(npz_files)
        print(f"✅ 已随机打乱样本列表（随机种子：{random_seed}）")
    
    # 限制最大样本数（随机后取前N个）
    if max_samples and max_samples < len(npz_files):
        npz_files = npz_files[:max_samples]
        print(f" 选取 {max_samples} 个样本进行评估")
    else:
        print(f" 评估全部 {len(npz_files)} 个样本")

    print(f"\n开始批量评估 {len(npz_files)} 个样本...")
    for i, npz_file in enumerate(npz_files):
        print(f"\n[{i+1}/{len(npz_files)}] 评估 {npz_file}")
        npz_path = os.path.join(npz_dir, npz_file)
        metrics, temp_dir = assess_single_sample(npz_path, ckpt_path, out_dir, device)  # 接收临时目录
        all_metrics.append(metrics)
        temp_dirs.append(temp_dir)  # 记录临时目录

    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = round(np.mean([m[key] for m in all_metrics]), 6)

    # 保存平均指标（记录随机相关信息）
    avg_metrics_path = os.path.join(out_dir, "average_metrics.txt")
    with open(avg_metrics_path, 'w') as f:
        f.write(f"批量评估样本数: {len(npz_files)}\n")
        f.write(f"是否随机选取: {random_sample}\n")
        f.write(f"随机种子: {random_seed}\n")
        f.write("="*50 + "\n")
        f.write("平均指标:\n")
        for k, v in avg_metrics.items():
            f.write(f"{k}: {v}\n")

    # 可视化平均指标
    avg_metric_vis_path = os.path.join(out_dir, "average_metrics.png")
    visualize_metrics(avg_metrics, avg_metric_vis_path)

    # ========== 批量评估完成后，统一删除所有临时目录 ==========
    import shutil
    import time
    for temp_dir in temp_dirs:
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                shutil.rmtree(temp_dir)
                print(f"已删除临时目录: {temp_dir}")
                break
            except OSError as e:
                retry_count += 1
                time.sleep(0.5)
                if retry_count >= max_retries:
                    print(f"警告：临时目录 {temp_dir} 删除失败，请手动执行 rm -rf {temp_dir}")

    print("\n" + "="*50)
    print("批量评估完成！平均指标：")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v}")
    print(f"结果保存到: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='/mnt/mydev2/Bob/LM2ANew/checkpoints_adan/newunetsplit250epoch1217/ckpt_step_19000.pt',help='ckpt path') # your ckpt path
    p.add_argument('--npz_dir', default='/mnt/mydev2/Bob/LM2ANew/npz_split/test',help='test npz dir') # your npz dir
    p.add_argument('--out_dir', default='/mnt/mydev2/Bob/LM2ANew/ceshi/final19000',help='output dir') # your output dir
    p.add_argument('--device', default='cuda:1',help='device') # your device
    p.add_argument('--max_samples', type=int, default=10,help='max samples (None for all)')
    
    p.add_argument('--no-random', action='store_false', dest='random_sample', default=True,
                   help='disable random sampling (enabled by default)')
    p.add_argument('--seed', type=int, default=100, help='random seed')
    
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assess_batch(
        args.npz_dir, 
        args.ckpt, 
        args.out_dir, 
        args.device, 
        args.max_samples,
        args.random_sample,  # 传递随机开关
        args.seed            # 传递随机种子
    )


    """
    before you run this script, 
    1.make sure you have installed the required packages
    2.modify the default paths in parse_args() or provide them via command line arguments
    
    """
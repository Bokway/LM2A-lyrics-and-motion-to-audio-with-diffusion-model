import numpy as np
import torch
import soundfile as sf
import bigvgan
import os

def npz_to_wav(npz_file_path, model, device):
    """
    单个npz文件转wav文件的核心函数
    :param npz_file_path: 输入的npz文件路径
    :param model: 预加载的BigVGAN模型
    :param device: 计算设备（cuda或cpu）
    """
    try:
        # 1. 加载npz文件并提取mel谱和采样率
        data = np.load(npz_file_path, allow_pickle=True)
        mel = data["mel"].astype(np.float32)  # (80, T)
        print(f"处理文件: {npz_file_path}，mel形状: {mel.shape}")

        # 2. 准备mel张量（移至指定设备，增加batch维度）
        mel_tensor = torch.tensor(mel)[None].to(device)  # (1, 80, T)
        sr = int(data["sr"])  # 从npz中提取采样率

        # 3. BigVGAN声码器推理（生成波形）
        with torch.inference_mode():  # 关闭梯度计算，节省内存并提升速度
            wav_gen = model(mel_tensor)  # (1, 1, T)

        # 4. 处理波形数据（去除多余维度，转cpu和numpy格式）
        wav = wav_gen.squeeze().cpu().numpy()  # 挤压维度为 (T,)

        # 5. 构造wav保存路径（同目录、同名，后缀改为wav）
        wav_file_path = os.path.splitext(npz_file_path)[0] + ".wav"

        # 6. 保存wav文件
        sf.write(wav_file_path, wav.astype("float32"), sr)
        print(f"成功保存: {wav_file_path}")
        return True

    except Exception as e:
        print(f"处理文件 {npz_file_path} 失败！错误信息: {str(e)}")
        return False

def batch_process_npz(folder_path):
    """
    批量处理文件夹下所有npz文件
    :param folder_path: 目标文件夹路径
    """
    # 1. 校验文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在！")
        return

    # 2. 检查GPU可用性，自动选择计算设备
    print("正在检查GPU可用性...")
    if torch.cuda.is_available():
        # 清除CUDA缓存，避免之前的错误影响
        torch.cuda.empty_cache()
        # 获取可用的GPU数量和名称
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        device = torch.device("cuda:0")
        print(f"检测到 {gpu_count} 个GPU可用，使用 {gpu_name} (cuda:0)")
    else:
        device = torch.device("cpu")
        print("未检测到可用GPU，将使用CPU进行计算（速度会较慢）")

    # 3. 预加载BigVGAN模型（仅加载一次，提升批量处理效率）
    print("正在加载BigVGAN模型...")
    try:
        model = bigvgan.BigVGAN.from_pretrained(
            'nvidia/bigvgan_22khz_80band',
            use_cuda_kernel=False
        )
        model.remove_weight_norm()
        model = model.eval().to(device)  # 移至自动选择的设备
        print("BigVGAN模型加载完成！")
    except Exception as e:
        print(f"模型加载失败！错误信息: {str(e)}")
        return

    # 4. 遍历文件夹下所有文件，筛选npz文件进行处理
    npz_file_count = 0
    processed_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 筛选后缀为.npz的文件（不区分大小写）
            if file.lower().endswith(".npz"):
                npz_file_count += 1
                # 构造完整的npz文件路径
                npz_full_path = os.path.join(root, file)
                # 处理单个npz文件
                if npz_to_wav(npz_full_path, model, device):
                    processed_count += 1

    # 5. 输出批量处理统计信息
    print("=" * 50)
    print(f"批量处理完成！")
    print(f"文件夹中共有npz文件: {npz_file_count} 个")
    print(f"成功处理并生成wav文件: {processed_count} 个")
    if npz_file_count != processed_count:
        print(f"处理失败: {npz_file_count - processed_count} 个")

# 移除argparse相关代码，直接定义目标文件夹
if __name__ == "__main__":
    target_folder = "/lm2d/testval"  # 固定文件夹路径
    #target_folder = "/lm2d/testorigin"  # 固定文件夹路径
    batch_process_npz(target_folder)